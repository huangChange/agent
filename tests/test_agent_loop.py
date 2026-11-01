"""Agent loop behaviour with a scripted LLM (no network): tool dispatch, sandboxed codegen, independent
evaluation on submit, input validation, budget limits, trace persistence."""

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from gsforge.agent.loop import RepairAgent
from gsforge.agent.roles import ROLES
from gsforge.agent.sandbox import Sandbox, static_tripwire
from gsforge.agent.skills import HarvestLibrary, load_skills
from gsforge.core.models import InstanceState, ObjectInstance
from gsforge.io.gaussians import gaussians_from_points
from gsforge.pipeline.factory import build_services
from gsforge.pipeline.stages import _evaluate

FIX_SCRIPT = """
import json, trimesh
m = trimesh.load('inputs/mesh.ply', force='mesh')
trimesh.repair.fill_holes(m)
m.export('outputs/fixed.ply')
json.dump({'watertight': float(m.is_watertight)}, open('outputs/metrics.json', 'w'))
print('filled holes, watertight =', m.is_watertight)
"""


def _usage():
    return SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
                           cache_creation_input_tokens=0)


def _tool(i, name, inp):
    return {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": inp}


class ScriptedLLM:
    """Plays a fixed repair strategy, reading ids out of the conversation like a model would."""

    def __init__(self, steps):
        self.steps = steps
        self.calls = 0
        self.seen_tools: list[str] = []
        self.system = ""

    async def turn(self, *, system, messages, tools, model=None, effort=None):
        self.seen_tools = [t["name"] for t in tools]
        self.system = system
        step = self.steps[self.calls]
        self.calls += 1
        content = step(messages)
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, usage=_usage())

    async def structured(self, **kw):
        raise AssertionError("not used")


def _first_text(messages):
    return messages[0]["content"][0]["text"]


def _last_result(messages):
    return json.loads(messages[-1]["content"][0]["content"][0]["text"])


@pytest.fixture()
def broken_state(settings):
    svc = build_services(settings, inline=True, use_llm=False)
    box = trimesh.creation.box(extents=(0.3, 0.2, 0.1))
    pts, fid = trimesh.sample.sample_surface(box, 20000, seed=0)
    g = svc.store.put_file(_write_g(settings, pts, box.face_normals[fid]), "gaussians")
    keep = np.flatnonzero(box.face_normals[:, 2] > -0.5)
    open_box = box.submesh([keep], append=True)
    p = settings.runtime.workdir + "/open.ply"
    open_box.export(p)
    m = svc.store.put_file(p, "mesh")
    st = InstanceState(instance=ObjectInstance(instance_id="obj-0", segment_id="s", gaussians=g),
                       current={"gaussians": g, "mesh": m})
    return svc, st


def _write_g(settings, pts, nrm):
    from pathlib import Path

    from gsforge.io.gaussians import write_gaussians

    p = Path(settings.runtime.workdir)
    p.mkdir(parents=True, exist_ok=True)
    write_gaussians(gaussians_from_points(pts, np.full((len(pts), 3), 0.5), nrm), p / "g.ply")
    return p / "g.ply"


def _agent(svc, llm):
    from pathlib import Path

    sandbox = Sandbox(svc.settings.agent.sandbox, svc.store, Path(svc.settings.runtime.workdir) / "sbx")
    return RepairAgent(llm=llm, runner=svc.runner, store=svc.store, settings=svc.settings, sandbox=sandbox,
                       harvest=HarvestLibrary(svc.store), skills=load_skills())


async def test_agent_repairs_with_codegen_and_passes_independent_qa(broken_state):
    svc, st = broken_state
    job = SimpleNamespace(job_id="t", inputs=None)
    report = await _evaluate(svc, "geometry", job, st)
    assert not report.passed and "watertight" in [c.name for c in report.hard_failures]

    def mesh_id(messages):
        return re.search(r'"mesh": \{\s*"id": "(art_[0-9a-f]{12})"', _first_text(messages)).group(1)

    steps = [
        lambda m: [{"type": "text", "text": "Inspect first."}, _tool(1, "mesh_report", {"mesh": mesh_id(m)})],
        lambda m: [_tool(2, "run_python", {"code": FIX_SCRIPT, "inputs": {"mesh": mesh_id(m)},
                                           "purpose": "close the open bottom"})],
        lambda m: [_tool(3, "submit", {"artifacts": {"mesh": _last_result(m)["outputs"]["fixed"]},
                                       "summary": "open contact region; closed with fan fill", "confidence": 0.9})],
    ]
    llm = ScriptedLLM(steps)

    async def evaluate(candidate, generated):
        trial = st.model_copy(deep=True)
        trial.current = dict(candidate)
        return await _evaluate(svc, "geometry", job, trial)

    out = await _agent(svc, llm).repair(stage="geometry", role=ROLES["geometry"], state=st, report=report,
                                        evaluate=evaluate)
    assert out.accepted, out.reason
    assert out.report.passed and out.turns == 3 and len(out.code_refs) == 1
    assert {"run_python", "submit", "repair_mesh", "render_previews"} <= set(llm.seen_tools)
    trace = [json.loads(line) for line in svc.store.local_path(out.trace).read_text().splitlines()]
    assert trace[0]["event"] == "start" and trace[-1]["outcome"]["accepted"] is True
    assert out.cost_usd > 0


async def test_invalid_input_is_reported_not_executed(broken_state):
    svc, st = broken_state
    steps = [lambda m: [_tool(1, "repair_mesh", {"mesh": "art_000000000000"})],
             lambda m: [_tool(2, "escalate", {"reason": "cannot continue"})]]
    llm = ScriptedLLM(steps)
    report = await _evaluate(svc, "geometry", SimpleNamespace(job_id="t"), st)

    async def evaluate(c, g):
        raise AssertionError("should not evaluate")

    out = await _agent(svc, llm).repair(stage="geometry", role=ROLES["geometry"], state=st, report=report,
                                        evaluate=evaluate)
    assert out.escalated and not out.accepted


async def test_budget_stops_the_loop(broken_state):
    svc, st = broken_state
    svc.settings.agent.max_turns = 2

    def stall(m):
        return [_tool(len(m), "mesh_report", {"mesh": st.current["mesh"].short_id})]

    llm = ScriptedLLM([stall] * 5)
    report = await _evaluate(svc, "geometry", SimpleNamespace(job_id="t"), st)
    out = await _agent(svc, llm).repair(stage="geometry", role=ROLES["geometry"], state=st, report=report,
                                        evaluate=None)
    assert not out.accepted and "turn limit" in out.reason and llm.calls == 2


def test_sandbox_tripwire():
    assert static_tripwire("import socket")
    assert static_tripwire("import os\nos.system('ls')")
    assert not static_tripwire("import numpy as np\nprint(np.ones(3))")
