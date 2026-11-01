"""Skill definitions, loading, and the skill path through the agent loop."""

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsforge.agent.roles import ROLES
from gsforge.agent.skills import SkillDefinitionError, SkillRegistry, load_skills
from gsforge.agent.skills.loader import BUILTIN_DIR

from .test_agent_loop import ScriptedLLM, _agent, _evaluate, _first_text, _last_result, _tool, broken_state  # noqa: F401


def _write_skill(root: Path, dirname: str, frontmatter: str, body: str = "Do the thing.") -> Path:
    d = root / dirname
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return d


def test_builtin_skills_load_and_validate():
    reg = load_skills()
    assert len(reg) >= 1  # a broken built-in skill would raise here, i.e. at worker startup
    skill = reg.get("close-open-base")
    assert skill.stages == ("geometry",) and skill.ref == "close-open-base@1.0.0"
    assert "reference.py" in skill.resources()
    assert [s.name for s in reg.for_stages(["texture"])] == []  # stage scoping


def test_catalog_is_deterministic_and_lists_only_stage_skills(tmp_path):
    _write_skill(tmp_path, "b-skill", "name: b-skill\ndescription: second\nstages: [geometry]\nversion: 1.0.0")
    _write_skill(tmp_path, "a-skill", "name: a-skill\ndescription: first\nstages: [geometry, texture]\nversion: 1.0.0")
    _write_skill(tmp_path, "c-skill", "name: c-skill\ndescription: other stage\nstages: [physics]\nversion: 1.0.0")
    reg = SkillRegistry([tmp_path])
    cat = reg.catalog(["geometry"])
    assert cat == reg.catalog(["geometry"])  # byte-stable -> prompt cache keeps hitting
    assert cat.index("- a-skill: first") < cat.index("- b-skill: second") and "c-skill" not in cat
    assert reg.catalog(["align"]) is None


@pytest.mark.parametrize("frontmatter, error", [
    ("name: other\ndescription: x\nstages: [geometry]\nversion: 1.0.0", "directory name"),
    ("name: bad-skill\ndescription: x\nstages: [geometri]\nversion: 1.0.0", "stages"),
    ("name: bad-skill\ndescription: ''\nstages: [geometry]\nversion: 1.0.0", "description"),
    ("name: bad-skill\ndescription: x\nstages: [geometry]\nversion: v1", "version"),
])
def test_invalid_definitions_fail_fast(tmp_path, frontmatter, error):
    _write_skill(tmp_path, "bad-skill", frontmatter)
    with pytest.raises(SkillDefinitionError, match=error):
        SkillRegistry([tmp_path])


def test_missing_frontmatter_and_duplicates_are_rejected(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    (d / "SKILL.md").write_text("just text", encoding="utf-8")
    with pytest.raises(SkillDefinitionError, match="frontmatter"):
        SkillRegistry([tmp_path])
    other = tmp_path / "other"
    _write_skill(other, "close-open-base", "name: close-open-base\ndescription: x\nstages: [geometry]\nversion: 9.0.0")
    with pytest.raises(SkillDefinitionError, match="duplicate"):
        SkillRegistry([BUILTIN_DIR, other])


def test_template_is_a_valid_skill_once_renamed(tmp_path):
    shutil.copytree(BUILTIN_DIR / "_template", tmp_path / "my-skill-name")
    assert SkillRegistry([tmp_path]).get("my-skill-name").version == "0.1.0"


async def test_agent_loads_skill_and_its_reference_script_fixes_open_base(broken_state):  # noqa: F811
    svc, st = broken_state
    job = SimpleNamespace(job_id="t", inputs=None)
    report = await _evaluate(svc, "geometry", job, st)

    def mesh_id(messages):
        return re.search(r'"mesh": \{\s*"id": "(art_[0-9a-f]{12})"', _first_text(messages)).group(1)

    def skill_files(messages):
        return _last_result(messages)["files"]

    steps = [
        lambda m: [_tool(1, "load_skill", {"name": "close-open-base"})],
        lambda m: [_tool(2, "run_python", {"code": skill_files(m)["reference.py"], "inputs": {"mesh": mesh_id(m)},
                                           "purpose": "cap the open base per close-open-base"})],
        lambda m: [_tool(3, "submit", {"artifacts": {"mesh": _last_result(m)["outputs"]["capped"]},
                                       "summary": "base closed at the contact plane", "confidence": 0.9})],
    ]
    llm = ScriptedLLM(steps)

    async def evaluate(candidate, generated):
        trial = st.model_copy(deep=True)
        trial.current = dict(candidate)
        return await _evaluate(svc, "geometry", job, trial)

    out = await _agent(svc, llm).repair(stage="geometry", role=ROLES["geometry"], state=st, report=report,
                                        evaluate=evaluate)
    assert "## Skills" in llm.system and "- close-open-base:" in llm.system
    assert "load_skill" in llm.seen_tools
    assert out.accepted, out.reason
    assert out.skills_used == ["close-open-base@1.0.0"]
    assert 0.1 < out.generated_ratio < 0.3  # the box bottom: 0.06 of 0.22 m^2, reported by the script
    trace = [json.loads(line) for line in svc.store.local_path(out.trace).read_text().splitlines()]
    assert trace[-1]["skills_used"] == ["close-open-base@1.0.0"]


async def test_skill_outside_stage_is_refused(broken_state):  # noqa: F811
    svc, st = broken_state
    steps = [lambda m: [_tool(1, "load_skill", {"name": "close-open-base"})],
             lambda m: [_tool(2, "escalate", {"reason": json.dumps(_last_result_raw(m))})]]
    llm = ScriptedLLM(steps)
    report = await _evaluate(svc, "texture", SimpleNamespace(job_id="t"), st)
    out = await _agent(svc, llm).repair(stage="texture", role=ROLES["texture"], state=st, report=report,
                                        evaluate=None)
    assert "load_skill" not in llm.seen_tools  # no texture skills -> tool not offered
    assert out.escalated and "not available" in out.reason and out.skills_used == []


def _last_result_raw(messages):
    return messages[-1]["content"][0]["content"][0]["text"]
