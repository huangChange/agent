"""The repair agent: observe -> think -> act -> verify, bounded by a budget.

Design points
* The agent edits nothing in place. It produces candidate artifacts; the pipeline decides.
* `submit` triggers the pipeline's *independent* evaluator. A failing submission is fed back and the
  loop continues; a passing one ends the loop. The agent can't declare success on its own.
* Every request, response and tool call is appended to a trace that is stored as an artifact and
  linked from the stage record (debugging, audits, eval datasets).
* Manual loop rather than the SDK tool runner: we interleave our own evaluation, budget checks and
  trace persistence between turns, and tool execution is dispatched to remote workers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..core.config import Settings
from ..core.errors import SandboxError
from ..core.models import ArtifactKind, ArtifactRef, InstanceState, QAReport
from ..runtime.runner import ToolRunner
from ..storage.artifacts import ArtifactStore
from ..tools.base import ToolSpec, resolve_artifact_ids
from ..tools.registry import tools_for_stage
from .llm import LLM, Usage
from .roles import Role
from .sandbox import Sandbox
from .skills import HarvestLibrary, SkillRegistry

log = logging.getLogger("gsforge.agent")
Evaluate = Callable[[dict[str, ArtifactRef], float | None], Awaitable[QAReport]]


# ---------------------------------------------------------------- built-in agent tools


class RunPythonInput(BaseModel):
    code: str = Field(description="complete Python script following the sandbox contract")
    inputs: dict[str, str] = Field(default_factory=dict, description="name -> artifact id to copy into ./inputs")
    purpose: str = Field(description="one sentence: what this script changes or measures, and why")


class EvaluateInput(BaseModel):
    artifacts: dict[str, str] = Field(description="role -> artifact id, e.g. {'mesh': 'art_...'}; roles not "
                                                  "given keep their current artifact")


class ViewInput(BaseModel):
    images: list[str] = Field(description="image artifact ids to look at", max_length=4)


class SubmitInput(BaseModel):
    artifacts: dict[str, str] = Field(description="role -> artifact id of the final candidate")
    summary: str = Field(description="root cause, what was changed, and what (if anything) was synthesized")
    confidence: float = Field(ge=0, le=1)


class EscalateInput(BaseModel):
    reason: str = Field(description="why this cannot be fixed automatically, precise enough for a human")


class LoadSkillInput(BaseModel):
    name: str = Field(description="skill name from the Skills catalog in the system prompt")


BUILTIN_DESCRIPTIONS = {
    "load_skill": "Load the full instructions (and reference files) of a skill from the Skills catalog.",
    "run_python": "Run a Python script in the isolated sandbox (contract in the system prompt). Returns exit "
                  "code, stdout/stderr tail, metrics.json and the ids of files written to ./outputs.",
    "evaluate": "Run this stage's quality gates on a candidate (independent evaluator). Returns pass/fail, "
                "score and failing checks. Use before submit.",
    "view": "Look at up to 4 image artifacts (previews, textures).",
    "submit": "Submit the final candidate. The pipeline re-evaluates it; if it fails you get the report back "
              "and can continue.",
    "escalate": "Stop and hand this instance to human review, with a precise reason.",
}
BUILTIN_INPUTS: dict[str, type[BaseModel]] = {"run_python": RunPythonInput, "evaluate": EvaluateInput,
                                               "view": ViewInput, "submit": SubmitInput,
                                               "escalate": EscalateInput, "load_skill": LoadSkillInput}


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    s = model.model_json_schema()
    s.pop("title", None)
    return s


# ---------------------------------------------------------------- state


@dataclass
class Budget:
    max_turns: int
    max_tool_calls: int
    max_cost_usd: float
    max_wall_s: float
    turns: int = 0
    tool_calls: int = 0
    started: float = field(default_factory=time.monotonic)

    def exhausted(self, cost: float) -> str | None:
        if self.turns >= self.max_turns:
            return f"turn limit {self.max_turns}"
        if self.tool_calls >= self.max_tool_calls:
            return f"tool call limit {self.max_tool_calls}"
        if cost >= self.max_cost_usd:
            return f"cost limit ${self.max_cost_usd:.2f}"
        if time.monotonic() - self.started >= self.max_wall_s:
            return f"wall clock limit {self.max_wall_s:.0f}s"
        return None


class Workspace:
    """Artifact ids the agent has seen, plus the metrics of the call that produced each one."""

    def __init__(self, current: dict[str, ArtifactRef]):
        self.refs: dict[str, ArtifactRef] = {}
        self.produced_metrics: dict[str, dict[str, float]] = {}
        self.current = dict(current)
        for ref in current.values():
            self.add(ref)

    def add(self, ref: ArtifactRef, metrics: dict[str, float] | None = None) -> str:
        self.refs[ref.short_id] = ref
        if metrics:
            self.produced_metrics[ref.short_id] = metrics
        return ref.short_id

    def get(self, art_id: str) -> ArtifactRef:
        try:
            return self.refs[art_id]
        except KeyError:
            raise KeyError(f"unknown artifact id {art_id}") from None

    def candidate(self, mapping: dict[str, str]) -> dict[str, ArtifactRef]:
        out = dict(self.current)
        for role, art_id in mapping.items():
            out[role] = self.get(art_id)
        return out

    def describe(self) -> dict[str, Any]:
        return {role: {"id": ref.short_id, "kind": ref.kind.value, **{k: v for k, v in ref.meta.items()
                                                                       if isinstance(v, (int, float, str))}}
                for role, ref in self.current.items()}


@dataclass
class RepairOutcome:
    accepted: bool
    escalated: bool
    reason: str
    artifacts: dict[str, ArtifactRef]
    report: QAReport | None
    generated_ratio: float | None
    summary: str = ""
    cost_usd: float = 0.0
    turns: int = 0
    trace: ArtifactRef | None = None
    code_refs: list[ArtifactRef] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)  # name@version, recorded in lineage


@dataclass
class _Run:
    """Per-repair mutable state shared by concurrently executing tool calls."""

    code_refs: list[ArtifactRef] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    allowed_skills: frozenset[str] = frozenset()


def _get(block: Any, key: str, default: Any = None) -> Any:
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def _jsonable(block: Any) -> Any:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    return str(block)


# ---------------------------------------------------------------- agent


class RepairAgent:
    def __init__(self, *, llm: LLM, runner: ToolRunner, store: ArtifactStore, settings: Settings,
                 sandbox: Sandbox, harvest: HarvestLibrary, skills: SkillRegistry):
        self.llm = llm
        self.runner = runner
        self.store = store
        self.settings = settings
        self.sandbox = sandbox
        self.harvest = harvest
        self.skills = skills

    def _tool_specs(self, role: Role) -> dict[str, ToolSpec]:
        specs: dict[str, ToolSpec] = {}
        for stage in (*role.stages, "*"):
            for spec in tools_for_stage(stage):
                specs[spec.name] = spec
        return dict(sorted(specs.items()))

    def _tool_defs(self, specs: dict[str, ToolSpec], has_skills: bool) -> list[dict[str, Any]]:
        defs = [s.agent_schema() for s in specs.values()]
        for name in ("evaluate", "view", "submit", "escalate", "run_python", "load_skill"):
            if name == "run_python" and not self.settings.agent.allow_codegen:
                continue
            if name == "load_skill" and not has_skills:
                continue
            defs.append({"name": name, "description": BUILTIN_DESCRIPTIONS[name],
                         "input_schema": _schema(BUILTIN_INPUTS[name])})
        return defs

    def _image_block(self, ref: ArtifactRef) -> dict[str, Any]:
        data = base64.standard_b64encode(self.store.local_path(ref).read_bytes()).decode("ascii")
        return {"type": "image", "source": {"type": "base64", "media_type": ref.media_type, "data": data}}

    async def repair(self, *, stage: str, role: Role, state: InstanceState, report: QAReport,
                     evaluate: Evaluate) -> RepairOutcome:
        cfg = self.settings.agent
        budget = Budget(cfg.max_turns, cfg.max_tool_calls, cfg.max_cost_usd, cfg.max_wall_s)
        usage = Usage()
        ws = Workspace(state.current)
        specs = self._tool_specs(role)
        skill_stages = (stage, *role.stages)
        available = self.skills.for_stages(skill_stages)
        tools = self._tool_defs(specs, has_skills=bool(available))
        catalog = self.skills.catalog(skill_stages)
        system = role.system_prompt() + (f"\n\n{catalog}" if catalog else "")
        trace: list[dict[str, Any]] = []
        run = _Run(allowed_skills=frozenset(s.name for s in available))
        model = self.settings.llm.model

        def cost() -> float:
            return usage.cost_usd(self.settings.llm.prices, model)

        # ---- initial observation
        intro: list[dict[str, Any]] = [{"type": "text", "text": self._task_text(stage, state, report, ws)}]
        for ref in report.previews[:2]:
            ws.add(ref)
            intro.append(self._image_block(ref))
        messages: list[dict[str, Any]] = [{"role": "user", "content": intro}]
        trace.append({"event": "start", "stage": stage, "role": role.name, "report": report.summary()})

        outcome: RepairOutcome | None = None
        nudged = False
        try:
            while outcome is None:
                if why := budget.exhausted(cost()):
                    outcome = RepairOutcome(False, False, f"budget exhausted: {why}", {}, None, None)
                    break
                budget.turns += 1
                resp = await self.llm.turn(system=system, messages=messages, tools=tools, model=model,
                                           effort=role.effort)
                usage.add(_get(resp, "usage"), model)
                content = _get(resp, "content", [])
                stop = _get(resp, "stop_reason")
                trace.append({"event": "response", "turn": budget.turns, "stop_reason": stop,
                              "content": [_jsonable(b) for b in content], "cost_usd": round(cost(), 4)})
                messages.append({"role": "assistant", "content": content})

                if stop == "refusal":
                    outcome = RepairOutcome(False, True, "model declined the request", {}, None, None)
                    break
                tool_uses = [b for b in content if _get(b, "type") == "tool_use"]
                if stop == "max_tokens" and tool_uses:
                    # a truncated tool input may parse as a valid-looking partial object: never run it
                    messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": _get(b, "id"), "is_error": True,
                         "content": "output was truncated at max_tokens; this call was not executed. "
                                    "Retry with a smaller input (shorter script, fewer artifacts)."}
                        for b in tool_uses]})
                    continue
                if not tool_uses:
                    if nudged:
                        outcome = RepairOutcome(False, True, "agent stopped without submit/escalate", {}, None, None)
                        break
                    nudged = True
                    messages.append({"role": "user", "content": "Continue: finish by calling submit with your "
                                                                "best candidate, or escalate."})
                    continue

                budget.tool_calls += len(tool_uses)
                results = await asyncio.gather(*(self._execute(b, specs, ws, stage, state, evaluate, run)
                                                 for b in tool_uses))
                blocks = []
                for b, (block, terminal) in zip(tool_uses, results):
                    blocks.append(block)
                    trace.append({"event": "tool", "name": _get(b, "name"), "input": _get(b, "input"),
                                  "is_error": block.get("is_error", False),
                                  "result": [c.get("text") for c in block["content"] if c.get("type") == "text"]
                                  if isinstance(block["content"], list) else block["content"]})
                    if terminal is not None and outcome is None:
                        outcome = terminal
                messages.append({"role": "user", "content": blocks})
        finally:
            trace.append({"event": "end", "turns": budget.turns, "tool_calls": budget.tool_calls,
                          "cost_usd": round(cost(), 4), "usage": usage.__dict__, "skills_used": run.skills_used,
                          "outcome": None if outcome is None else {"accepted": outcome.accepted,
                                                                   "escalated": outcome.escalated,
                                                                   "reason": outcome.reason}})
            trace_ref = self.store.put_bytes("\n".join(json.dumps(e, default=str) for e in trace).encode("utf-8"),
                                             ArtifactKind.LOG, ".jsonl", {"stage": stage})
        outcome.cost_usd, outcome.turns, outcome.trace = cost(), budget.turns, trace_ref
        outcome.code_refs, outcome.skills_used = run.code_refs, run.skills_used
        return outcome

    def _task_text(self, stage: str, state: InstanceState, report: QAReport, ws: Workspace) -> str:
        parts = [
            f"Stage `{stage}` failed its quality gates for instance `{state.instance.instance_id}`"
            f" (label: {state.instance.label or 'unknown'}).",
            "Gate report:\n```json\n" + json.dumps(report.summary(), indent=2) + "\n```",
            "Current artifacts (role -> id):\n```json\n" + json.dumps(ws.describe(), indent=2) + "\n```",
        ]
        facts = {k: v for k, v in state.facts.items() if k in ("up", "transform", "bbox_size_m", "material_class",
                                                                "support_plane_height", "role")}
        if facts:
            parts.append("Known facts:\n```json\n" + json.dumps(facts, indent=2, default=str) + "\n```")
        examples = self.harvest.examples(report)
        if examples:
            parts.append("Scripts that fixed the same failure signature before (reference, adapt as needed):")
            for ex in examples:
                parts.append(f"- {ex['summary']}\n```python\n{ex['code']}\n```")
        if report.previews:
            parts.append("Attached: renders of the current mesh and of the gaussians from the same cameras.")
        return "\n\n".join(parts)

    async def _execute(self, block: Any, specs: dict[str, ToolSpec], ws: Workspace, stage: str,
                       state: InstanceState, evaluate: Evaluate,
                       run: _Run) -> tuple[dict[str, Any], RepairOutcome | None]:
        tool_id, name, raw = _get(block, "id"), _get(block, "name"), _get(block, "input") or {}

        def result(payload: Any, is_error: bool = False, images: list[ArtifactRef] | None = None):
            text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for ref in (images or [])[:2]:
                content.append(self._image_block(ref))
            out = {"type": "tool_result", "tool_use_id": tool_id, "content": content}
            if is_error:
                out["is_error"] = True
            return out

        try:
            if name in BUILTIN_INPUTS:
                args = BUILTIN_INPUTS[name].model_validate(raw)
            elif name in specs:
                resolved = resolve_artifact_ids(raw, ws.get)
                args = specs[name].params_model.model_validate(resolved)
            else:
                return result(f"unknown tool '{name}'", True), None
        except (ValidationError, KeyError, TypeError) as e:
            # covers schema violations and the silently-truncated inputs eager streaming can produce
            return result({"INVALID_INPUT": str(e)}, True), None

        try:
            if name == "escalate":
                return result("escalated"), RepairOutcome(False, True, args.reason, {}, None, None)
            if name == "load_skill":
                if args.name not in run.allowed_skills:
                    return result(f"skill '{args.name}' is not available for this stage; "
                                  f"available: {sorted(run.allowed_skills)}", True), None
                skill = self.skills.get(args.name)
                if skill.ref not in run.skills_used:
                    run.skills_used.append(skill.ref)
                return result({"name": skill.name, "version": skill.version, "instructions": skill.body,
                               "files": skill.resources()}), None
            if name == "view":
                refs = [ws.get(i) for i in args.images]
                return result(f"{len(refs)} image(s) attached", images=refs), None
            if name == "evaluate":
                cand = ws.candidate(args.artifacts)
                rep = await evaluate(cand, self._generated(ws, args.artifacts))
                return result(rep.summary()), None
            if name == "submit":
                cand = ws.candidate(args.artifacts)
                gen = self._generated(ws, args.artifacts)
                rep = await evaluate(cand, gen)
                if rep.passed:
                    return result({"accepted": True, **rep.summary()}), RepairOutcome(
                        True, False, "passed independent evaluation", cand, rep, gen, args.summary)
                return result({"accepted": False, "note": "submission failed independent evaluation; "
                                                          "continue or escalate", **rep.summary()}), None
            if name == "run_python":
                inputs = {k: ws.get(v) for k, v in args.inputs.items()}
                try:
                    sb = await self.sandbox.run(args.code, inputs)
                except SandboxError as e:
                    return result(f"rejected by sandbox policy: {e}", True), None
                if sb.code_ref:
                    run.code_refs.append(sb.code_ref)
                ids = {k: ws.add(v, sb.metrics) for k, v in sb.outputs.items()}
                payload = {"exit_code": sb.exit_code, "timed_out": sb.timed_out, "outputs": ids,
                           "metrics": sb.metrics, "stdout": sb.stdout, "stderr": sb.stderr}
                return result(payload, sb.exit_code != 0), None

            rec = await self.runner.run(name, args, job_id="agent", instance_id=state.instance.instance_id)
            if rec.error:
                return result({"error": rec.error}, True), None
            r = rec.result
            ids = {k: ws.add(v, r.metrics) for k, v in r.outputs.items()}
            payload = {"outputs": ids, "metrics": r.metrics, "facts": r.facts, "warnings": r.warnings,
                       "cache_hit": rec.cache_hit}
            for ref in r.previews:
                ws.add(ref)
            return result(payload, images=r.previews), None
        except KeyError as e:
            return result(str(e), True), None
        except Exception as e:  # tool bugs must not kill the loop; the model sees the error and adapts
            log.exception("agent tool %s failed", name)
            return result({"error": f"{type(e).__name__}: {e}"}, True), None

    @staticmethod
    def _generated(ws: Workspace, mapping: dict[str, str]) -> float | None:
        for art_id in mapping.values():
            g = ws.produced_metrics.get(art_id, {}).get("generated_ratio")
            if g is not None:
                return float(g)
        return None
