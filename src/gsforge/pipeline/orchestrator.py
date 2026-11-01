"""Job orchestration.

scene ──prepare──▶ up vector, cleaned segments
      ──split────▶ instances (support surfaces patched)
      ──per instance, concurrently────────────────────────────────────────────────────────────
          geometry ▶ align ▶ texture ▶ material ▶ physics ▶ export
          each stage:  run deterministic tools ─▶ QA gates ─pass─▶ next
                                                       └fail─▶ repair agent ─▶ independent re-QA
                                                                   └still failing─▶ needs_review
      ──scene manifest (instance poses in the capture frame, grades, costs)

State is checkpointed after every stage; re-running a job resumes where it stopped, and the
content-addressed step cache makes already-computed tool calls free.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..agent.roles import role_for_stage
from ..agent.skills import HarvestLibrary
from ..core.errors import ToolError
from ..core.models import (
    AssetManifest,
    AssetModel,
    Grade,
    InstanceState,
    JobSpec,
    Joint,
    Lineage,
    Link,
    ObjectInstance,
    QAReport,
    StageRecord,
    StageStatus,
    utcnow,
)
from ..exporters.files import export_glb, export_obj_with_texture, export_ply
from ..exporters.sim import write_mjcf, write_urdf
from ..qa.gates import evaluate_gates, grade
from .services import Services
from .stages import STAGE_ORDER, STAGES, StageOutput, _evaluate, world_pose

log = logging.getLogger("gsforge.pipeline")


@dataclass
class JobResult:
    job_id: str
    instances: list[InstanceState]
    delivery_dir: Path
    cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def grades(self) -> dict[str, str]:
        return {s.instance.instance_id: (s.grade.value if s.grade else "none") for s in self.instances}


class Pipeline:
    def __init__(self, svc: Services):
        self.svc = svc
        self.harvest = HarvestLibrary(svc.store)

    # ------------------------------------------------------------ checkpoints
    def _ckpt_key(self, job: JobSpec, instance_id: str) -> str:
        return f"jobs/{job.job_id}/instances/{instance_id}.json"

    def _save(self, job: JobSpec, st: InstanceState) -> None:
        self.svc.store.put_named(self._ckpt_key(job, st.instance.instance_id), st.model_dump_json(indent=2).encode())

    def _load_instances(self, job: JobSpec) -> list[InstanceState] | None:
        raw = self.svc.store.get_named(f"jobs/{job.job_id}/instances.json")
        if raw is None:
            return None
        states = []
        for iid in json.loads(raw):
            data = self.svc.store.get_named(self._ckpt_key(job, iid))
            if data is None:
                return None
            states.append(InstanceState.model_validate_json(data))
        return states

    # ------------------------------------------------------------ entry point
    async def run(self, job: JobSpec) -> JobResult:
        warnings: list[str] = []
        states = self._load_instances(job)
        if states is None:
            states = await self._prepare_and_split(job, warnings)
            for st in states:
                self._save(job, st)
            self.svc.store.put_named(f"jobs/{job.job_id}/instances.json",
                                     json.dumps([s.instance.instance_id for s in states]).encode())
        else:
            log.info("resuming job %s with %d instances", job.job_id, len(states))

        sem = asyncio.Semaphore(self.svc.settings.runtime.max_concurrent_instances)

        async def guarded(st: InstanceState) -> InstanceState:
            async with sem:
                return await self._run_instance(job, st)

        done = await asyncio.gather(*(guarded(s) for s in states), return_exceptions=True)
        finished: list[InstanceState] = []
        for st, res in zip(states, done):
            if isinstance(res, BaseException):
                log.error("instance %s crashed: %s", st.instance.instance_id, res, exc_info=res)
                warnings.append(f"{st.instance.instance_id}: {type(res).__name__}: {res}")
                st.grade = Grade.REJECT
                finished.append(st)
            else:
                finished.append(res)
        out_dir = Path(self.svc.settings.export.out_dir) / job.inputs.scene_id
        self._write_scene_manifest(job, finished, out_dir)
        cost = sum(r.cost_usd for s in finished for r in s.stages.values())
        return JobResult(job.job_id, finished, out_dir, cost, warnings)

    # ------------------------------------------------------------ scene level
    async def _prepare_and_split(self, job: JobSpec, warnings: list[str]) -> list[InstanceState]:
        svc, jid = self.svc, job.job_id
        cams = job.inputs.capture.cameras if job.inputs.capture else None
        up_res = await svc.call("estimate_up", jid, "scene", scene_gaussians=job.inputs.scene_gaussians,
                                hint=job.inputs.up_axis, cameras=cams)
        up = up_res.facts["up"]
        states: list[InstanceState] = []
        for seg in job.inputs.segments:
            clean = await svc.call("clean_gaussians", jid, seg.segment_id, gaussians=seg.gaussians)
            split = await svc.call("split_instances", jid, seg.segment_id, gaussians=clean.outputs["gaussians"],
                                   up=tuple(up))
            report = evaluate_gates("split", {**split.metrics,
                                              "unassigned_ratio": split.metrics["unassigned"] / max(
                                                  clean.metrics["output_count"], 1)},
                                    svc.settings.qa.get("split", {}))
            if not report.passed:
                warnings.append(f"segment {seg.segment_id}: split gates failed {report.summary()['failures']}")
            instances = split.facts["instances"]
            has_supported = any(i["role"] == "supported" for i in instances)
            for k, info in enumerate(instances):
                g_ref = split.outputs[info["key"]]
                facts: dict[str, Any] = {"up": up, "role": info["role"]}
                generated = 0.0
                if info["role"] == "support" and has_supported and split.facts["support_plane_height"] is not None:
                    patch = await svc.call("patch_support_surface", jid, seg.segment_id, gaussians=g_ref,
                                           plane_height=split.facts["support_plane_height"], up=tuple(up))
                    g_ref = patch.outputs["gaussians"]
                    generated = patch.metrics["generated_ratio"]
                    facts["support_plane_height"] = split.facts["support_plane_height"]
                # undersides are never observed: close them at the contact height (the support plane for
                # objects resting on one, the object's own lowest points otherwise: legs on the floor)
                base_h = split.facts["support_plane_height"] if info["role"] == "supported" else None
                base = await svc.call("close_contact_base", jid, seg.segment_id, gaussians=g_ref, up=tuple(up),
                                      base_height=base_h)
                g_ref = base.outputs["gaussians"]
                generated += base.metrics["generated_ratio"]
                # the segment's label belongs to its main object; supported objects get classified later
                label = seg.label_hint if info["role"] in ("support",) or (len(instances) == 1) else None
                inst = ObjectInstance(instance_id=f"{seg.segment_id}-{k}", segment_id=seg.segment_id, label=label,
                                      gaussians=g_ref, bbox_min=tuple(info["bbox_min"]),
                                      bbox_max=tuple(info["bbox_max"]))
                st = InstanceState(instance=inst, current={"gaussians": g_ref}, facts=facts,
                                   generated_ratio=generated)
                st.stages["split"] = StageRecord(stage="split", status=StageStatus.PASSED if report.passed
                                                 else StageStatus.NEEDS_REVIEW, qa=report, finished_at=utcnow())
                states.append(st)
        for st in states:  # supported_by links for the scene manifest
            if st.facts["role"] == "supported":
                sup = next((s for s in states if s.instance.segment_id == st.instance.segment_id
                            and s.facts["role"] == "support"), None)
                st.instance.supported_by = sup.instance.instance_id if sup else None
        return states

    # ------------------------------------------------------------ instance level
    async def _run_instance(self, job: JobSpec, st: InstanceState) -> InstanceState:
        for stage in STAGE_ORDER:
            rec = st.stages.get(stage)
            if rec and rec.status in (StageStatus.PASSED, StageStatus.REPAIRED, StageStatus.SKIPPED,
                                      StageStatus.NEEDS_REVIEW):
                continue
            policy = self.svc.settings.stage_policy(stage)
            if not policy.enabled:
                st.stages[stage] = StageRecord(stage=stage, status=StageStatus.SKIPPED)
                continue
            ok = await self._run_stage(job, st, stage)
            self._save(job, st)
            if not ok:
                break
        reports = [r.qa for r in st.stages.values() if r.qa is not None]
        failed = any(r.status == StageStatus.FAILED for r in st.stages.values())
        review = any(r.status == StageStatus.NEEDS_REVIEW for r in st.stages.values())
        st.grade = Grade.REJECT if failed else Grade.C if review else grade(reports, st.generated_ratio)
        if not failed:
            self._export(job, st)
        self._save(job, st)
        return st

    async def _run_stage(self, job: JobSpec, st: InstanceState, stage: str) -> bool:
        rec = StageRecord(stage=stage, started_at=utcnow())
        st.stages[stage] = rec
        try:
            out: StageOutput = await STAGES[stage](self.svc, job, st)
        except ToolError as e:
            rec.status, rec.error, rec.finished_at = StageStatus.FAILED, str(e), utcnow()
            log.warning("stage %s failed for %s: %s", stage, st.instance.instance_id, e)
            return False
        self._apply(st, out)
        rec.outputs = {k: v for k, v in out.current.items()}
        report = await _evaluate(self.svc, stage, job, st)
        if stage == "texture":
            report = await self._critic_checks(job, st, report)
        rec.qa = report
        if report.passed:
            rec.status, rec.finished_at = StageStatus.PASSED, utcnow()
            return True

        policy = self.svc.settings.stage_policy(stage)
        role = role_for_stage(stage)
        use_agent = self.svc.agent is not None and role is not None and self.svc.settings.agent.enabled
        if policy.on_fail == "agent" and use_agent:
            report.previews = await self._previews(job, st)
            score_before = report.score

            async def evaluate(candidate, generated):
                trial = st.model_copy(deep=True)
                trial.current = dict(candidate)
                if generated is not None:
                    trial.generated_ratio = generated
                return await _evaluate(self.svc, stage, job, trial)

            outcome = await self.svc.agent.repair(stage=stage, role=role, state=st, report=report, evaluate=evaluate)
            rec.agent_trace, rec.cost_usd, rec.skills_used = outcome.trace, outcome.cost_usd, outcome.skills_used
            if outcome.accepted and outcome.report is not None:
                st.current = dict(outcome.artifacts)
                if outcome.generated_ratio is not None:
                    st.generated_ratio = outcome.generated_ratio
                rec.qa, rec.status = outcome.report, StageStatus.REPAIRED
                rec.outputs = dict(outcome.artifacts)
                self.harvest.record(report, outcome.code_refs, score_before, outcome.report.score, outcome.summary,
                                   job.job_id, st.instance.instance_id)
                rec.finished_at = utcnow()
                return True
            rec.error = outcome.reason
        if policy.on_fail == "fail":
            rec.status = StageStatus.FAILED
            rec.finished_at = utcnow()
            return False
        # keep going with the best deterministic result so reviewers see a complete asset
        rec.status, rec.finished_at = StageStatus.NEEDS_REVIEW, utcnow()
        return True

    @staticmethod
    def _apply(st: InstanceState, out: StageOutput) -> None:
        st.current.update(out.current)
        st.facts.update(out.facts)
        if out.generated_ratio is not None:
            st.generated_ratio = max(st.generated_ratio, out.generated_ratio)
        if out.warnings:
            st.facts.setdefault("warnings", []).extend(out.warnings)

    async def _previews(self, job: JobSpec, st: InstanceState) -> list:
        rec = await self.svc.runner.run("render_previews", {"mesh": st.current.get("mesh"),
                                                            "gaussians": st.current["gaussians"],
                                                            "views": self.svc.settings.agent.preview_views},
                                        job_id=job.job_id, instance_id=st.instance.instance_id)
        return rec.result.previews if rec.result else []

    async def _critic_checks(self, job: JobSpec, st: InstanceState, report: QAReport) -> QAReport:
        """Independent VLM review; adds soft checks and fills in label / material when unknown."""
        if self.svc.critic is None:
            return report
        previews = await self._previews(job, st)
        if len(previews) < 2:
            return report
        try:
            verdict, usage = await self.svc.critic.review(previews[0], previews[1], label_hint=st.instance.label)
        except Exception as e:  # the critic is advisory: never fail a stage because it is unavailable
            log.warning("critic unavailable for %s: %s", st.instance.instance_id, e)
            return report
        st.facts.setdefault("material_class", verdict.material_class)
        st.facts["critic"] = verdict.model_dump()
        if not st.instance.label:
            st.instance.label = verdict.category
        rules = self.svc.settings.qa.get("critic", {})
        extra = evaluate_gates("critic", verdict.metrics(), rules)
        report.checks += extra.checks
        report.previews = previews
        return report

    # ------------------------------------------------------------ delivery
    def _export(self, job: JobSpec, st: InstanceState) -> None:
        svc, inst = self.svc, st.instance
        out = Path(svc.settings.export.out_dir) / job.inputs.scene_id / inst.instance_id
        out.mkdir(parents=True, exist_ok=True)
        ctx = svc.local_ctx(job.job_id, inst.instance_id)
        mesh = ctx.load_mesh(st.current["mesh"])
        mat = st.facts.get("material", {})
        files: dict[str, str] = {}
        formats = svc.settings.export.formats
        if "glb" in formats:
            export_glb({inst.instance_id: mesh}, out / "asset.glb", mat.get("roughness", 0.8), mat.get("metallic", 0.0))
            files["glb"] = "asset.glb"
        if "ply" in formats:
            export_ply(mesh, out / "asset.ply")
            files["ply"] = "asset.ply"
        if "gaussians_ply" in formats:
            (out / "gaussians.ply").write_bytes(svc.store.local_path(st.current["gaussians"]).read_bytes())
            files["gaussians_ply"] = "gaussians.ply"
        if {"urdf", "mjcf"} & set(formats):
            sim = out / "sim"
            sim.mkdir(exist_ok=True)
            written = export_obj_with_texture(mesh, sim / "visual.obj")
            texture = next((p.name for p in written if p.suffix.lower() == ".png"), None)
            col_files = []
            for key in sorted(k for k in st.current if k.startswith("collision_")):
                cm = ctx.load_mesh(st.current[key])
                p = sim / f"{key}.obj"
                p.write_text(cm.export(file_type="obj"), encoding="utf-8")
                col_files.append(p.name)
            link = Link(name="base", visual_mesh=st.current["mesh"], mass=st.facts.get("mass", 1.0),
                        center_of_mass=tuple(st.facts.get("center_of_mass", (0, 0, 0))),
                        inertia=tuple(st.facts.get("inertia", (1e-3, 0, 0, 1e-3, 0, 1e-3))),
                        friction=mat.get("friction", 0.6), material_class=st.facts.get("material_class"))
            asset = AssetModel(name=inst.instance_id, links=[link],
                               joints=[Joint.model_validate(j) for j in st.facts.get("joints", [])])
            if "urdf" in formats:
                write_urdf(asset, {"base": "visual.obj"}, {"base": col_files}, sim / "asset.urdf")
                files["urdf"] = "sim/asset.urdf"
            if "mjcf" in formats:
                write_mjcf(asset, {"base": "visual.obj"}, {"base": col_files}, sim / "asset.xml",
                           {"base": texture} if texture else None)
                files["mjcf"] = "sim/asset.xml"
        size = tuple(float(x) for x in (mesh.bounds[1] - mesh.bounds[0]))
        manifest = AssetManifest(
            job_id=job.job_id, scene_id=job.inputs.scene_id, instance_id=inst.instance_id, label=inst.label,
            grade=st.grade or Grade.C, files=files, bbox_size_m=size, generated_ratio=st.generated_ratio,
            articulated=bool(st.facts.get("joints")),
            qa=[r.qa.summary() for r in st.stages.values() if r.qa is not None],
            lineage=Lineage(pipeline_version=__version__, tool_versions=svc.tool_versions(),
                            model_versions={"agent": svc.settings.llm.model, "critic": svc.settings.llm.critic_model},
                            skills_used=sorted({s for r in st.stages.values() for s in r.skills_used}),
                            input_hashes={"segment_gaussians": next(s.gaussians.sha256 for s in job.inputs.segments
                                                                    if s.segment_id == inst.segment_id),
                                          "scene_gaussians": job.inputs.scene_gaussians.sha256},
                            config_hash=svc.settings.fingerprint),
        )
        (out / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def _write_scene_manifest(self, job: JobSpec, states: list[InstanceState], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        scene = {
            "job_id": job.job_id, "scene_id": job.inputs.scene_id, "pipeline_version": __version__,
            "config_hash": self.svc.settings.fingerprint, "created_at": utcnow().isoformat(),
            "instances": [{
                "instance_id": s.instance.instance_id, "segment_id": s.instance.segment_id, "label": s.instance.label,
                "grade": s.grade.value if s.grade else None, "supported_by": s.instance.supported_by,
                "world_from_asset": world_pose(s.facts["transform"]) if "transform" in s.facts else None,
                "stages": {k: v.status.value for k, v in s.stages.items()},
                "cost_usd": round(sum(r.cost_usd for r in s.stages.values()), 4),
            } for s in states],
        }
        (out_dir / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
