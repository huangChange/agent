"""Deterministic stage implementations. Each takes the instance state, calls tools, and returns the
updates; QA gating, agent escalation and checkpointing are handled by the orchestrator."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import DependencyMissingError, ToolError
from ..core.models import ArtifactRef, InstanceState, JobSpec, QAReport
from ..qa.evaluators import evaluate_stage
from .services import Services

log = logging.getLogger("gsforge.stages")


@dataclass
class StageOutput:
    current: dict[str, ArtifactRef] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    generated_ratio: float | None = None
    warnings: list[str] = field(default_factory=list)


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# ---------------------------------------------------------------- geometry


async def geometry(svc: Services, job: JobSpec, st: InstanceState) -> StageOutput:
    """Run every configured extraction candidate, repair each, keep the best by QA score."""
    jid, iid = job.job_id, st.instance.instance_id
    gcfg = svc.settings.geometry
    gaussians = st.current["gaussians"]
    candidates: list[tuple[str, dict[str, Any]]] = []
    for name in gcfg.candidates:
        if name == "density_mc":
            candidates.append(("extract_density_mc", {"gaussians": gaussians, "resolution": gcfg.voxel_resolution,
                                                      "min_opacity": gcfg.min_opacity}))
        elif name == "scene_mesh_crop" and job.inputs.scene_mesh is not None:
            candidates.append(("extract_scene_mesh_crop", {"scene_mesh": job.inputs.scene_mesh,
                                                           "gaussians": gaussians}))
        elif name == "tsdf_render" and svc.gpu_available:
            candidates.append(("extract_tsdf_render", {"gaussians": gaussians}))

    async def step(tool: str, mesh: ArtifactRef, **kw: Any) -> ArtifactRef:
        return (await svc.call(tool, jid, iid, mesh=mesh, **kw)).outputs["mesh"]

    best: tuple[tuple[bool, float, float], ArtifactRef, str] | None = None
    warnings: list[str] = []
    for tool, params in candidates:
        try:
            mesh = (await svc.call(tool, jid, iid, **params)).outputs["mesh"]
            mesh = await step("repair_mesh", mesh, max_hole_edges=gcfg.hole_fill_max_edges)
            if tool == "extract_density_mc":
                mesh = await step("smooth_mesh", mesh, iterations=8)
            mesh = await step("simplify_mesh", mesh, target_faces=gcfg.target_faces)
            # decimation can create a few non-manifold edges: a final cleanup pass
            mesh = await step("repair_mesh", mesh, max_hole_edges=gcfg.hole_fill_max_edges)
        except (ToolError, DependencyMissingError) as e:
            warnings.append(f"candidate {tool} failed: {e}")
            continue
        trial = st.model_copy(deep=True)
        trial.current["mesh"] = mesh
        report: QAReport = await _evaluate(svc, "geometry", job, trial)
        fscore = next((c.value for c in report.checks if c.name == "fscore" and c.value is not None), 0.0)
        key = (report.passed, report.score, fscore)
        log.info("geometry candidate %s for %s: passed=%s score=%.3f", tool, iid, report.passed, report.score)
        if best is None or key > best[0]:
            best = (key, mesh, tool)
    if best is None:
        raise ToolError("no geometry candidate succeeded: " + "; ".join(warnings))
    return StageOutput(current={"mesh": best[1]}, facts={"geometry_method": best[2]}, warnings=warnings)


async def align(svc: Services, job: JobSpec, st: InstanceState) -> StageOutput:
    scale = 1.0 if job.inputs.capture is None or job.inputs.capture.metric_scale else float(st.facts.get("scale", 1.0))
    r = await svc.call("canonicalize_pose", job.job_id, st.instance.instance_id, mesh=st.current["mesh"],
                       gaussians=st.current["gaussians"], up=tuple(st.facts["up"]), scale=scale)
    # "geometry" stays the closed, unsplit mesh for physics; texturing later replaces "mesh" with a
    # UV-unwrapped copy whose seam vertices are duplicated (no longer a closed volume)
    return StageOutput(current={"mesh": r.outputs["mesh"], "geometry": r.outputs["mesh"],
                                "gaussians": r.outputs["gaussians"]},
                       facts={"transform": r.facts["transform"], "bbox_size_m": r.facts["bbox_size_m"]})


async def texture(svc: Services, job: JobSpec, st: InstanceState) -> StageOutput:
    jid, iid = job.job_id, st.instance.instance_id
    r = await svc.call("transfer_vertex_colors", jid, iid, mesh=st.current["mesh"], gaussians=st.current["gaussians"])
    colored, generated = r.outputs["mesh"], r.metrics["generated_ratio"]
    out = StageOutput(current={"mesh": colored}, facts={"vertex_colors": True}, generated_ratio=generated)
    cap = job.inputs.capture
    size = svc.settings.texture.size
    try:
        if cap is not None and svc.settings.texture.prefer_capture_images and svc.gpu_available:
            b = await svc.call("bake_from_capture", jid, iid, mesh=colored, cameras=cap.cameras,
                               images_dir=cap.images_dir, canonical_from_world=st.facts["transform"],
                               texture_size=size)
            out.generated_ratio = b.metrics["generated_ratio"]
        elif _available("xatlas"):
            b = await svc.call("bake_vertex_colors", jid, iid, mesh=colored, texture_size=size)
        else:
            b = None
            out.warnings.append("xatlas missing: delivering vertex colors without a texture map")
        if b is not None:
            out.current.update(mesh=b.outputs["mesh"], texture=b.outputs["texture"])
    except (ToolError, DependencyMissingError) as e:
        out.warnings.append(f"texture bake failed, keeping vertex colors: {e}")
    return out


async def material(svc: Services, job: JobSpec, st: InstanceState) -> StageOutput:
    r = await svc.call("assign_material", job.job_id, st.instance.instance_id,
                       material_class=st.facts.get("material_class"), label=st.instance.label)
    return StageOutput(facts={"material": r.facts, "material_class": r.facts["material_class"]})


async def physics(svc: Services, job: JobSpec, st: InstanceState) -> StageOutput:
    mat = st.facts.get("material", {})
    method = svc.settings.export.collision
    if method == "coacd" and not _available("coacd"):
        method = "convex_hull"
    if method == "none":
        return StageOutput()
    r = await svc.call("build_collision", job.job_id, st.instance.instance_id,
                       mesh=st.current.get("geometry", st.current["mesh"]),
                       method=method, density=float(mat.get("density", 1000.0)))
    current = {k: v for k, v in r.outputs.items() if k.startswith("collision_")}
    facts = {k: r.facts[k] for k in ("mass", "center_of_mass", "inertia")}
    warnings = list(r.warnings)
    if _available("mujoco"):
        sim = await svc.call("sim_drop_test", job.job_id, st.instance.instance_id,
                             collision=[current[k] for k in sorted(current)], mass=facts["mass"],
                             center_of_mass=tuple(facts["center_of_mass"]), inertia=tuple(facts["inertia"]),
                             friction=float(mat.get("friction", 0.6)))
        facts["sim"] = sim.metrics
        warnings += sim.warnings
    return StageOutput(current=current, facts=facts, warnings=warnings)


STAGES = {"geometry": geometry, "align": align, "texture": texture, "material": material, "physics": physics}
STAGE_ORDER = ("geometry", "align", "texture", "material", "physics")


async def _evaluate(svc: Services, stage: str, job: JobSpec, st: InstanceState) -> QAReport:
    import asyncio

    ctx = svc.local_ctx(job.job_id, st.instance.instance_id)
    report = await asyncio.to_thread(evaluate_stage, ctx, stage, st)
    return report if report is not None else QAReport(stage=stage)


def world_pose(transform: list[list[float]]) -> list[list[float]]:
    """Object pose in the capture's world frame = inverse of canonical_from_world."""
    return np.linalg.inv(np.asarray(transform)).tolist()
