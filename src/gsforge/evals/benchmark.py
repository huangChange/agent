"""Benchmark: run the pipeline on a synthetic scene and score deliveries against ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..core.config import Settings
from ..pipeline.factory import build_services
from ..pipeline.ingest import ingest_job
from ..pipeline.orchestrator import Pipeline
from .synthetic import make_table_scene


def mesh_fscore(pred: trimesh.Trimesh, gt: trimesh.Trimesh, tau: float, n: int = 100000) -> dict[str, float]:
    a, _ = trimesh.sample.sample_surface(pred, n, seed=0)
    b, _ = trimesh.sample.sample_surface(gt, n, seed=1)
    d_ab, _ = cKDTree(b).query(a)
    d_ba, _ = cKDTree(a).query(b)
    p, r = float((d_ab <= tau).mean()), float((d_ba <= tau).mean())
    return {"precision": p, "recall": r, "fscore": 2 * p * r / (p + r) if p + r else 0.0,
            "chamfer_m": float((d_ab.mean() + d_ba.mean()) / 2)}


async def run_synthetic(workdir: str | Path, settings: Settings, use_llm: bool = False,
                        inline: bool = False) -> dict[str, Any]:
    workdir = Path(workdir)
    scene = make_table_scene(workdir / "scene")
    svc = build_services(settings, inline=inline, use_llm=use_llm)
    job = ingest_job({"job_id": "synthetic-table", "scene_id": "synthetic_table",
                      "scene_gaussians": str(scene.scene_gaussians), "scene_mesh": str(scene.scene_mesh),
                      "up_axis": "auto", "segments": [{"id": "table", "gaussians": str(scene.segment_gaussians),
                                                       "label": "table"}]}, svc.store)
    try:
        result = await Pipeline(svc).run(job)
    finally:
        svc.runner.backend.shutdown()
    gts = {k: trimesh.load(str(p), force="mesh") for k, p in scene.gt_meshes.items()}
    scores: dict[str, Any] = {}
    for st in result.instances:
        if "transform" not in st.facts or "mesh" not in st.current:
            continue
        mesh = trimesh.load(str(svc.store.local_path(st.current["mesh"])), force="mesh")
        mesh.apply_transform(np.linalg.inv(np.asarray(st.facts["transform"])))  # back to the capture frame
        c = mesh.bounds.mean(axis=0)
        name = min(gts, key=lambda k: np.linalg.norm(gts[k].bounds.mean(axis=0) - c))
        tau = 0.01 * float(np.linalg.norm(gts[name].extents))
        scores[st.instance.instance_id] = {"matched_gt": name, "grade": st.grade.value if st.grade else None,
                                           **mesh_fscore(mesh, gts[name], tau)}
    summary = {"instances_found": len(result.instances), "instances_expected": scene.expected_instances,
               "grades": result.grades, "per_instance": scores, "cost_usd": result.cost_usd,
               "warnings": result.warnings}
    (workdir / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
