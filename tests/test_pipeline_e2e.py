"""End-to-end: synthetic table scene -> split -> geometry -> ... -> delivery files, no LLM."""

import json
from pathlib import Path

import trimesh

from gsforge.evals.benchmark import mesh_fscore
from gsforge.pipeline.factory import build_services
from gsforge.pipeline.ingest import ingest_job
from gsforge.pipeline.orchestrator import Pipeline


async def test_table_scene_end_to_end(settings, table_scene):
    svc = build_services(settings, inline=True, use_llm=False)
    job = ingest_job({"job_id": "e2e", "scene_id": "synthetic", "scene_gaussians": str(table_scene.scene_gaussians),
                      "scene_mesh": str(table_scene.scene_mesh),
                      "segments": [{"id": "table", "gaussians": str(table_scene.segment_gaussians), "label": "table"}]},
                     svc.store)
    result = await Pipeline(svc).run(job)
    assert len(result.instances) == 3, result.warnings

    out = Path(settings.export.out_dir) / "synthetic"
    scene = json.loads((out / "scene.json").read_text())
    assert len(scene["instances"]) == 3
    for st in result.instances:
        d = out / st.instance.instance_id
        manifest = json.loads((d / "manifest.json").read_text())
        assert {"glb", "ply", "gaussians_ply"} <= set(manifest["files"])
        assert manifest["lineage"]["tool_versions"]["split_instances"]
        glb = trimesh.load(d / "asset.glb", force="mesh")
        assert len(glb.faces) > 100
        assert st.stages["geometry"].qa is not None
        assert st.facts["mass"] > 0

    # the table is delivered with the area under the remote and mug filled
    table = next(s for s in result.instances if s.facts["role"] == "support")
    assert table.instance.label == "table"
    assert table.generated_ratio > 0

    # resume: a second run loads checkpoints and recomputes nothing
    again = await Pipeline(svc).run(job)
    assert again.grades == result.grades
    svc.runner.backend.shutdown()


def test_mesh_fscore_identity():
    m = trimesh.creation.box(extents=(1, 1, 1))
    assert mesh_fscore(m, m, tau=0.01)["fscore"] > 0.99
