"""Turn a job description with local paths / URIs into a JobSpec of content-addressed artifacts."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml

from ..core.errors import InputError
from ..core.models import ArtifactKind, CaptureInputs, JobSpec, SceneInputs, SegmentInput
from ..storage.artifacts import ArtifactStore


def _put(store: ArtifactStore, path: str | Path, kind: ArtifactKind, base: Path):
    p = Path(path)
    p = p if p.is_absolute() else base / p
    if not p.exists():
        raise InputError(f"input file not found: {p}")
    return store.put_file(p, kind)


def ingest_job(spec: dict[str, Any], store: ArtifactStore, base_dir: str | Path = ".") -> JobSpec:
    base = Path(base_dir)
    try:
        segments = [SegmentInput(segment_id=str(s["id"]), label_hint=s.get("label"),
                                 gaussians=_put(store, s["gaussians"], ArtifactKind.GAUSSIANS, base))
                    for s in spec["segments"]]
        capture = None
        if spec.get("capture"):
            c = spec["capture"]
            images = Path(c["images_dir"])
            capture = CaptureInputs(images_dir=str(images if images.is_absolute() else base / images),
                                    cameras=_put(store, c["cameras"], ArtifactKind.CAMERAS, base),
                                    metric_scale=bool(c.get("metric_scale", True)))
        inputs = SceneInputs(
            scene_id=str(spec["scene_id"]),
            scene_gaussians=_put(store, spec["scene_gaussians"], ArtifactKind.GAUSSIANS, base),
            scene_mesh=_put(store, spec["scene_mesh"], ArtifactKind.MESH, base) if spec.get("scene_mesh") else None,
            segments=segments, capture=capture, up_axis=spec.get("up_axis", "auto"))
    except KeyError as e:
        raise InputError(f"job spec is missing field {e}") from None
    return JobSpec(job_id=str(spec.get("job_id") or f"job-{uuid.uuid4().hex[:10]}"), inputs=inputs,
                   profile=spec.get("profile", "default"), priority=int(spec.get("priority", 0)))


def load_job_file(path: str | Path, store: ArtifactStore) -> JobSpec:
    path = Path(path)
    return ingest_job(yaml.safe_load(path.read_text(encoding="utf-8")), store, path.parent)
