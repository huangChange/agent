"""Observation tools: how the agent (and human reviewers) *see* the current state."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ..core.models import ArtifactRef
from ..io.cameras import orbit_cameras
from ..qa.metrics import fidelity_metrics, topology_metrics
from ..render.raster import contact_sheet, render_mesh, render_splats
from .base import ToolContext, ToolResult, tool

PREVIEW_MAX_FACES = 20000


class PreviewParams(BaseModel):
    mesh: ArtifactRef | None = None
    gaussians: ArtifactRef | None = Field(None, description="reference rendering of the gaussians")
    views: int = Field(6, ge=1, le=12)
    size: int = Field(384, ge=128, le=1024)
    elevation_deg: float = 25.0


@tool("render_previews", stages=("*",), version="1.0.0", cacheable=True)
def render_previews(ctx: ToolContext, p: PreviewParams) -> ToolResult:
    """Render orbit views of the mesh and/or the gaussians into contact sheets (same cameras, so the
    two sheets are directly comparable). Use this to LOOK at the result before and after a change."""
    if p.mesh is None and p.gaussians is None:
        return ToolResult(warnings=["nothing to render"])
    mesh = ctx.load_mesh(p.mesh) if p.mesh else None
    g = ctx.load_gaussians(p.gaussians) if p.gaussians else None
    if mesh is not None:
        lo, hi = mesh.bounds
    else:
        lo, hi = g.bbox(min_opacity=0.1)
    center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) * 1.1
    cams = orbit_cameras(center, radius, p.views, elevation_deg=p.elevation_deg, size=p.size)
    previews, outputs = [], {}
    if mesh is not None:
        m = mesh
        if len(m.faces) > PREVIEW_MAX_FACES and m.visual.kind != "vertex":
            m = m.simplify_quadric_decimation(face_count=PREVIEW_MAX_FACES)
        sheet = contact_sheet([render_mesh(m, c).rgb for c in cams])
        ref = ctx.put_image(sheet, "mesh_views.png", view="mesh")
        previews.append(ref)
        outputs["mesh_views"] = ref
    if g is not None:
        sheet = contact_sheet([render_splats(g, c).rgb for c in cams])
        ref = ctx.put_image(sheet, "gaussian_views.png", view="gaussians")
        previews.append(ref)
        outputs["gaussian_views"] = ref
    return ToolResult(outputs=outputs, previews=previews)


class ReportParams(BaseModel):
    mesh: ArtifactRef
    gaussians: ArtifactRef | None = None


@tool("mesh_report", stages=("*",), version="1.0.0")
def mesh_report(ctx: ToolContext, p: ReportParams) -> ToolResult:
    """Topology (watertight, components, non-manifold / boundary edges) and, if gaussians are given,
    two-sided fidelity (recall, precision, F-score, p95 distances as a fraction of bbox diagonal)."""
    mesh = ctx.load_mesh(p.mesh)
    metrics = topology_metrics(mesh)
    if p.gaussians:
        metrics.update(fidelity_metrics(mesh, ctx.load_gaussians(p.gaussians)))
    size = mesh.bounds[1] - mesh.bounds[0]
    metrics.update({"size_x": float(size[0]), "size_y": float(size[1]), "size_z": float(size[2])})
    return ToolResult(metrics=metrics)
