"""Appearance transfer from gaussians to mesh vertices (CPU baseline; always available)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import trimesh
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

from ...core.models import ArtifactRef
from ..base import ToolContext, ToolResult, tool


def diffuse_fill(mesh: trimesh.Trimesh, values: np.ndarray, known: np.ndarray, max_iters: int = 1000) -> np.ndarray:
    """Fill unknown vertex values by repeated neighbour averaging over the mesh graph (harmonic-ish)."""
    edges = mesh.edges_unique
    n = len(mesh.vertices)
    A = sp.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n)).tocsr()
    vals = values.copy()
    known = known.copy()
    for _ in range(max_iters):
        if known.all():
            break
        k = known.astype(np.float64)
        s = A @ (vals * k[:, None])
        c = A @ k
        frontier = (~known) & (c > 0)
        if not frontier.any():
            break
        vals[frontier] = s[frontier] / c[frontier, None]
        known = known | frontier
    if not known.all():  # disconnected islands without any observation
        vals[~known] = vals[known].mean(axis=0) if known.any() else 0.5
    return vals


def unobserved_area_ratio(mesh: trimesh.Trimesh, observed_vertex: np.ndarray) -> float:
    bad_face = ~observed_vertex[mesh.faces].all(axis=1)
    return float(mesh.area_faces[bad_face].sum() / max(mesh.area, 1e-12))


class VertexColorParams(BaseModel):
    mesh: ArtifactRef
    gaussians: ArtifactRef
    k: int = Field(8, ge=1, le=64)
    min_opacity: float = 0.2
    max_distance_ratio: float = Field(0.02, description="vertices farther than this x bbox diagonal from "
                                                        "any gaussian count as unobserved")


@tool("transfer_vertex_colors", stages=("texture",), version="1.0.0")
def transfer_vertex_colors(ctx: ToolContext, p: VertexColorParams) -> ToolResult:
    """Color each mesh vertex from nearby gaussians (opacity- and distance-weighted base color).
    Vertices with no gaussian nearby (occluded bottoms, contact areas) are filled by diffusion over
    the mesh and reported in metrics['generated_ratio'] (area fraction)."""
    mesh = ctx.load_mesh(p.mesh)
    g = ctx.load_gaussians(p.gaussians)
    keep = g.opacity >= p.min_opacity
    pts, rgb, op = g.means[keep].astype(np.float64), g.rgb[keep], g.opacity[keep]
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
    max_d = p.max_distance_ratio * diag
    d, nb = cKDTree(pts).query(mesh.vertices, k=p.k)
    d, nb = d.reshape(len(mesh.vertices), -1), nb.reshape(len(mesh.vertices), -1)
    sigma = max(float(np.median(d[:, 0])), 1e-6) * 2
    w = op[nb] * np.exp(-(d ** 2) / (2 * sigma ** 2)) * (d <= max_d)
    wsum = w.sum(axis=1)
    observed = wsum > 1e-8
    colors = np.zeros((len(mesh.vertices), 3))
    colors[observed] = (rgb[nb[observed]] * w[observed, :, None]).sum(axis=1) / wsum[observed, None]
    colors = diffuse_fill(mesh, colors, observed)
    rgba = np.concatenate([np.clip(colors * 255, 0, 255), np.full((len(colors), 1), 255)], axis=1).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)
    ratio = unobserved_area_ratio(mesh, observed)
    return ToolResult(outputs={"mesh": ctx.put_mesh(mesh, "colored.ply")},
                      metrics={"generated_ratio": ratio, "observed_vertex_ratio": float(observed.mean())})
