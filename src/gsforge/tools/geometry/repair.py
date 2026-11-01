"""Deterministic mesh repair, smoothing and decimation."""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel, Field

from ...core.models import ArtifactRef
from ...qa.metrics import topology_metrics
from ..base import ToolContext, ToolResult, tool


def _try_pymeshlab_close_holes(mesh: trimesh.Trimesh, max_hole_edges: int) -> tuple[trimesh.Trimesh, bool]:
    try:
        import pymeshlab
    except ImportError:
        return mesh, False
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_close_holes(maxholesize=max_hole_edges)
    m = ms.current_mesh()
    return trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=True), True


def remove_nonmanifold_faces(mesh: trimesh.Trimesh) -> int:
    """Drop every face touching an edge shared by more than two faces; the resulting small holes are
    closed by hole filling. Returns the number of faces removed."""
    edges = mesh.edges_sorted
    if not len(edges):
        return 0
    uniq, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    bad_edge = counts[inverse.ravel()] > 2
    if not bad_edge.any():
        return 0
    bad_faces = np.unique(mesh.edges_face[bad_edge])
    keep = np.ones(len(mesh.faces), bool)
    keep[bad_faces] = False
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return int(len(bad_faces))


class RepairParams(BaseModel):
    mesh: ArtifactRef
    min_component_ratio: float = Field(0.02, description="drop components with less than this fraction "
                                                         "of the total surface area")
    keep_largest_only: bool = Field(False, description="keep only the largest connected component")
    fill_holes: bool = True
    max_hole_edges: int = Field(200, description="largest hole (boundary edge count) to close")


@tool("repair_mesh", stages=("geometry",), version="1.1.0")
def repair_mesh(ctx: ToolContext, p: RepairParams) -> ToolResult:
    """Standard cleanup: merge duplicate vertices, drop degenerate/duplicate faces, remove small
    floating components, make winding consistent and outward, close holes (pymeshlab if available,
    trimesh fan fill otherwise). Reports topology before and after."""
    mesh = ctx.load_mesh(p.mesh)
    before = topology_metrics(mesh)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    removed_nm = remove_nonmanifold_faces(mesh)

    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        areas = np.array([m.area for m in parts])
        if p.keep_largest_only:
            parts = [parts[int(np.argmax(areas))]]
        else:
            parts = [m for m, a in zip(parts, areas) if a >= p.min_component_ratio * areas.sum()]
        mesh = trimesh.util.concatenate(parts)

    used_pymeshlab = False
    if p.fill_holes and not mesh.is_watertight:
        mesh, used_pymeshlab = _try_pymeshlab_close_holes(mesh, p.max_hole_edges)
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh, multibody=True)
    after = topology_metrics(mesh)
    warnings = [] if used_pymeshlab else ["pymeshlab not installed: large holes may remain open"]
    return ToolResult(outputs={"mesh": ctx.put_mesh(mesh, "repaired.ply")},
                      metrics={**{f"before_{k}": v for k, v in before.items()},
                               **{f"after_{k}": v for k, v in after.items()},
                               "nonmanifold_faces_removed": float(removed_nm)},
                      warnings=warnings if not after["watertight"] else [])


class SmoothParams(BaseModel):
    mesh: ArtifactRef
    iterations: int = Field(10, ge=1, le=100)
    lamb: float = 0.5
    nu: float = 0.53


@tool("smooth_mesh", stages=("geometry",), version="1.0.0")
def smooth_mesh(ctx: ToolContext, p: SmoothParams) -> ToolResult:
    """Taubin smoothing: removes marching-cubes staircase artifacts without shrinking the mesh."""
    mesh = ctx.load_mesh(p.mesh)
    trimesh.smoothing.filter_taubin(mesh, lamb=p.lamb, nu=p.nu, iterations=p.iterations)
    return ToolResult(outputs={"mesh": ctx.put_mesh(mesh, "smoothed.ply")})


class SimplifyParams(BaseModel):
    mesh: ArtifactRef
    target_faces: int = Field(60000, ge=500)


@tool("simplify_mesh", stages=("geometry",), version="1.0.0")
def simplify_mesh(ctx: ToolContext, p: SimplifyParams) -> ToolResult:
    """Quadric edge-collapse decimation to at most `target_faces` faces (no-op if already below)."""
    mesh = ctx.load_mesh(p.mesh)
    n0 = len(mesh.faces)
    if n0 > p.target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=p.target_faces)
    return ToolResult(outputs={"mesh": ctx.put_mesh(mesh, "simplified.ply")},
                      metrics={"faces_before": n0, "faces_after": len(mesh.faces)})
