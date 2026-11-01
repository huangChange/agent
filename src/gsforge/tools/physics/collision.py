"""Collision geometry and mass properties for simulation."""

from __future__ import annotations

from typing import Literal

import trimesh
from pydantic import BaseModel, Field

from ...core.errors import DependencyMissingError
from ...core.models import ArtifactRef
from ..base import ToolContext, ToolResult, tool


def convex_decompose(mesh: trimesh.Trimesh, threshold: float, max_parts: int) -> list[trimesh.Trimesh]:
    try:
        import coacd
    except ImportError as e:
        raise DependencyMissingError("coacd", "geo") from e
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold,
                            max_convex_hull=max_parts)
    return [trimesh.Trimesh(v, f, process=True) for v, f in parts]


class CollisionParams(BaseModel):
    mesh: ArtifactRef
    method: Literal["convex_hull", "coacd"] = "convex_hull"
    density: float = Field(1000.0, gt=0, description="kg/m^3")
    coacd_threshold: float = Field(0.05, description="concavity threshold; lower = more, tighter parts")
    max_parts: int = 32
    proxy_faces: int = Field(8000, description="decompose a decimated proxy: collision needs no visual "
                                               "detail, and decomposition time grows fast with face count")


@tool("build_collision", stages=("physics",), version="1.1.0")
def build_collision(ctx: ToolContext, p: CollisionParams) -> ToolResult:
    """Build simulation collision geometry (single convex hull, or CoACD approximate convex
    decomposition for concave objects like drawers, cups, chairs) and compute mass, center of mass
    and inertia tensor from the visual mesh volume and the material density."""
    mesh = ctx.load_mesh(p.mesh)
    if p.method == "convex_hull":
        parts = [mesh.convex_hull]
    else:
        proxy = mesh.simplify_quadric_decimation(face_count=p.proxy_faces) if len(mesh.faces) > p.proxy_faces else mesh
        parts = convex_decompose(proxy, p.coacd_threshold, p.max_parts)
    solid = mesh if mesh.is_volume else mesh.convex_hull
    solid.density = p.density
    mass = float(solid.mass)
    com = solid.center_mass
    inertia = solid.moment_inertia  # about the center of mass
    outputs = {f"collision_{i}": ctx.put_mesh(part, f"collision_{i}.ply") for i, part in enumerate(parts)}
    warnings = [] if mesh.is_volume else ["visual mesh is not a closed volume; mass from its convex hull"]
    hull_vol = sum(part.volume for part in parts)
    return ToolResult(
        outputs=outputs,
        facts={"mass": mass, "center_of_mass": com.tolist(),
               "inertia": [float(inertia[i, j]) for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]},
        metrics={"mass_kg": mass, "collision_parts": float(len(parts)),
                 "collision_volume_ratio": float(hull_vol / max(solid.volume, 1e-12))},
        warnings=warnings,
    )
