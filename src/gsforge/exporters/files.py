"""Primary deliverables: GLB (glTF 2.0, +Y up, PBR) and PLY (mesh with vertex colors)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# canonical +Z up  ->  glTF +Y up:  (x, y, z) -> (x, z, -y)
Z_UP_TO_Y_UP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def _apply_pbr(mesh: trimesh.Trimesh, roughness: float, metallic: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    if mesh.visual.kind == "texture" and isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
        mesh.visual.material.roughnessFactor = roughness
        mesh.visual.material.metallicFactor = metallic
    return mesh


def export_glb(parts: dict[str, trimesh.Trimesh], path: str | Path, roughness: float = 0.8,
               metallic: float = 0.0, transforms: dict[str, np.ndarray] | None = None) -> Path:
    """One node per part (a rigid asset has one part; articulated assets one per link), so the
    hierarchy survives into DCC tools and USD conversion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for name, mesh in parts.items():
        local = (transforms or {}).get(name, np.eye(4))
        scene.add_geometry(_apply_pbr(mesh, roughness, metallic), node_name=name, geom_name=name,
                           transform=Z_UP_TO_Y_UP @ local)
    path.write_bytes(scene.export(file_type="glb"))
    return path


def export_ply(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Binary PLY with per-vertex RGBA (textures are sampled down to vertex colors). Stays +Z up."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mesh.copy()
    if m.visual.kind == "texture":
        m.visual = m.visual.to_color()
    path.write_bytes(trimesh.exchange.ply.export_ply(m, encoding="binary"))
    return path


def export_obj_with_texture(mesh: trimesh.Trimesh, path: str | Path) -> list[Path]:
    """OBJ + MTL + PNG for simulators that don't read glTF (MuJoCo, URDF loaders)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text, files = trimesh.exchange.obj.export_obj(mesh, include_texture=True, return_texture=True,
                                                  mtl_name=f"{path.stem}.mtl")
    path.write_text(text, encoding="utf-8")
    written = [path]
    for name, data in files.items():
        p = path.parent / name
        p.write_bytes(data)
        written.append(p)
    return written
