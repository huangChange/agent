"""Working-mesh I/O. Internally meshes are trimesh objects in meters, +Z up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from ..core.errors import InputError


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise InputError(f"{path}: not a non-empty triangle mesh")
    return loaded


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path


def vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """RGBA uint8 (V, 4) if the mesh carries per-vertex colors, else None."""
    if mesh.visual.kind == "vertex":
        return np.asarray(mesh.visual.vertex_colors)
    return None
