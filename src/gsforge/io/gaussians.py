"""3DGS point cloud I/O (the original INRIA ply layout, also produced by gsplat / nerfstudio exports)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from ..core.errors import InputError

SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    means: np.ndarray  # (N, 3) float32
    scales_log: np.ndarray  # (N, 3) log-scale, as stored
    quats_wxyz: np.ndarray  # (N, 4) unnormalized, as stored
    opacity_logit: np.ndarray  # (N,)
    sh_dc: np.ndarray  # (N, 3)
    sh_rest: np.ndarray  # (N, R) raw f_rest_* columns, R may be 0

    def __len__(self) -> int:
        return int(self.means.shape[0])

    # ---- activated views
    @property
    def opacity(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logit))

    @property
    def scales(self) -> np.ndarray:
        return np.exp(self.scales_log)

    @property
    def rgb(self) -> np.ndarray:
        """View-independent base color in [0, 1] (SH band 0 only)."""
        return np.clip(0.5 + SH_C0 * self.sh_dc, 0.0, 1.0)

    @property
    def rotations(self) -> Rotation:
        q = self.quats_wxyz / np.linalg.norm(self.quats_wxyz, axis=1, keepdims=True).clip(1e-12)
        return Rotation.from_quat(q[:, [1, 2, 3, 0]])  # scipy uses xyzw

    def normals(self) -> np.ndarray:
        """Axis of the smallest scale: the surface normal for flat (well-converged) gaussians."""
        mats = self.rotations.as_matrix()  # (N, 3, 3), columns are local axes
        idx = np.argmin(self.scales_log, axis=1)
        return mats[np.arange(len(self)), :, idx]

    def bbox(self, min_opacity: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        pts = self.means[self.opacity >= min_opacity] if min_opacity > 0 else self.means
        if len(pts) == 0:
            raise InputError("no gaussians above the opacity threshold")
        return pts.min(axis=0), pts.max(axis=0)

    # ---- editing
    def subset(self, mask: np.ndarray) -> Gaussians:
        return Gaussians(self.means[mask], self.scales_log[mask], self.quats_wxyz[mask],
                         self.opacity_logit[mask], self.sh_dc[mask], self.sh_rest[mask])

    @staticmethod
    def concat(parts: list[Gaussians]) -> Gaussians:
        return Gaussians(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                           ("means", "scales_log", "quats_wxyz", "opacity_logit", "sh_dc", "sh_rest")))

    def transformed(self, rotation: np.ndarray, translation: np.ndarray, scale: float = 1.0) -> Gaussians:
        """Apply x' = s * R x + t. Higher-order SH are not rotated; they are dropped to avoid silent errors."""
        r = Rotation.from_matrix(rotation)
        means = (scale * (rotation @ self.means.T)).T + translation
        q = (r * self.rotations).as_quat()[:, [3, 0, 1, 2]]
        rest = self.sh_rest if np.allclose(rotation, np.eye(3)) else np.zeros((len(self), 0), np.float32)
        return Gaussians(means.astype(np.float32), (self.scales_log + np.log(scale)).astype(np.float32),
                         q.astype(np.float32), self.opacity_logit, self.sh_dc, rest)


def read_gaussians(path: str | Path) -> Gaussians:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise InputError(f"{path}: no 'vertex' element")
    v = ply["vertex"].data
    names = v.dtype.names
    required = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"scale_{i}" for i in range(3)] + [
        f"rot_{i}" for i in range(4)]
    missing = [n for n in required if n not in names]
    if missing:
        raise InputError(f"{path}: not a 3DGS ply, missing properties {missing}")

    def stack(cols: list[str]) -> np.ndarray:
        return np.stack([np.asarray(v[c], dtype=np.float32) for c in cols], axis=1)

    rest_cols = sorted((n for n in names if n.startswith("f_rest_")), key=lambda s: int(s.rsplit("_", 1)[1]))
    sh_rest = stack(rest_cols) if rest_cols else np.zeros((len(v), 0), np.float32)
    return Gaussians(
        means=stack(["x", "y", "z"]),
        scales_log=stack([f"scale_{i}" for i in range(3)]),
        quats_wxyz=stack([f"rot_{i}" for i in range(4)]),
        opacity_logit=np.asarray(v["opacity"], dtype=np.float32),
        sh_dc=stack(["f_dc_0", "f_dc_1", "f_dc_2"]),
        sh_rest=sh_rest,
    )


def write_gaussians(g: Gaussians, path: str | Path) -> None:
    n = len(g)
    cols: list[tuple[str, np.ndarray]] = [("x", g.means[:, 0]), ("y", g.means[:, 1]), ("z", g.means[:, 2])]
    cols += [("nx", np.zeros(n)), ("ny", np.zeros(n)), ("nz", np.zeros(n))]
    cols += [(f"f_dc_{i}", g.sh_dc[:, i]) for i in range(3)]
    cols += [(f"f_rest_{i}", g.sh_rest[:, i]) for i in range(g.sh_rest.shape[1])]
    cols += [("opacity", g.opacity_logit)]
    cols += [(f"scale_{i}", g.scales_log[:, i]) for i in range(3)]
    cols += [(f"rot_{i}", g.quats_wxyz[:, i]) for i in range(4)]
    arr = np.empty(n, dtype=[(name, "f4") for name, _ in cols])
    for name, values in cols:
        arr[name] = values
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def gaussians_from_points(points: np.ndarray, colors: np.ndarray, normals: np.ndarray | None = None,
                          scale: float = 0.005, opacity: float = 0.95) -> Gaussians:
    """Flat gaussians oriented to the given normals. Used by the synthetic eval generator and tests."""
    n = len(points)
    scales = np.full((n, 3), scale, np.float32)
    quats = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    if normals is not None:
        # rotate local z onto the normal, then flatten local z
        z = np.array([0.0, 0.0, 1.0])
        rots = []
        for nrm in normals:
            nrm = nrm / max(np.linalg.norm(nrm), 1e-12)
            axis = np.cross(z, nrm)
            s = np.linalg.norm(axis)
            angle = np.arctan2(s, float(np.dot(z, nrm)))
            rv = axis / s * angle if s > 1e-9 else (np.array([np.pi, 0, 0]) if nrm[2] < 0 else np.zeros(3))
            rots.append(rv)
        quats = Rotation.from_rotvec(np.asarray(rots)).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
        scales[:, 2] = scale * 0.1
    logit = np.log(opacity / (1 - opacity))
    return Gaussians(
        means=points.astype(np.float32),
        scales_log=np.log(scales).astype(np.float32),
        quats_wxyz=quats,
        opacity_logit=np.full(n, logit, np.float32),
        sh_dc=((np.clip(colors, 0, 1) - 0.5) / SH_C0).astype(np.float32),
        sh_rest=np.zeros((n, 0), np.float32),
    )
