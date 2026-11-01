"""Gravity alignment, metric scale and canonical pose.

Canonical frame for every delivered asset: meters, +Z up, origin at the bottom-center of the bbox.
(GLB export converts to glTF's +Y up; URDF / MJCF keep +Z.)
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

from ...core.errors import ToolError
from ...core.models import ArtifactRef
from ..base import ToolContext, ToolResult, tool

AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1 + 1e-9:  # opposite: rotate pi around any orthogonal axis
        axis = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def ransac_plane(pts: np.ndarray, threshold: float, iters: int = 500,
                 seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Returns (unit normal, offset d with n.x = d, inlier mask) of the plane with most inliers."""
    rng = np.random.default_rng(seed)
    best = (np.array([0, 0, 1.0]), 0.0, np.zeros(len(pts), bool))
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        d = float(n @ s[0])
        inl = np.abs(pts @ n - d) <= threshold
        if inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    # least-squares refit on a shrinking inlier band: objects resting on the plane fall inside a wide
    # band and would tilt a single refit (a remote on a table tilts it by degrees)
    for band in (threshold, threshold / 3, threshold / 9):
        inl = np.abs(pts @ n - d) <= band
        if inl.sum() < 3:
            break
        q = pts[inl]
        c = q.mean(axis=0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        n = vt[-1] * (1.0 if vt[-1] @ n >= 0 else -1.0)
        d = float(n @ c)
    return n, d, inl


def orient_axis(pts: np.ndarray, n: np.ndarray, slab: float = 0.03) -> np.ndarray:
    """Pick the sign of a vertical axis: the floor is the extreme of the height distribution that
    carries a dense sheet (the opposite extreme is object tops, or a sparsely captured ceiling)."""
    h = pts @ n
    lo, hi = np.percentile(h, [0.5, 99.5])
    span = max(hi - lo, 1e-9)
    low_mass = np.mean(h <= lo + slab * span)
    high_mass = np.mean(h >= hi - slab * span)
    return n if low_mass >= high_mass else -n


class UpParams(BaseModel):
    scene_gaussians: ArtifactRef
    hint: str = Field("auto", description="'auto' or one of +x -x +y -y +z -z (capture convention)")
    cameras: ArtifactRef | None = Field(None, description="capture transforms.json; upright handheld "
                                                          "cameras give the most reliable up direction")
    min_opacity: float = 0.3
    max_points: int = 200000


@tool("estimate_up", stages=("prepare",), version="1.2.0")
def estimate_up(ctx: ToolContext, p: UpParams) -> ToolResult:
    """Estimate the world up vector. The axis is the dominant plane normal (floor / table tops);
    its sign comes from the capture cameras' mean up direction when available, otherwise from which
    end of the height distribution carries the floor sheet."""
    if p.hint != "auto":
        return ToolResult(facts={"up": list(map(float, AXES[p.hint])), "source": "hint"})
    g = ctx.load_gaussians(p.scene_gaussians)
    pts = g.means[g.opacity >= p.min_opacity].astype(np.float64)
    if len(pts) < 100:
        raise ToolError("scene too sparse to estimate the floor")
    if len(pts) > p.max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), p.max_points, replace=False)]
    extent = float(np.linalg.norm(np.percentile(pts, 98, axis=0) - np.percentile(pts, 2, axis=0)))
    # inlier band from the sampling density, not the scene size: a scene-relative band is wider than
    # a table top is thick, and a plane wedged between its two faces then wins
    spacing = float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))
    n, _, inl = ransac_plane(pts, threshold=max(3 * spacing, 1e-4 * extent))
    source = "ransac+height_extremes"
    up = orient_axis(pts, n)
    warnings: list[str] = []
    if p.cameras is not None:
        from ...io.cameras import load_transforms_json

        cams = load_transforms_json(ctx.store.local_path(p.cameras))
        cam_up = -np.mean([c.c2w[:3, 1] for c in cams], axis=0)  # OpenCV: +y is image-down
        cam_up /= max(np.linalg.norm(cam_up), 1e-12)
        if abs(float(cam_up @ n)) >= np.cos(np.radians(20)):
            up, source = (n if cam_up @ n > 0 else -n), "ransac+cameras"
        else:  # dominant plane is a wall or the capture is unusual: trust the cameras
            up, source = cam_up, "cameras"
            warnings.append("dominant plane is not horizontal w.r.t. the cameras; using camera up")
    return ToolResult(facts={"up": up.tolist(), "source": source},
                      metrics={"plane_inlier_ratio": float(inl.mean())}, warnings=warnings)


class CanonParams(BaseModel):
    mesh: ArtifactRef
    gaussians: ArtifactRef
    up: tuple[float, float, float]
    scale: float = Field(1.0, gt=0, description="multiply into meters (1.0 when capture is metric)")


@tool("canonicalize_pose", stages=("align",), version="1.0.0")
def canonicalize_pose(ctx: ToolContext, p: CanonParams) -> ToolResult:
    """Rotate so `up` maps to +Z, scale to meters, and move the origin to the bottom-center of the
    bbox. Applies the identical transform to mesh and gaussians and records it in facts['transform']."""
    mesh = ctx.load_mesh(p.mesh)
    g = ctx.load_gaussians(p.gaussians)
    R = rotation_between(np.asarray(p.up, float), np.array([0.0, 0.0, 1.0]))
    v = p.scale * (R @ mesh.vertices.T).T
    lo, hi = v.min(axis=0), v.max(axis=0)
    t = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    mesh.vertices = v + t
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = p.scale * R, t
    g2 = g.transformed(R, t, p.scale)
    size = hi - lo
    return ToolResult(
        outputs={"mesh": ctx.put_mesh(mesh, "canonical.ply"), "gaussians": ctx.put_gaussians(g2, "canonical.ply")},
        facts={"transform": T.tolist(), "bbox_size_m": size.tolist()},
        metrics={"size_x": float(size[0]), "size_y": float(size[1]), "size_z": float(size[2])},
    )
