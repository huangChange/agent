"""Synthetic scenes with ground truth: known meshes -> surface-sampled gaussians -> pipeline -> compare.

Real captures have no ground truth, so regressions in the pipeline (or in the agent's prompts) are
measured here first. Occlusion is simulated the way it happens in real captures: no samples on
object bottoms or on support surfaces under resting objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from ..io.gaussians import Gaussians, gaussians_from_points, write_gaussians


@dataclass
class SyntheticScene:
    root: Path
    scene_gaussians: Path
    segment_gaussians: Path
    scene_mesh: Path
    gt_meshes: dict[str, Path]
    expected_instances: int


def _box(extents, center) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    return m


def _sample(mesh: trimesh.Trimesh, density: float, rng: np.random.Generator,
            drop=None) -> tuple[np.ndarray, np.ndarray]:
    n = max(int(mesh.area * density), 200)
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
    normals = mesh.face_normals[fid]
    if drop is not None:
        keep = ~drop(pts, normals)
        pts, normals = pts[keep], normals[keep]
    return pts, normals


def make_table_scene(root: str | Path, density: float = 60000.0, seed: int = 0,
                     small_object_density: float | None = None) -> SyntheticScene:
    """Floor + table segment containing a TV remote and a mug resting on the table top.

    Small objects get a higher gaussian density, as in real captures: they are filmed at close range
    and 3DGS densifies where there is fine detail (a 20 cm remote typically gets 10k+ gaussians)."""
    small_density = small_object_density or 8 * density
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    top_h, top_t = 0.74, 0.04
    table = trimesh.util.concatenate(
        [_box((1.2, 0.7, top_t), (0, 0, top_h - top_t / 2))]
        + [_box((0.05, 0.05, top_h - top_t), (sx * 0.55, sy * 0.3, (top_h - top_t) / 2))
           for sx in (-1, 1) for sy in (-1, 1)])
    remote = _box((0.19, 0.05, 0.022), (0.2, 0.05, top_h + 0.011))
    mug = trimesh.creation.cylinder(radius=0.042, height=0.1, sections=48)
    mug.apply_translation((-0.25, -0.1, top_h + 0.05))
    floor = _box((4.0, 4.0, 0.02), (0, 0, -0.01))

    def under(objs, z, tol=0.004):
        def drop(p, n):
            hit = np.zeros(len(p), bool)
            for o in objs:
                lo, hi = o.bounds
                hit |= (np.abs(p[:, 2] - z) < tol) & (p[:, 0] >= lo[0]) & (p[:, 0] <= hi[0]) & \
                       (p[:, 1] >= lo[1]) & (p[:, 1] <= hi[1])
            return hit
        return drop

    def bottom(p, n):  # downward-facing faces of resting objects are never observed
        return n[:, 2] < -0.9

    parts: list[Gaussians] = []

    def add(mesh, color_fn, drop=None, d=density):
        pts, nrm = _sample(mesh, d, rng, drop)
        # splat size follows sampling density, as 3DGS densification does
        parts.append(gaussians_from_points(pts, color_fn(pts), nrm, scale=0.7 / np.sqrt(d)))
        return parts[-1]

    wood = lambda p: np.tile([0.55, 0.38, 0.22], (len(p), 1)) + 0.05 * np.sin(p[:, :1] * 60)  # noqa: E731
    def table_hidden(p, n):  # under the remote and mug, and the leg soles on the floor
        return under([remote, mug], top_h)(p, n) | ((n[:, 2] < -0.9) & (p[:, 2] < 0.01))

    g_table = add(table, wood, drop=table_hidden)
    def remote_color(p):  # dark body with a lighter button grid
        c = np.tile([0.08, 0.08, 0.09], (len(p), 1))
        buttons = (np.sin(p[:, 0] * 160) > 0.6) & (np.sin(p[:, 1] * 160) > 0.6) & (p[:, 2] > top_h + 0.02)
        c[buttons] = [0.7, 0.7, 0.72]
        return c
    g_remote = add(remote, remote_color, drop=bottom, d=small_density)
    g_mug = add(mug, lambda p: np.tile([0.85, 0.85, 0.8], (len(p), 1)), drop=bottom, d=small_density)
    add(floor, lambda p: np.tile([0.5, 0.5, 0.5], (len(p), 1)),
                  drop=lambda p, n: (n[:, 2] < 0.9) | ((np.abs(p[:, 0]) < 0.6) & (np.abs(p[:, 1]) < 0.35)),
                  d=density / 20)

    scene = Gaussians.concat(parts)
    segment = Gaussians.concat([g_table, g_remote, g_mug])
    paths = SyntheticScene(root=root, scene_gaussians=root / "scene.ply", segment_gaussians=root / "segment_table.ply",
                           scene_mesh=root / "scene_mesh.ply",
                           gt_meshes={"table": root / "gt_table.ply", "remote": root / "gt_remote.ply",
                                      "mug": root / "gt_mug.ply"}, expected_instances=3)
    write_gaussians(scene, paths.scene_gaussians)
    write_gaussians(segment, paths.segment_gaussians)
    noisy = trimesh.util.concatenate([table, remote, mug, floor])
    noisy.vertices += rng.normal(0, 0.001, noisy.vertices.shape)
    noisy.export(paths.scene_mesh)
    for name, m in (("table", table), ("remote", remote), ("mug", mug)):
        m.export(paths.gt_meshes[name])
    (root / "meta.json").write_text(json.dumps({"expected_instances": 3, "seed": seed}), encoding="utf-8")
    return paths
