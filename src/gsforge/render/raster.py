"""Small CPU renderers for QA and agent previews.

They are deliberately simple (z-buffered triangle raster, sorted alpha splats) so they run on any
worker without OpenGL/CUDA. Fidelity is enough for silhouettes and for a VLM to judge shape and
appearance; GPU workers use render/gs_render.py for the gaussian side when available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ..io.cameras import Camera
from ..io.gaussians import Gaussians


@dataclass
class RenderOutput:
    rgb: np.ndarray  # (H, W, 3) uint8
    alpha: np.ndarray  # (H, W) float in [0, 1]
    depth: np.ndarray  # (H, W) z-depth, 0 where empty


def render_mesh(mesh: trimesh.Trimesh, cam: Camera, background: int = 255,
                light_dir: np.ndarray | None = None) -> RenderOutput:
    H, W = cam.height, cam.width
    zbuf = np.full((H, W), np.inf)
    fid = np.full((H, W), -1, np.int64)
    uv, z = cam.project(mesh.vertices)
    tri_uv, tri_z = uv[mesh.faces], z[mesh.faces]
    visible = (tri_z > 1e-6).all(axis=1)
    for f in np.flatnonzero(visible):
        p = tri_uv[f]
        x0, y0 = np.floor(p.min(axis=0)).astype(int)
        x1, y1 = np.ceil(p.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zt = tri_z[f]
        depth = 1.0 / (w0 / zt[0] + w1 / zt[1] + w2 / zt[2])  # perspective-correct depth
        iy, ix = np.nonzero(inside)
        iy, ix = iy + y0, ix + x0
        d = depth[inside]
        closer = d < zbuf[iy, ix]
        zbuf[iy[closer], ix[closer]] = d[closer]
        fid[iy[closer], ix[closer]] = f

    hit = fid >= 0
    # direction the light travels, in camera coords (OpenCV: +z forward, +y down): from the upper left
    light = np.array([0.3, 0.5, 0.8]) if light_dir is None else light_dir
    light = cam.c2w[:3, :3] @ (light / np.linalg.norm(light))
    normals = mesh.face_normals
    visual = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
    if visual.kind == "vertex":
        base = np.asarray(visual.vertex_colors)[mesh.faces][:, :, :3].mean(axis=1) / 255.0
    else:
        base = np.full((len(mesh.faces), 3), 0.75)
    shade = 0.35 + 0.65 * np.clip(-(normals @ light), 0, 1)
    face_rgb = np.clip(base * shade[:, None], 0, 1)
    rgb = np.full((H, W, 3), background, np.uint8)
    rgb[hit] = (face_rgb[fid[hit]] * 255).astype(np.uint8)
    depth = np.where(hit, zbuf, 0.0)
    return RenderOutput(rgb=rgb, alpha=hit.astype(np.float32), depth=depth)


def render_splats(g: Gaussians, cam: Camera, min_opacity: float = 0.05, background: int = 255,
                  max_radius_px: int = 12) -> RenderOutput:
    """Approximate gaussian render: isotropic screen-space splats, back-to-front alpha compositing."""
    H, W = cam.height, cam.width
    keep = g.opacity >= min_opacity
    uv, z = cam.project(g.means[keep].astype(np.float64))
    op = g.opacity[keep]
    col = g.rgb[keep]
    sigma_world = g.scales[keep].max(axis=1)
    fx = cam.K[0, 0]
    radius = np.clip(2.0 * fx * sigma_world / np.clip(z, 1e-6, None), 0.7, max_radius_px)
    ok = (z > 1e-6) & (uv[:, 0] >= -max_radius_px) & (uv[:, 0] < W + max_radius_px) & \
         (uv[:, 1] >= -max_radius_px) & (uv[:, 1] < H + max_radius_px)
    order = np.flatnonzero(ok)[np.argsort(-z[ok])]  # far to near
    acc_rgb = np.full((H, W, 3), background / 255.0)
    acc_a = np.zeros((H, W))
    depth = np.zeros((H, W))
    for i in order:
        r = radius[i]
        cx, cy = uv[i]
        x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, W))
        y0, y1 = int(max(cy - r, 0)), int(min(cy + r + 1, H))
        if x0 >= x1 or y0 >= y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        a = op[i] * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (r / 2) ** 2))
        a = a[..., None]
        acc_rgb[y0:y1, x0:x1] = acc_rgb[y0:y1, x0:x1] * (1 - a) + col[i] * a
        acc_a[y0:y1, x0:x1] = acc_a[y0:y1, x0:x1] * (1 - a[..., 0]) + a[..., 0]
        depth[y0:y1, x0:x1] = np.where(a[..., 0] > 0.3, z[i], depth[y0:y1, x0:x1])
    return RenderOutput(rgb=(np.clip(acc_rgb, 0, 1) * 255).astype(np.uint8), alpha=acc_a, depth=depth)


def contact_sheet(images: list[np.ndarray], cols: int = 3, pad: int = 4) -> np.ndarray:
    """Tile images into one grid so a single image block carries many views (fewer image tokens)."""
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        sheet[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = im
    return sheet
