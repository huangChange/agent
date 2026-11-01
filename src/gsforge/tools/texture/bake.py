"""UV unwrapping and texture baking.

bake_vertex_colors   CPU. Unwrap (xatlas) and rasterize vertex colors into a texture. Baseline.
bake_from_capture    GPU. Project the original capture photos onto the mesh with z-buffer visibility
                     and view-angle weighting (nvdiffrast). This is the production path for sharp
                     textures and legible text; unobserved texels fall back to the vertex-color bake
                     and are reported as generated.
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image
from pydantic import BaseModel, Field
from scipy import ndimage

from ...core.errors import DependencyMissingError, ToolError
from ...core.models import ArtifactRef
from ..base import ToolContext, ToolResult, tool


def unwrap(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (vertices, faces, uvs, vmapping) of the unwrapped mesh; uvs use OBJ convention (v up)."""
    try:
        import xatlas
    except ImportError as e:
        raise DependencyMissingError("xatlas", "geo") from e
    vmapping, faces, uvs = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    return mesh.vertices[vmapping], faces.astype(np.int64), uvs.astype(np.float64), vmapping


def raster_uv(uvs: np.ndarray, faces: np.ndarray, values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize per-vertex values into UV space. Returns (image (S, S, C) float, mask (S, S))."""
    C = values.shape[1]
    img = np.zeros((size, size, C))
    mask = np.zeros((size, size), bool)
    px = np.stack([uvs[:, 0] * size, (1.0 - uvs[:, 1]) * size], axis=1)  # row 0 = top = v 1
    for f in range(len(faces)):
        tri = px[faces[f]]
        x0, y0 = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(tri.max(axis=0)).astype(int), size - 1)
        if x0 > x1 or y0 > y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = tri
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        w2 = 1 - w0 - w1
        eps = -1e-3  # slight conservative raster closes hairline gaps between charts
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        vals = values[faces[f]]
        iy, ix = np.nonzero(inside)
        img[iy + y0, ix + x0] = (w0[inside, None] * vals[0] + w1[inside, None] * vals[1] + w2[inside, None] * vals[2])
        mask[iy + y0, ix + x0] = True
    return img, mask


def dilate(img: np.ndarray, mask: np.ndarray, pixels: int = 8) -> np.ndarray:
    """Push chart colors outward so mipmapping / bilinear filtering does not bleed background into seams."""
    img, mask = img.copy(), mask.copy()
    for _ in range(pixels):
        grown = ndimage.binary_dilation(mask) & ~mask
        if not grown.any():
            break
        k = np.ones((3, 3))
        s = np.stack([ndimage.convolve(img[..., c] * mask, k, mode="constant") for c in range(img.shape[2])], -1)
        n = ndimage.convolve(mask.astype(float), k, mode="constant")
        img[grown] = s[grown] / n[grown, None]
        mask |= grown
    return img


def textured_mesh(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray, texture: np.ndarray,
                  roughness: float = 0.8, metallic: float = 0.0) -> trimesh.Trimesh:
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture), roughnessFactor=roughness, metallicFactor=metallic)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


class BakeVCParams(BaseModel):
    mesh: ArtifactRef = Field(description="mesh with vertex colors")
    texture_size: int = Field(2048, ge=256, le=8192)


@tool("bake_vertex_colors", stages=("texture",), version="1.0.0")
def bake_vertex_colors(ctx: ToolContext, p: BakeVCParams) -> ToolResult:
    """UV-unwrap the mesh (xatlas) and bake its vertex colors into a base-color texture.
    Output is a textured GLB working mesh plus the texture PNG."""
    mesh = ctx.load_mesh(p.mesh)
    if mesh.visual.kind != "vertex":
        raise ToolError("mesh has no vertex colors; run transfer_vertex_colors first")
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
    verts, faces, uvs, vmap = unwrap(mesh)
    img, mask = raster_uv(uvs, faces, colors[vmap], p.texture_size)
    img = dilate(img, mask)
    tex = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = textured_mesh(verts, faces, uvs, tex)
    return ToolResult(outputs={"mesh": ctx.put_mesh(out, "textured.glb"),
                               "texture": ctx.put_image(tex, "basecolor.png")},
                      metrics={"uv_coverage": float(mask.mean())})


class BakeCaptureParams(BaseModel):
    mesh: ArtifactRef = Field(description="canonical-frame mesh with vertex colors (fallback appearance)")
    cameras: ArtifactRef = Field(description="transforms.json of the original capture")
    images_dir: str
    canonical_from_world: list[list[float]] = Field(description="4x4 transform from canonicalize_pose")
    texture_size: int = 2048
    max_image_side: int = 2048
    angle_power: float = Field(4.0, description="higher = prefer frontal views more strongly (sharper)")
    depth_tolerance: float = Field(0.01, description="relative z-test tolerance for visibility")


@tool("bake_from_capture", stages=("texture",), version="1.0.0", resource="gpu")
def bake_from_capture(ctx: ToolContext, p: BakeCaptureParams) -> ToolResult:
    """Bake a base-color texture by projecting the ORIGINAL capture photos onto the mesh with z-buffer
    visibility and view-angle weighting (sharpest result, best for text/labels). Texels seen by no
    photo fall back to the vertex colors and are reported in metrics['generated_ratio']. GPU only."""
    try:
        import nvdiffrast.torch as dr
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise DependencyMissingError("nvdiffrast", "gpu") from e
    from ...io.cameras import load_transforms_json

    mesh = ctx.load_mesh(p.mesh)
    if mesh.visual.kind != "vertex":
        raise ToolError("mesh needs vertex colors as fallback appearance")
    T = np.asarray(p.canonical_from_world, float)
    cams = load_transforms_json(ctx.store.local_path(p.cameras), p.images_dir)
    verts, faces, uvs, vmap = unwrap(mesh)
    fallback = np.asarray(mesh.visual.vertex_colors)[vmap, :3] / 255.0
    vnormals = trimesh.Trimesh(verts, faces, process=False).vertex_normals

    dev = torch.device("cuda")
    glctx = dr.RasterizeCudaContext()
    S = p.texture_size
    tri = torch.from_numpy(faces.astype(np.int32)).to(dev)
    v = torch.from_numpy(verts.astype(np.float32)).to(dev)
    uv = torch.from_numpy(uvs.astype(np.float32)).to(dev)
    uv_clip = torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], 1)[None]
    rast_uv, _ = dr.rasterize(glctx, uv_clip, tri, resolution=[S, S])
    texel_mask = rast_uv[0, ..., 3] > 0
    pos, _ = dr.interpolate(v[None], rast_uv, tri)
    nrm, _ = dr.interpolate(torch.from_numpy(vnormals.astype(np.float32)).to(dev)[None], rast_uv, tri)
    pos, nrm = pos[0], F.normalize(nrm[0], dim=-1)
    acc = torch.zeros((S, S, 3), device=dev)
    wsum = torch.zeros((S, S), device=dev)
    near, far = 0.01, 100.0
    a, b = (far + near) / (far - near), -2 * far * near / (far - near)

    for cam in cams:
        if cam.image_path is None or not cam.image_path.exists():
            continue
        img = Image.open(cam.image_path).convert("RGB")
        scale = min(1.0, p.max_image_side / max(img.size))
        W, H = int(round(img.size[0] * scale)), int(round(img.size[1] * scale))
        img_t = torch.from_numpy(np.asarray(img.resize((W, H), Image.BILINEAR), np.float32) / 255.0).to(dev)
        K = cam.K.copy()
        K[:2] *= scale
        w2c = torch.from_numpy((np.linalg.inv(cam.c2w) @ np.linalg.inv(T)).astype(np.float32)).to(dev)
        vc = v @ w2c[:3, :3].T + w2c[:3, 3]
        z = vc[:, 2:3]
        clip = torch.cat([(2 * K[0, 0] / W) * vc[:, :1] + (2 * K[0, 2] / W - 1) * z,
                          (2 * K[1, 1] / H) * vc[:, 1:2] + (2 * K[1, 2] / H - 1) * z,
                          a * z + b, z], 1)[None]
        rast_c, _ = dr.rasterize(glctx, clip.contiguous(), tri, resolution=[H, W])
        zbuf, _ = dr.interpolate(z[None].contiguous(), rast_c, tri)
        zbuf = torch.where(rast_c[..., 3:] > 0, zbuf, torch.full_like(zbuf, 1e9))  # (1, H, W, 1)

        pc = pos @ w2c[:3, :3].T + w2c[:3, 3]
        pz = pc[..., 2]
        pu = K[0, 0] * pc[..., 0] / pz.clamp(min=1e-6) + K[0, 2]
        pv = K[1, 1] * pc[..., 1] / pz.clamp(min=1e-6) + K[1, 2]
        grid = torch.stack([2 * pu / W - 1, 2 * pv / H - 1], -1)[None]
        zs = F.grid_sample(zbuf.permute(0, 3, 1, 2), grid, mode="nearest", align_corners=False)[0, 0]
        col = F.grid_sample(img_t.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0]
        center = (T @ np.r_[cam.center, 1.0])[:3].astype(np.float32)  # camera center, canonical frame
        view_dir = F.normalize(torch.from_numpy(center).to(dev) - pos, dim=-1)
        cos = (nrm * view_dir).sum(-1).clamp(min=0)
        inb = (grid.abs() <= 1).all(-1)[0]
        vis = texel_mask & inb & (pz > 0) & ((pz - zs).abs() <= p.depth_tolerance * pz)
        w = torch.where(vis, cos ** p.angle_power, torch.zeros_like(cos))
        acc += col.permute(1, 2, 0) * w[..., None]
        wsum += w

    observed = (wsum > 1e-4).cpu().numpy()[::-1]  # nvdiffrast row 0 = v 0 (bottom); images use top-down
    photo = (acc / wsum.clamp(min=1e-8)[..., None]).cpu().numpy()[::-1]
    mask = texel_mask.cpu().numpy()[::-1]
    fb_img, _ = raster_uv(uvs, faces, fallback, S)
    tex = np.where(observed[..., None], photo, fb_img)
    tex = dilate(tex, mask)
    tex8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    generated = float((mask & ~observed).sum() / max(mask.sum(), 1))
    out = textured_mesh(verts, faces, uvs, tex8)
    return ToolResult(outputs={"mesh": ctx.put_mesh(out, "baked.glb"), "texture": ctx.put_image(tex8, "basecolor.png")},
                      metrics={"generated_ratio": generated, "uv_coverage": float(mask.mean())})
