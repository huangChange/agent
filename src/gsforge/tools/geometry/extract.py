"""Geometry candidates. The pipeline runs every configured candidate and keeps the one with the best
QA score against the gaussians, so no single method has to be right for every object.

  density_mc       CPU. Opacity-weighted density field -> solid occupancy -> marching cubes. Robust,
                   always watertight, slightly smooth. The baseline that never fails to produce a mesh.
  scene_mesh_crop  CPU. Cut the instance out of the provided scene mesh. Sharp where that mesh is good;
                   open at contact regions, so it usually needs repair.
  tsdf_render      GPU. Render depth of the instance gaussians from many views (gsplat) and fuse a TSDF.
                   Best detail for 3DGS input; requires the `gpu` extra.
"""

from __future__ import annotations

import numpy as np
import trimesh
from pydantic import BaseModel, Field
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

from ...core.errors import ToolError
from ...core.models import ArtifactRef
from ...io.cameras import Camera, orbit_cameras
from ..base import ToolContext, ToolResult, tool


def _grid(lo: np.ndarray, hi: np.ndarray, resolution: int, pad: int = 4,
          min_voxel: float = 0.0) -> tuple[np.ndarray, float, np.ndarray]:
    voxel = max(float((hi - lo).max()) / resolution, min_voxel)
    origin = lo - pad * voxel
    dims = np.ceil((hi - lo) / voxel).astype(int) + 2 * pad + 1
    return origin, voxel, dims


def _marching_cubes(field: np.ndarray, level: float, origin: np.ndarray, voxel: float) -> trimesh.Trimesh:
    if not (field.min() < level < field.max()):
        raise ToolError("iso level outside field range; nothing to extract")
    verts, faces, _, _ = measure.marching_cubes(field, level=level, spacing=(voxel, voxel, voxel))
    mesh = trimesh.Trimesh(verts + origin, faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())  # zero-area triangles where the iso surface grazes a corner
    mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:  # sub-voxel specks around the main surface
        areas = np.array([p.area for p in parts])
        mesh = trimesh.util.concatenate([p for p, a in zip(parts, areas) if a >= 0.01 * areas.max()])
    # orient outward regardless of the library's convention (is_volume would already demand volume > 0)
    if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume < 0:
        mesh.invert()
    return mesh


class DensityParams(BaseModel):
    gaussians: ArtifactRef
    resolution: int = Field(192, ge=32, le=768, description="voxels along the longest bbox axis")
    min_opacity: float = 0.05
    level_ratio: float = Field(0.3, description="iso level as a fraction of the median density at "
                                                "gaussian centers; lower = fatter surface")
    fill_cavities: bool = Field(True, description="fill fully enclosed cavities -> one outer shell")
    compensate_offset: bool = Field(True, description="move the surface back inward by the blur offset "
                                                      "(thin sheets are kept as they are)")


@tool("extract_density_mc", stages=("geometry",), version="1.3.0")
def extract_density_mc(ctx: ToolContext, p: DensityParams) -> ToolResult:
    """Extract a watertight mesh from the opacity-weighted gaussian density field via marching cubes.
    Robust baseline candidate; loses detail below ~2 voxels. Raise `resolution` for small details,
    lower `level_ratio` if thin parts disappear."""
    g = ctx.load_gaussians(p.gaussians)
    keep = g.opacity >= p.min_opacity
    pts, w = g.means[keep].astype(np.float64), g.opacity[keep]
    if len(pts) < 20:
        raise ToolError("not enough solid gaussians for extraction")
    lo, hi = np.percentile(pts, 0.1, axis=0), np.percentile(pts, 99.9, axis=0)
    sub = pts if len(pts) <= 50000 else pts[np.random.default_rng(0).choice(len(pts), 50000, replace=False)]
    spacing = float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1]))
    # a grid finer than the sampling only resolves the gaps between gaussians: cap the resolution
    # pad > blur reach (1.55 sigma, sigma <= 4) + closing radius, so morphology never meets the border
    origin, voxel, dims = _grid(lo, hi, p.resolution, pad=12, min_voxel=0.75 * spacing)
    ijk = np.rint((pts - origin) / voxel).astype(np.int64)  # nearest grid node: MC samples at nodes
    inside = ((ijk >= 0) & (ijk < dims)).all(axis=1)
    density = np.zeros(dims, np.float32)
    np.add.at(density, tuple(ijk[inside].T), w[inside])
    # Blur to ~2x the MEAN point spacing (~2x nearest-neighbour for random placement): with fewer than
    # ~20 points under the kernel, density noise dips below the iso level and the shell gets holes.
    # The offset compensation below moves the surface back, so wide blur costs detail, not position.
    sigma = float(np.clip(max(np.median(g.scales[keep].max(axis=1)), 4.0 * spacing) / voxel, 1.5, 6.0))
    density = ndimage.gaussian_filter(density, sigma)
    at_points = density[tuple(ijk[inside].T)]
    level = p.level_ratio * float(np.median(at_points))
    occ = density > level
    ball = ndimage.generate_binary_structure(3, 1)
    occ = ndimage.binary_closing(occ, structure=ball, iterations=2)  # seal pinholes before filling
    if p.fill_cavities:
        occ = ndimage.binary_fill_holes(occ)
    labels, n = ndimage.label(occ)
    if n > 1:  # drop specks: keep components with >= 1% of the largest one's volume
        sizes = np.bincount(labels.ravel())[1:]
        occ = np.isin(labels, 1 + np.flatnonzero(sizes >= 0.01 * sizes.max()))
    if p.compensate_offset:
        # the iso level of a blurred sheet lies sigma*sqrt(2 ln(1/level_ratio)) outside the true surface
        off = sigma * np.sqrt(2 * np.log(1 / p.level_ratio))
        shrunk = ndimage.distance_transform_edt(occ) > off
        # parts thinner than 2*off would vanish: keep them at their blurred thickness instead
        near_shrunk = ndimage.distance_transform_edt(~shrunk) <= off + 1.5 if shrunk.any() else np.zeros_like(occ)
        thin = occ & ~near_shrunk
        thin_lab, n_thin = ndimage.label(thin)
        if n_thin:  # a real thin part is a sizeable sheet, not the leftover rim voxels of a solid
            sizes = np.bincount(thin_lab.ravel())[1:]
            thin = np.isin(thin_lab, 1 + np.flatnonzero(sizes >= max(27, 0.002 * occ.sum())))
        occ = shrunk | thin
    field = ndimage.gaussian_filter(occ.astype(np.float32), 1.0)
    mesh = _marching_cubes(field, 0.5, origin, voxel)
    return ToolResult(outputs={"mesh": ctx.put_mesh(mesh, "density_mc.ply", method="density_mc")},
                      metrics={"voxel_size": voxel, "faces": len(mesh.faces)})


class CropParams(BaseModel):
    scene_mesh: ArtifactRef
    gaussians: ArtifactRef
    max_distance: float | None = Field(None, description="keep faces within this distance of the instance "
                                                         "gaussians; auto = 2 x median gaussian spacing")
    min_opacity: float = 0.1


@tool("extract_scene_mesh_crop", stages=("geometry",), version="1.0.0")
def extract_scene_mesh_crop(ctx: ToolContext, p: CropParams) -> ToolResult:
    """Cut this instance out of the provided scene mesh: keeps faces whose three vertices all lie near
    the instance's gaussians. Quality depends on the scene mesh; contact regions come out open."""
    g = ctx.load_gaussians(p.gaussians)
    pts = g.means[g.opacity >= p.min_opacity].astype(np.float64)
    tree = cKDTree(pts)
    if p.max_distance is None:
        d, _ = tree.query(pts, k=2)
        max_d = 2.0 * float(np.median(d[:, 1])) + 1e-6
        max_d = max(max_d, 0.005)
    else:
        max_d = p.max_distance
    scene = ctx.load_mesh(p.scene_mesh)
    lo, hi = pts.min(axis=0) - 2 * max_d, pts.max(axis=0) + 2 * max_d
    tri = scene.triangles
    overlap = ((tri.max(axis=1) >= lo) & (tri.min(axis=1) <= hi)).all(axis=1)  # face bbox vs instance bbox
    if not overlap.any():
        raise ToolError("scene mesh does not overlap this instance")
    region = scene.submesh([np.flatnonzero(overlap)], append=True)
    # large triangles (coarse or decimated scene meshes) are split so the per-vertex test is meaningful
    v, f = trimesh.remesh.subdivide_to_size(region.vertices, region.faces, max_edge=2 * max_d, max_iter=12)
    region = trimesh.Trimesh(v, f, process=True)
    dist, _ = tree.query(region.vertices, k=1, distance_upper_bound=max_d)
    face_mask = np.isfinite(dist)[region.faces].all(axis=1)
    if face_mask.sum() < 10:
        raise ToolError("scene mesh has (almost) no faces near this instance")
    mesh = region.submesh([np.flatnonzero(face_mask)], append=True)
    return ToolResult(outputs={"mesh": ctx.put_mesh(mesh, "scene_crop.ply", method="scene_mesh_crop")},
                      metrics={"faces": len(mesh.faces), "max_distance": max_d})


def tsdf_fuse(depths: list[np.ndarray], cams: list[Camera], lo: np.ndarray, hi: np.ndarray,
              resolution: int = 256, trunc_voxels: float = 4.0,
              weights: list[np.ndarray] | None = None) -> trimesh.Trimesh:
    """Classic TSDF integration (Curless & Levoy) on CPU. Depth maps are z-depth in meters, 0 = invalid."""
    origin, voxel, dims = _grid(lo, hi, resolution)
    trunc = trunc_voxels * voxel
    axes = [origin[i] + voxel * np.arange(dims[i]) for i in range(3)]
    tsdf = np.ones(int(np.prod(dims)), np.float32)
    wsum = np.zeros(int(np.prod(dims)), np.float32)
    chunk = 1 << 21
    total = int(np.prod(dims))
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total))
        i, j, k = np.unravel_index(idx, dims)
        pts = np.stack([axes[0][i], axes[1][j], axes[2][k]], axis=1)
        for v, (depth, cam) in enumerate(zip(depths, cams)):
            uv, z = cam.project(pts)
            u, vv = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
            valid = (z > 1e-6) & (u >= 0) & (u < cam.width) & (vv >= 0) & (vv < cam.height)
            d = np.zeros_like(z)
            d[valid] = depth[vv[valid], u[valid]]
            valid &= d > 0
            sdf = d - z
            valid &= sdf >= -trunc
            if not valid.any():
                continue
            s = np.clip(sdf[valid] / trunc, -1.0, 1.0)
            wv = np.ones_like(s) if weights is None else weights[v][vv[valid], u[valid]]
            sel = idx[valid]
            cur_w = wsum[sel]
            tsdf[sel] = (tsdf[sel] * cur_w + s * wv) / np.clip(cur_w + wv, 1e-9, None)
            wsum[sel] = cur_w + wv
    tsdf[wsum == 0] = 1.0
    field = -tsdf.reshape(dims)  # inside (negative sdf) -> high values, matching _marching_cubes
    return _marching_cubes(field, 0.0, origin, voxel)


class TSDFParams(BaseModel):
    gaussians: ArtifactRef
    views: int = Field(48, ge=8, le=256)
    image_size: int = 768
    resolution: int = 256
    alpha_threshold: float = Field(0.5, description="ignore depth where accumulated alpha is below this")


@tool("extract_tsdf_render", stages=("geometry",), version="1.0.0", resource="gpu")
def extract_tsdf_render(ctx: ToolContext, p: TSDFParams) -> ToolResult:
    """GPU candidate: render expected depth of the instance gaussians from a sphere of views and fuse a
    TSDF. Highest detail for 3DGS input. Needs a CUDA worker with the `gpu` extra (gsplat)."""
    from ...render.gs_render import render_gaussians  # lazy: pulls torch / gsplat

    g = ctx.load_gaussians(p.gaussians)
    lo, hi = g.bbox(min_opacity=0.1)
    center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) * 1.2
    cams = []
    for el in (-20.0, 15.0, 45.0, 75.0):
        cams += orbit_cameras(center, radius, p.views // 4, elevation_deg=el, size=p.image_size)
    depths = []
    for cam in cams:
        out = render_gaussians(g, cam, with_depth=True)
        depth = np.where(out.alpha >= p.alpha_threshold, out.depth, 0.0)
        depths.append(depth)
    mesh = tsdf_fuse(depths, cams, lo, hi, resolution=p.resolution)
    return ToolResult(outputs={"mesh": ctx.put_mesh(mesh, "tsdf.ply", method="tsdf_render")},
                      metrics={"faces": len(mesh.faces), "views": len(cams)})
