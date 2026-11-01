"""Split one segmented gaussian group into physical instances (table + remote -> two instances),
and patch the support surface hidden under the objects that were resting on it.

The geometric split is a *proposal*. The segmentation agent role cross-checks it with 2D instance
masks / a VLM and can override it (merge or re-split clusters) before geometry starts.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy import ndimage
from scipy.spatial import cKDTree

from ...core.errors import ToolError
from ...core.models import ArtifactRef
from ...io.gaussians import Gaussians
from ..base import ToolContext, ToolResult, tool


def voxel_components(pts: np.ndarray, voxel: float, max_dim: int = 512) -> np.ndarray:
    """Connected components of points under 26-connectivity of occupied voxels. Returns a label per point.

    Voxel labelling scales to millions of gaussians, unlike radius graphs."""
    lo = pts.min(axis=0)
    span = pts.max(axis=0) - lo
    voxel = max(voxel, float(span.max()) / max_dim)
    ijk = np.floor((pts - lo) / voxel).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    occ = np.zeros(dims, dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    labels, _ = ndimage.label(occ, structure=np.ones((3, 3, 3), dtype=bool))
    return labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]] - 1


def find_support_plane(heights: np.ndarray, bin_size: float, min_ratio: float,
                       relative_strength: float = 0.5) -> float | None:
    """Height of the support sheet: the *highest* horizontal sheet holding at least `min_ratio` of the
    points and at least `relative_strength` x the densest sheet. Choosing the highest matters: a table
    top's underside is as dense as its upper face (which has holes where objects rest on it)."""
    lo, hi = float(heights.min()), float(heights.max())
    if hi - lo < 3 * bin_size:
        return None
    hist, edges = np.histogram(heights, bins=max(int((hi - lo) / bin_size), 1), range=(lo, hi))
    window = np.convolve(hist, np.ones(3, dtype=hist.dtype), mode="same")  # sheet = 3 adjacent bins
    strong = (window >= min_ratio * len(heights)) & (window >= relative_strength * window.max())
    peaks = [i for i in np.flatnonzero(strong)
             if window[i] >= window[max(i - 1, 0)] and window[i] >= window[min(i + 1, len(window) - 1)]]
    if not peaks:
        return None
    top = max(peaks)
    sel = (heights >= edges[max(top - 1, 0)]) & (heights <= edges[min(top + 2, len(edges) - 1)])
    return float(np.median(heights[sel]))


def plane_frame(up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit up vector plus two in-plane axes."""
    up = up / np.linalg.norm(up)
    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, helper)
    e1 /= np.linalg.norm(e1)
    return up, e1, np.cross(up, e1)


def sampling_scales(pts: np.ndarray, spacing: float | None) -> tuple[float, float]:
    """(nearest-neighbour spacing, synthesis spacing). Synthesized gaussians use 2 x NN spacing, which
    matches the areal density of the real (randomly placed) gaussians, so the density field and
    generated_ratio treat real and synthesized surface alike."""
    probe = pts if len(pts) <= 50000 else pts[np.random.default_rng(0).choice(len(pts), 50000, replace=False)]
    nn = float(np.median(cKDTree(probe).query(probe, k=2)[0][:, 1])) if len(probe) > 1 else 1e-3
    return nn, (spacing or 2.0 * nn)


def grid_points_in(mask: np.ndarray, cell: float, lo: np.ndarray, spacing: float, e1: np.ndarray,
                   e2: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Regular points at `spacing` covering the True cells of a 2D grid, lifted into 3D."""
    ci, cj = np.nonzero(mask)
    if len(ci) == 0:
        return np.zeros((0, 3))
    u0, u1 = lo[0] + ci.min() * cell, lo[0] + (ci.max() + 1) * cell
    v0, v1 = lo[1] + cj.min() * cell, lo[1] + (cj.max() + 1) * cell
    uu, vv = np.meshgrid(np.arange(u0 + spacing / 2, u1, spacing), np.arange(v0 + spacing / 2, v1, spacing),
                         indexing="ij")
    u, v = uu.ravel(), vv.ravel()
    inside = mask[np.clip(((u - lo[0]) / cell).astype(int), 0, mask.shape[0] - 1),
                  np.clip(((v - lo[1]) / cell).astype(int), 0, mask.shape[1] - 1)]
    return u[inside, None] * e1 + v[inside, None] * e2 + offset


def _footprint_grid(xy: np.ndarray, cell: float, lo: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    """Occupied cells, closed over small gaps. The grid is padded first: scipy's closing treats the
    outside as background, so without padding it erodes outlines that come near the border."""
    it, pad = 2, 4
    ij = np.clip(np.floor((xy - lo) / cell).astype(np.int64), 0, np.array(dims) - 1)
    grid = np.zeros((dims[0] + 2 * pad, dims[1] + 2 * pad), dtype=bool)
    grid[ij[:, 0] + pad, ij[:, 1] + pad] = True
    # 8-connected (square) element: a cross cannot bridge diagonal gaps in a sparsely sampled outline
    return ndimage.binary_closing(grid, structure=np.ones((3, 3), bool), iterations=it)[pad:-pad, pad:-pad]


def filled_footprint(xy: np.ndarray, cell: float, lo: np.ndarray, dims: tuple[int, int], k: int = 2) -> np.ndarray:
    """Area enclosed by a sparsely sampled outline: dilate by k, fill, erode by k. Filling between the
    two steps matters: plain closing cannot bridge gaps on the convex side of a curved outline (the
    outside of the curve gives the erosion no support), so the outline would stay leaky."""
    pad = k + 2
    ij = np.clip(np.floor((xy - lo) / cell).astype(np.int64), 0, np.array(dims) - 1)
    grid = np.zeros((dims[0] + 2 * pad, dims[1] + 2 * pad), dtype=bool)
    grid[ij[:, 0] + pad, ij[:, 1] + pad] = True
    square = np.ones((3, 3), bool)
    filled = ndimage.binary_fill_holes(ndimage.binary_dilation(grid, structure=square, iterations=k))
    return (ndimage.binary_erosion(filled, structure=square, iterations=k) | grid)[pad:-pad, pad:-pad]


class SplitParams(BaseModel):
    gaussians: ArtifactRef
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    min_opacity: float = 0.1
    link_distance: float | None = Field(None, description="voxel size for connectivity; auto = 3x median "
                                                          "nearest-neighbour spacing")
    min_instance_ratio: float = Field(0.002, description="clusters smaller than this fraction are noise")
    min_plane_ratio: float = Field(0.15, description="a support plane must hold this fraction of points")
    clearance: float | None = Field(None, description="height above the plane that counts as 'resting "
                                                      "on it'; auto = 1.5 x link_distance")


@tool("split_instances", stages=("split",), version="1.5.0")
def split_instances(ctx: ToolContext, p: SplitParams) -> ToolResult:
    """Propose physical instances inside one segment: detects a dominant horizontal support surface
    (table top, shelf), separates clusters resting on it, and splits disconnected parts. Outputs one
    gaussian file per instance (outputs['instance_<k>']) and facts['instances'] describing roles
    (support / supported / standalone), point counts and contact footprints."""
    g = ctx.load_gaussians(p.gaussians)
    up = np.asarray(p.up, dtype=np.float64)
    up /= np.linalg.norm(up)
    solid = g.opacity >= p.min_opacity
    pts = g.means[solid].astype(np.float64)
    if len(pts) < 50:
        raise ToolError(f"segment has only {len(pts)} solid gaussians")

    nn, _ = cKDTree(pts).query(pts, k=2)
    spacing = float(np.median(nn[:, 1]))
    link = p.link_distance or 3.0 * spacing
    clearance = p.clearance or 1.5 * link
    heights = pts @ up
    min_pts = max(int(p.min_instance_ratio * len(pts)), 20)

    plane_h = find_support_plane(heights, bin_size=link, min_ratio=p.min_plane_ratio)
    groups: list[tuple[str, np.ndarray]] = []  # (role, indices into pts)
    contacts: dict[int, list[list[float]]] = {}
    if plane_h is not None:
        # 2D frame on the plane
        helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(up, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        xy = np.stack([pts @ e1, pts @ e2], axis=1)
        on_plane = np.abs(heights - plane_h) <= link
        above = heights > plane_h + clearance * 0.5
        cell = link
        lo = xy[on_plane].min(axis=0) - 2 * cell
        dims = tuple((np.ceil((xy[on_plane].max(axis=0) + 2 * cell - lo) / cell)).astype(int))
        interior = ndimage.binary_erosion(filled_footprint(xy[on_plane], cell, lo, dims), iterations=2)

        above_idx = np.flatnonzero(above)
        labels = voxel_components(pts[above_idx], link) if len(above_idx) else np.zeros(0, int)
        support_mask = ~above
        for lab in np.unique(labels):
            comp = above_idx[labels == lab]
            if len(comp) < min_pts:
                support_mask[comp] = True  # too small to be an object; keep with the support
                continue
            near_plane = comp[heights[comp] <= plane_h + clearance * 2]
            base = near_plane if len(near_plane) else comp
            ij = np.floor((xy[base] - lo) / cell).astype(np.int64)
            inside = (ij >= 0).all(axis=1) & (ij[:, 0] < dims[0]) & (ij[:, 1] < dims[1])
            frac_inside = float(interior[ij[inside, 0], ij[inside, 1]].mean()) if inside.any() else 0.0
            lowest = float(heights[comp].min())
            resting = frac_inside >= 0.8 and lowest <= plane_h + clearance * 2
            if resting:
                groups.append(("supported", comp))
                base_xy = xy[base]
                contacts[len(groups) - 1] = [base_xy.min(axis=0).tolist(), base_xy.max(axis=0).tolist()]
            else:  # e.g. a chair back rising from the seat edge: part of the support object
                support_mask[comp] = True
        # Points between the plane and the "above" threshold include the lowest part of the resting objects'
        # walls. Grow each resting object down through that band so it keeps its bottom edge (needed to
        # trace its footprint later). Height alone can't separate the sheet from a wall's foot within the
        # sheet's tolerance, but orientation can: sheet gaussians face up, wall gaussians face sideways.
        facing_up = np.abs(g.normals()[solid] @ up) >= 0.7
        band = np.flatnonzero(support_mask & ~facing_up & (heights >= plane_h - 2 * spacing)
                              & (heights <= plane_h + clearance * 2))
        resting = [(k, idx) for k, (role, idx) in enumerate(groups) if role == "supported"]
        if len(band) and resting:
            for _ in range(int(np.ceil(clearance / spacing)) + 1):
                owner_pts = np.concatenate([pts[idx] for _, idx in resting])
                owner_lab = np.concatenate([np.full(len(idx), k) for k, idx in resting])
                d, nb = cKDTree(owner_pts).query(pts[band], k=1, distance_upper_bound=link)
                hit = np.isfinite(d)
                if not hit.any():
                    break
                for k, _ in resting:
                    grabbed = band[hit & (owner_lab[np.minimum(nb, len(owner_lab) - 1)] == k)]
                    groups[k] = (groups[k][0], np.concatenate([groups[k][1], grabbed]))
                support_mask[band[hit]] = False
                band = band[~hit]
                resting = [(k, groups[k][1]) for k, _ in resting]
                if not len(band):
                    break
        sup_idx = np.flatnonzero(support_mask)
        sup_labels = voxel_components(pts[sup_idx], link)
        counts = np.bincount(sup_labels)
        main = int(np.argmax(counts))
        groups.insert(0, ("support", sup_idx[sup_labels == main]))
        contacts = {k + 1: v for k, v in contacts.items()}
        for lab in np.flatnonzero(counts >= min_pts):
            if lab != main:
                groups.append(("standalone", sup_idx[sup_labels == lab]))
    else:
        labels = voxel_components(pts, link)
        counts = np.bincount(labels)
        for lab in np.argsort(-counts):
            if counts[lab] >= min_pts:
                groups.append(("standalone", np.flatnonzero(labels == lab)))

    # Assign every gaussian (incl. translucent ones excluded above) to the nearest instance.
    owner_solid = np.full(len(pts), -1)
    for k, (_, idx) in enumerate(groups):
        owner_solid[idx] = k
    assigned = owner_solid >= 0
    tree = cKDTree(pts[assigned])
    _, nearest = tree.query(g.means.astype(np.float64), k=1)
    owner_all = owner_solid[assigned][nearest]
    dist_ok = np.linalg.norm(g.means - pts[assigned][nearest], axis=1) <= 4 * link
    owner_all[~dist_ok] = -1

    outputs, instances = {}, []
    for k, (role, _) in enumerate(groups):
        sub: Gaussians = g.subset(owner_all == k)
        ref = ctx.put_gaussians(sub, f"instance_{k}.ply", role=role)
        outputs[f"instance_{k}"] = ref
        lo3, hi3 = sub.bbox()
        instances.append({"key": f"instance_{k}", "role": role, "count": len(sub),
                          "bbox_min": lo3.tolist(), "bbox_max": hi3.tolist(),
                          "contact_xy": contacts.get(k)})
    return ToolResult(
        outputs=outputs,
        facts={"instances": instances, "support_plane_height": plane_h, "link_distance": link,
               "up": up.tolist()},
        metrics={"instance_count": len(instances), "unassigned": int((owner_all < 0).sum())},
    )


class PatchParams(BaseModel):
    gaussians: ArtifactRef = Field(description="support instance gaussians")
    plane_height: float
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    spacing: float | None = Field(None, description="spacing of synthesized gaussians; default: matched to "
                                                      "this instance's own sampling density")
    min_opacity: float = 0.1


@tool("patch_support_surface", stages=("split", "geometry"), version="1.4.0")
def patch_support_surface(ctx: ToolContext, p: PatchParams) -> ToolResult:
    """Fill holes in a support surface left by objects that rested on it (the table area under a
    remote). Detects interior holes in the plane's occupancy footprint and synthesizes flat gaussians
    there, colored from the hole rim. Synthesized gaussians are counted in metrics['generated_ratio']."""
    g = ctx.load_gaussians(p.gaussians)
    up, e1, e2 = plane_frame(np.asarray(p.up, float))
    solid = g.opacity >= p.min_opacity
    nn, spacing = sampling_scales(g.means[solid], p.spacing)
    h = g.means @ up
    on_plane = solid & (np.abs(h - p.plane_height) <= 6 * nn)
    if on_plane.sum() < 10:
        return ToolResult(outputs={"gaussians": p.gaussians}, metrics={"generated_ratio": 0.0},
                          warnings=["no plane points near the given height"])
    xy = np.stack([g.means[on_plane] @ e1, g.means[on_plane] @ e2], axis=1)
    cell = 3 * nn
    lo = xy.min(axis=0) - cell
    dims = tuple(np.ceil((xy.max(axis=0) + cell - lo) / cell).astype(int))
    grid = _footprint_grid(xy, cell, lo, dims)
    holes = ndimage.binary_fill_holes(grid) & ~grid
    if not holes.any():
        return ToolResult(outputs={"gaussians": p.gaussians}, metrics={"generated_ratio": 0.0})
    new_pts = grid_points_in(holes, cell, lo, spacing, e1, e2, p.plane_height * up)
    # color: inverse-distance blend of the k nearest real plane gaussians (rim)
    plane_pts = g.means[on_plane]
    d, nb = cKDTree(plane_pts).query(new_pts, k=8)
    w = 1.0 / np.clip(d, 1e-6, None)
    dc = (g.sh_dc[on_plane][nb] * w[..., None]).sum(axis=1) / w.sum(axis=1, keepdims=True)
    template = g.subset(np.flatnonzero(on_plane)[nb[:, 0]])
    template.means = new_pts.astype(np.float32)
    template.sh_dc = dc.astype(np.float32)
    template.sh_rest = np.zeros((len(new_pts), g.sh_rest.shape[1]), np.float32)
    patched = Gaussians.concat([g, template])
    ratio = len(new_pts) / len(patched)
    return ToolResult(outputs={"gaussians": ctx.put_gaussians(patched, "patched.ply", generated=len(new_pts))},
                      metrics={"generated_ratio": ratio, "generated_count": len(new_pts)},
                      facts={"hole_cells": int(holes.sum())})


class BaseParams(BaseModel):
    gaussians: ArtifactRef
    base_height: float | None = Field(None, description="height of the surface the object rests on; "
                                                        "default = the object's own lowest points")
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    spacing: float | None = Field(None, description="spacing of synthesized gaussians; default: matched to "
                                                      "this instance's own sampling density")
    band: float | None = Field(None, description="height band above the base used to trace the footprint; "
                                                 "default = 3 x spacing")
    min_opacity: float = 0.1


@tool("close_contact_base", stages=("split", "geometry"), version="1.6.0")
def close_contact_base(ctx: ToolContext, p: BaseParams) -> ToolResult:
    """Close the never-observed underside of an object that rests on a surface (remote on a table,
    table legs on the floor). Traces the object's footprint from its lowest band of points, fills it,
    and synthesizes flat gaussians at the contact height, colored from the nearest real points. This
    lets extraction produce a closed solid. Synthesized gaussians are reported as generated."""
    g = ctx.load_gaussians(p.gaussians)
    up, e1, e2 = plane_frame(np.asarray(p.up, float))
    solid = g.opacity >= p.min_opacity
    nn, spacing = sampling_scales(g.means[solid], p.spacing)
    h = g.means @ up
    base = float(np.percentile(h[solid], 0.5)) if p.base_height is None else p.base_height
    band = p.band or 6 * nn
    low = solid & (h <= base + band)
    if low.sum() < 10:
        return ToolResult(outputs={"gaussians": p.gaussians}, metrics={"generated_ratio": 0.0},
                          warnings=["no points near the base height; nothing closed"])
    xy = np.stack([g.means[low] @ e1, g.means[low] @ e2], axis=1)
    cell = 3 * nn
    lo = xy.min(axis=0) - 2 * cell
    dims = tuple(np.ceil((xy.max(axis=0) + 2 * cell - lo) / cell).astype(int))
    footprint = filled_footprint(xy, cell, lo, dims)  # sampled outline -> enclosed area
    new_pts = grid_points_in(footprint, cell, lo, spacing, e1, e2, base * up)
    # skip spots where the underside WAS observed. Only points at the base height count (a flat object's
    # top lies in the tracing band too), and the test is per point so the cap reaches the walls.
    at_base = solid & (np.abs(h - base) <= 3 * nn)
    if at_base.any():
        d_obs, _ = cKDTree(g.means[at_base]).query(new_pts, k=1)
        new_pts = new_pts[d_obs > spacing]
    if len(new_pts) == 0:
        return ToolResult(outputs={"gaussians": p.gaussians}, metrics={"generated_ratio": 0.0})
    src = np.flatnonzero(low)
    d, nb = cKDTree(g.means[src]).query(new_pts, k=8)
    w = 1.0 / np.clip(d, 1e-6, None)
    dc = (g.sh_dc[src][nb] * w[..., None]).sum(axis=1) / w.sum(axis=1, keepdims=True)
    cap = g.subset(src[nb[:, 0]])
    cap.means = new_pts.astype(np.float32)
    cap.sh_dc = dc.astype(np.float32)
    cap.sh_rest = np.zeros((len(new_pts), g.sh_rest.shape[1]), np.float32)
    # flat, facing down: rotate local z (the thin axis) onto -up
    from scipy.spatial.transform import Rotation

    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, -up)
    angle = np.arctan2(np.linalg.norm(axis), float(z @ -up))
    rv = axis / np.linalg.norm(axis) * angle if np.linalg.norm(axis) > 1e-9 else np.array([np.pi, 0.0, 0.0])
    cap.quats_wxyz = np.tile(Rotation.from_rotvec(rv).as_quat()[[3, 0, 1, 2]], (len(new_pts), 1)).astype(np.float32)
    s = np.log(np.full((len(new_pts), 3), spacing * 1.5))
    s[:, 2] = np.log(spacing * 0.15)
    cap.scales_log = s.astype(np.float32)
    closed = Gaussians.concat([g, cap])
    ratio = len(new_pts) / len(closed)
    return ToolResult(outputs={"gaussians": ctx.put_gaussians(closed, "base_closed.ply", generated=len(new_pts))},
                      metrics={"generated_ratio": ratio, "generated_count": len(new_pts)},
                      facts={"base_height": base, "footprint_cells": int(footprint.sum())})
