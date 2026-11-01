"""Quality metrics. Pure functions over meshes / gaussians so they can be unit tested and reused by
gates, tools and the eval harness alike."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree

from ..core.config import CONFIG_DIR
from ..io.gaussians import Gaussians


def topology_metrics(mesh: trimesh.Trimesh) -> dict[str, float]:
    edges = mesh.edges_sorted
    if len(edges):
        _, counts = np.unique(edges, axis=0, return_counts=True)
        boundary, non_manifold = int((counts == 1).sum()), int((counts > 2).sum())
    else:
        boundary = non_manifold = 0
    degenerate = int((~mesh.nondegenerate_faces()).sum()) if len(mesh.faces) else 0
    return {
        "faces": float(len(mesh.faces)),
        "vertices": float(len(mesh.vertices)),
        "watertight": float(mesh.is_watertight),
        "winding_consistent": float(mesh.is_winding_consistent),
        "components": float(mesh.body_count),
        "boundary_edges": float(boundary),
        "non_manifold_edges": float(non_manifold),
        "degenerate_faces": float(degenerate),
    }


def fidelity_metrics(mesh: trimesh.Trimesh, gaussians: Gaussians, min_opacity: float = 0.3,
                     tau_ratio: float = 0.01, n_samples: int = 200000, seed: int = 0) -> dict[str, float]:
    """Two-sided agreement between mesh surface and solid gaussian centers, normalized by bbox diagonal.

    recall    = fraction of gaussians within tau of the mesh   (missing geometry lowers it)
    precision = fraction of mesh surface within tau of gaussians (hallucinated / bloated geometry)
    Distances to the mesh are approximated with a dense surface sample, which is accurate to ~ the
    sample spacing and orders of magnitude faster than exact point-triangle queries.

    tau = max(tau_ratio * diag, 1.5 * gaussian spacing): the tolerance never asks for more precision
    than the input resolves (matters for small objects only; dense input keeps the ratio)."""
    pts = gaussians.means[gaussians.opacity >= min_opacity].astype(np.float64)
    if len(pts) == 0 or len(mesh.faces) == 0:
        return {"recall": 0.0, "precision": 0.0, "fscore": 0.0, "gs_to_mesh_p95": 1.0, "mesh_to_gs_p95": 1.0}
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    rng = np.random.default_rng(seed)
    probe = pts if len(pts) <= 50000 else pts[rng.choice(len(pts), 50000, replace=False)]
    spacing = float(np.median(cKDTree(probe).query(probe, k=2)[0][:, 1]))
    tau = max(tau_ratio * diag, 1.5 * spacing)
    if len(pts) > n_samples:
        pts = pts[rng.choice(len(pts), n_samples, replace=False)]
    surf, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    d_g2m, _ = cKDTree(surf).query(pts, k=1)
    d_m2g, _ = cKDTree(pts).query(surf, k=1)
    recall = float((d_g2m <= tau).mean())
    precision = float((d_m2g <= tau).mean())
    f = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "fscore": f,
        "gs_to_mesh_p95": float(np.percentile(d_g2m, 95) / diag),
        "mesh_to_gs_p95": float(np.percentile(d_m2g, 95) / diag),
        "chamfer_l1": float((d_g2m.mean() + d_m2g.mean()) / 2 / diag),
        "tolerance_m": tau,
    }


def silhouette_iou(mesh_masks: list[np.ndarray], ref_masks: list[np.ndarray]) -> dict[str, float]:
    ious = []
    for a, b in zip(mesh_masks, ref_masks):
        union = np.logical_or(a, b).sum()
        ious.append(np.logical_and(a, b).sum() / union if union else 1.0)
    return {"silhouette_iou_mean": float(np.mean(ious)), "silhouette_iou_min": float(np.min(ious))}


_PRIORS_CACHE: dict[str, Any] = {}


def load_priors(name: str) -> dict[str, Any]:
    if name not in _PRIORS_CACHE:
        path = CONFIG_DIR / "priors" / f"{name}.yaml"
        _PRIORS_CACHE[name] = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _PRIORS_CACHE[name]


def scale_plausibility(label: str | None, size_m: np.ndarray) -> dict[str, float]:
    """1.0 if the largest dimension is within the category's [min, max] in configs/priors/sizes.yaml,
    decaying with the log-ratio outside it. Unknown labels return no metric."""
    if not label:
        return {}
    prior = load_priors("sizes").get(label.lower())
    if not prior:
        return {}
    longest = float(np.max(size_m))
    lo, hi = float(prior["min"]), float(prior["max"])
    if lo <= longest <= hi:
        score = 1.0
    else:
        ratio = longest / lo if longest < lo else longest / hi
        score = float(np.exp(-abs(np.log(ratio))))
    return {"scale_plausibility": score, "longest_dim_m": longest}
