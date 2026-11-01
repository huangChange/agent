"""Remove floaters and segmentation bleed from a segmented gaussian group."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

from ...core.models import ArtifactRef
from ..base import ToolContext, ToolResult, tool


class CleanParams(BaseModel):
    gaussians: ArtifactRef
    min_opacity: float = Field(0.02, description="drop gaussians below this activated opacity")
    max_scale_ratio: float = Field(0.15, description="drop gaussians whose largest axis exceeds this "
                                                     "fraction of the bbox diagonal (background blobs)")
    outlier_k: int = Field(16, ge=4, description="neighbors for statistical outlier removal")
    outlier_std: float = Field(3.0, description="drop points whose mean kNN distance > mean + std * this")


@tool("clean_gaussians", stages=("prepare",), version="1.0.0")
def clean_gaussians(ctx: ToolContext, p: CleanParams) -> ToolResult:
    """Drop near-transparent, oversized and statistically isolated gaussians (floaters, bleed from
    neighbouring segments). Returns the cleaned gaussians and how many were removed by each rule."""
    g = ctx.load_gaussians(p.gaussians)
    n0 = len(g)
    keep = g.opacity >= p.min_opacity
    lo, hi = np.percentile(g.means[keep], [1, 99], axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    keep_scale = g.scales.max(axis=1) <= p.max_scale_ratio * diag
    keep &= keep_scale

    idx = np.flatnonzero(keep)
    removed_outliers = 0
    if len(idx) > p.outlier_k + 1:
        pts = g.means[idx]
        d, _ = cKDTree(pts).query(pts, k=p.outlier_k + 1)
        mean_d = d[:, 1:].mean(axis=1)
        inlier = mean_d <= mean_d.mean() + p.outlier_std * mean_d.std()
        removed_outliers = int((~inlier).sum())
        keep[idx[~inlier]] = False

    out = g.subset(keep)
    return ToolResult(
        outputs={"gaussians": ctx.put_gaussians(out, "clean.ply")},
        metrics={"input_count": n0, "output_count": len(out), "removed_outliers": removed_outliers,
                 "removed_ratio": 1.0 - len(out) / max(n0, 1)},
    )
