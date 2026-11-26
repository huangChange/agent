"""Geometry helpers (numpy + scipy only)."""
from __future__ import annotations

import numpy as np


def sample_surface(V, F, n, rng):
    """Area-weighted uniform sampling of a triangle mesh. Returns (points, normals, total_area)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cross, axis=1)
    total = float(area2.sum()) / 2.0
    if total <= 0:
        raise ValueError("mesh has zero surface area")
    idx = rng.choice(len(F), size=n, p=area2 / area2.sum())
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    pa, pb, pc = a[idx], b[idx], c[idx]
    pts = pa + u[:, None] * (pb - pa) + v[:, None] * (pc - pa)
    nrm = cross[idx] / np.maximum(area2[idx, None], 1e-30)
    return pts, nrm, total


def fit_plane(P):
    """Least-squares plane. Returns unit normal n and offset d with n.x + d = 0."""
    c = P.mean(axis=0)
    X = P - c
    w, vecs = np.linalg.eigh(X.T @ X)
    n = vecs[:, 0]
    return n, -float(n @ c)


def ransac_plane(P, thresh, rng, iters=1000, mode="any", axis=None, eval_n=50000):
    """RANSAC plane fit.

    mode="any": 3-point hypotheses.
    mode="vertical": planes that contain direction `axis` (normal perpendicular to axis).
    Returns (n, d, inlier_mask) or None.
    """
    if len(P) < 3:
        return None
    E = P if len(P) <= eval_n else P[rng.choice(len(P), eval_n, replace=False)]
    best_score, best = -1, None
    chunk = 100
    for start in range(0, iters, chunk):
        m = min(chunk, iters - start)
        if mode == "vertical":
            ij = rng.integers(0, len(E), (m, 2))
            base = E[ij[:, 0]]
            nrm = np.cross(E[ij[:, 1]] - base, axis[None, :])
        else:
            ijk = rng.integers(0, len(E), (m, 3))
            base = E[ijk[:, 0]]
            nrm = np.cross(E[ijk[:, 1]] - base, E[ijk[:, 2]] - base)
        L = np.linalg.norm(nrm, axis=1)
        ok = L > 1e-12
        if not ok.any():
            continue
        nrm = nrm[ok] / L[ok, None]
        dd = -(nrm * base[ok]).sum(axis=1)
        score = (np.abs(E @ nrm.T + dd[None, :]) < thresh).sum(axis=0)
        j = int(score.argmax())
        if score[j] > best_score:
            best_score, best = int(score[j]), (nrm[j], float(dd[j]))
    if best is None:
        return None
    n, d = best
    for _ in range(3):
        inl = np.abs(P @ n + d) < thresh
        if inl.sum() < 3:
            break
        n2, d2 = fit_plane(P[inl])
        if mode == "vertical":
            n2 = n2 - (n2 @ axis) * axis
            n2 /= np.linalg.norm(n2)
            d2 = -float(n2 @ P[inl].mean(axis=0))
        if n2 @ n < 0:
            n2, d2 = -n2, -d2
        n, d = n2, d2
    inl = np.abs(P @ n + d) < thresh
    return n, d, inl


def extract_planes(P, k, thresh, min_inliers, rng, mode="any", axis=None, iters=1000):
    """Sequential RANSAC. Returns list of dicts {n, d, idx} with idx into P."""
    remaining = np.arange(len(P))
    planes = []
    for _ in range(k):
        if len(remaining) < min_inliers:
            break
        res = ransac_plane(P[remaining], thresh, rng, iters=iters, mode=mode, axis=axis)
        if res is None:
            break
        n, d, inl = res
        if inl.sum() < min_inliers:
            break
        planes.append({"n": n, "d": d, "idx": remaining[inl]})
        remaining = remaining[~inl]
    return planes


def rotation_a_to_b(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        u = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(a, [0, 1.0, 0])
        u /= np.linalg.norm(u)
        return 2 * np.outer(u, u) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def weighted_median(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, cw[-1] / 2.0)])


def angle_deg(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


AXES = {
    "+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
    "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]),
    "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0]),
}


def nearest_axis(v):
    name = max(AXES, key=lambda k: float(AXES[k] @ v) / np.linalg.norm(v))
    return name, angle_deg(AXES[name], v)


def look_at_c2w(eye, target, up):
    """OpenCV convention camera-to-world (x right, y down, z forward)."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    up = np.asarray(up, float)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w
