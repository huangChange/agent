"""Loaders for scene data. Everything format-specific lives here so it can be adapted
to in-house formats without touching the checking logic."""
from __future__ import annotations

import json
import os
import re

import numpy as np

from . import plyio

INSTANCE_FIELD_CANDIDATES = [
    "instance_id", "instance", "ins_id", "inst_id", "obj_id", "object_id",
    "segment_id", "seg_id", "instance_label", "label", "id",
]


# --------------------------------------------------------------------------- mesh
def load_mesh(path):
    """Return (V float64 [N,3], F int64 [M,3], info)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        ply = plyio.read_ply(path)
        v = ply["data"]["vertex"]
        V = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        face = ply["data"].get("face", {})
        key = next((k for k in ("vertex_indices", "vertex_index") if k in face), None)
        if key is None:
            raise ValueError("mesh PLY has no face element")
        F = _triangulate(face[key])
        info = {"format": "ply/" + ply["format"], "vertex_fields": list(v.keys())}
    elif ext == ".obj":
        V, F = _load_obj(path)
        info = {"format": "obj"}
    else:
        try:
            import trimesh  # optional
        except ImportError as e:
            raise ValueError(f"unsupported mesh format {ext}; install trimesh or convert to ply/obj") from e
        m = trimesh.load(path, force="mesh")
        V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
        info = {"format": ext.lstrip(".") + " (trimesh)"}
    info.update({"num_vertices": int(len(V)), "num_faces": int(len(F))})
    return V, F, info


def _triangulate(faces):
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        if faces.shape[1] == 3:
            return faces.astype(np.int64)
        polys = list(faces)
    else:
        polys = faces
    tris = []
    for p in polys:
        for i in range(1, len(p) - 1):
            tris.append((p[0], p[i], p[i + 1]))
    return np.asarray(tris, dtype=np.int64)


def _load_obj(path):
    V, polys = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                polys.append([i - 1 if i > 0 else len(V) + i for i in idx])
    return np.asarray(V, np.float64), _triangulate(polys)


# ---------------------------------------------------------------------- gaussians
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussians(path, instance_field=None, instance_npy=None):
    """Return dict with xyz, opacity (activated, 0..1), instance (int64 or None), info."""
    ply = plyio.read_ply(path)
    v = ply["data"]["vertex"]
    fields = list(v.keys())
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    info = {"ply_format": ply["format"], "num_gaussians": int(len(xyz)), "fields": fields}

    if "opacity" in v:
        o = v["opacity"].astype(np.float64)
        is_logit = bool(o.min() < 0 or o.max() > 1)
        info["opacity_storage"] = "logit (applied sigmoid)" if is_logit else "already 0..1"
        opacity = sigmoid(o) if is_logit else o
    else:
        info["opacity_storage"] = "missing (assumed 1.0)"
        opacity = np.ones(len(xyz))

    n_rest = sum(1 for k in fields if k.startswith("f_rest_"))
    if any(k.startswith("f_dc_") for k in fields):
        coeffs = n_rest // 3 + 1
        info["sh_degree"] = int(round(np.sqrt(coeffs))) - 1
    info["has_scale"] = all(f"scale_{i}" in v for i in range(3))
    info["has_rotation"] = all(f"rot_{i}" in v for i in range(4))

    instance = None
    if instance_npy:
        instance = np.load(instance_npy).astype(np.int64).reshape(-1)
        if len(instance) != len(xyz):
            raise ValueError(f"instance npy has {len(instance)} entries, gaussians have {len(xyz)}")
        info["instance_source"] = "npy"
    else:
        field = instance_field
        if field is None:
            field = next((c for c in INSTANCE_FIELD_CANDIDATES if c in v), None)
        if field is not None:
            if field not in v:
                raise ValueError(f"instance field '{field}' not found; available: {fields}")
            instance = np.asarray(v[field]).astype(np.int64)
            info["instance_source"] = f"ply field '{field}'"
    if instance is not None:
        info["num_instances"] = int(len(np.unique(instance)))
    else:
        info["instance_source"] = "none"
    return {"xyz": xyz, "opacity": opacity, "instance": instance, "info": info, "raw": v}


# ------------------------------------------------------------------------- labels
def load_labels(path):
    """Accepts {"17": "chair"} / {"17": {"label": "chair"}} / [{"id": 17, "label": "chair"}]."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out = {}
    if isinstance(obj, dict):
        for k, val in obj.items():
            out[int(k)] = val if isinstance(val, str) else str(val.get("label") or val.get("name") or val.get("class"))
    else:
        for item in obj:
            iid = item.get("id", item.get("instance_id"))
            out[int(iid)] = str(item.get("label") or item.get("name") or item.get("class"))
    return out


def normalize_label(s):
    s = str(s).strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[\s#]*\d+$", "", s)  # "chair_3" -> "chair"
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------------ cameras
def _gl_to_cv(c2w):
    c2w = np.array(c2w, dtype=np.float64)
    c2w[:3, 1:3] *= -1
    return c2w


def _quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_cameras(path, fmt="auto", convention="auto"):
    """Return list of {'c2w' (OpenCV 4x4), 'fx','fy','cx','cy','width','height'} (intrinsics may be None).

    fmt: transforms_json | colmap_txt | json_c2w | npy_c2w | auto
    convention: opencv | opengl | auto (transforms_json -> opengl, others -> opencv)
    """
    if fmt == "auto":
        if path.endswith(".npy"):
            fmt = "npy_c2w"
        elif path.endswith(".txt"):
            fmt = "colmap_txt"
        else:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            fmt = "transforms_json" if isinstance(obj, dict) and "frames" in obj else "json_c2w"
    cams = []
    if fmt == "transforms_json":
        conv = "opengl" if convention == "auto" else convention
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for fr in obj["frames"]:
            g = {**obj, **fr}
            c2w = np.array(fr["transform_matrix"], float)
            w, h = g.get("w"), g.get("h")
            fx = g.get("fl_x")
            if fx is None and "camera_angle_x" in g and w:
                fx = 0.5 * w / np.tan(0.5 * g["camera_angle_x"])
            cams.append({"c2w": _gl_to_cv(c2w) if conv == "opengl" else c2w,
                         "fx": fx, "fy": g.get("fl_y", fx), "cx": g.get("cx"), "cy": g.get("cy"),
                         "width": w, "height": h})
    elif fmt == "colmap_txt":
        intr = _read_colmap_cameras(os.path.join(os.path.dirname(path), "cameras.txt"))
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
        for ln in lines[::2]:
            t = ln.split()
            R = _quat_to_R(*map(float, t[1:5]))
            tv = np.array(list(map(float, t[5:8])))
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ tv
            cams.append({"c2w": c2w, **intr.get(int(t[8]), {})})
    elif fmt in ("json_c2w", "npy_c2w"):
        conv = "opencv" if convention == "auto" else convention
        if fmt == "npy_c2w":
            items = [{"c2w": m} for m in np.load(path)]
        else:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            items = obj if isinstance(obj, list) else obj.get("cameras", [])
        for it in items:
            c2w = np.array(it["c2w"], float)
            cams.append({"c2w": _gl_to_cv(c2w) if conv == "opengl" else c2w,
                         "fx": it.get("fx"), "fy": it.get("fy"), "cx": it.get("cx"), "cy": it.get("cy"),
                         "width": it.get("width"), "height": it.get("height")})
    else:
        raise ValueError(f"unknown camera format {fmt}")
    return cams


def _read_colmap_cameras(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            t = ln.split()
            cid, model, w, h = int(t[0]), t[1], int(t[2]), int(t[3])
            p = list(map(float, t[4:]))
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
                fx = fy = p[0]
                cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[:4]
            out[cid] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": w, "height": h}
    return out
