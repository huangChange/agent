"""Camera rigs. Internal convention: OpenCV (x right, y down, z forward), camera-to-world 4x4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.errors import InputError


@dataclass
class Camera:
    name: str
    K: np.ndarray  # (3, 3)
    c2w: np.ndarray  # (4, 4), OpenCV convention
    width: int
    height: int
    image_path: Path | None = None

    @property
    def w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w)

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (pixel uv (N, 2), depth (N,))."""
        cam = (self.w2c[:3, :3] @ pts.T + self.w2c[:3, 3:4]).T
        z = cam[:, 2]
        uv = (self.K @ cam.T).T
        return uv[:, :2] / np.clip(z[:, None], 1e-9, None), z


def load_transforms_json(path: str | Path, images_root: str | Path | None = None) -> list[Camera]:
    """nerfstudio / instant-ngp transforms.json (OpenGL c2w)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    root = Path(images_root) if images_root else path.parent
    frames = data.get("frames")
    if not frames:
        raise InputError(f"{path}: no frames")
    cams = []
    for i, fr in enumerate(frames):
        def get(k, fr=fr):  # per-frame intrinsics override global ones
            return fr.get(k, data.get(k))

        w, h = int(get("w")), int(get("h"))
        fx, fy = float(get("fl_x")), float(get("fl_y") or get("fl_x"))
        cx, cy = float(get("cx") or w / 2), float(get("cy") or h / 2)
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1.0  # OpenGL -> OpenCV
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        fp = fr.get("file_path")
        cams.append(Camera(name=Path(fp).stem if fp else f"frame_{i:05d}", K=K, c2w=c2w, width=w, height=h,
                           image_path=(root / fp) if fp else None))
    return cams


def orbit_cameras(center: np.ndarray, radius: float, n: int, elevation_deg: float = 25.0,
                  size: int = 512, fov_deg: float = 40.0, up: np.ndarray | None = None) -> list[Camera]:
    """Evenly spaced look-at cameras around an object; used for QA renders and agent previews."""
    up = np.array([0.0, 0.0, 1.0]) if up is None else up / np.linalg.norm(up)
    f = 0.5 * size / np.tan(np.radians(fov_deg) / 2)
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    el = np.radians(elevation_deg)
    cams = []
    for i in range(n):
        az = 2 * np.pi * i / n
        d = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up
        eye = center + radius * d
        fwd = center - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
        cams.append(Camera(name=f"orbit_{i:02d}", K=K, c2w=c2w, width=size, height=size))
    return cams
