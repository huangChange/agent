"""GPU gaussian rasterization via gsplat (CUDA). Imported lazily by GPU tools only."""

from __future__ import annotations

import numpy as np

from ..core.errors import DependencyMissingError
from ..io.cameras import Camera
from ..io.gaussians import Gaussians
from .raster import RenderOutput

_device_cache: dict[str, object] = {}


def render_gaussians(g: Gaussians, cam: Camera, with_depth: bool = True, sh_degree: int | None = None) -> RenderOutput:
    try:
        import torch
        from gsplat import rasterization
    except ImportError as e:
        raise DependencyMissingError("gsplat", "gpu") from e
    if not torch.cuda.is_available():
        raise DependencyMissingError("CUDA device", "gpu")
    dev = torch.device("cuda")
    key = id(g)
    if _device_cache.get("key") != key:  # upload once per gaussian set, reuse across views
        q = g.quats_wxyz / np.linalg.norm(g.quats_wxyz, axis=1, keepdims=True)
        rest = g.sh_rest.reshape(len(g), 3, -1).transpose(0, 2, 1) if g.sh_rest.shape[1] else None
        sh = g.sh_dc[:, None, :] if rest is None else np.concatenate([g.sh_dc[:, None, :], rest], axis=1)
        _device_cache.update(
            key=key,
            means=torch.from_numpy(g.means).to(dev),
            quats=torch.from_numpy(q.astype(np.float32)).to(dev),
            scales=torch.from_numpy(g.scales.astype(np.float32)).to(dev),
            opacities=torch.from_numpy(g.opacity.astype(np.float32)).to(dev),
            sh=torch.from_numpy(sh.astype(np.float32)).to(dev),
        )
    c = _device_cache
    n_coeffs = c["sh"].shape[1]
    degree = int(round(np.sqrt(n_coeffs))) - 1 if sh_degree is None else sh_degree
    viewmat = torch.from_numpy(cam.w2c.astype(np.float32)).to(dev)[None]
    K = torch.from_numpy(cam.K.astype(np.float32)).to(dev)[None]
    with torch.no_grad():
        colors, alphas, _ = rasterization(
            c["means"], c["quats"], c["scales"], c["opacities"], c["sh"], viewmat, K,
            cam.width, cam.height, sh_degree=degree, render_mode="RGB+ED" if with_depth else "RGB",
        )
    out = colors[0].cpu().numpy()
    alpha = alphas[0, ..., 0].cpu().numpy()
    rgb = (np.clip(out[..., :3], 0, 1) * 255).astype(np.uint8)
    depth = out[..., 3] if with_depth else np.zeros(alpha.shape, np.float32)
    return RenderOutput(rgb=rgb, alpha=alpha, depth=depth)
