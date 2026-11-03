"""SSIM with the ``fused_ssim`` contract.

Uses fused-ssim (CUDA, MPS, XPU kernels) when its compiled kernel loaded,
else a pure PyTorch implementation. DROPIN3D_SSIM_BACKEND = "torch" forces
the PyTorch path.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def torch_ssim(
    img1: torch.Tensor, img2: torch.Tensor, padding: str = "same", train: bool = True
) -> torch.Tensor:
    """Mean SSIM, same definition as ``fused_ssim``.

    img1, img2: [B, C, H, W] in [0, 1]. 11x11 Gaussian window (sigma 1.5),
    zero padding, C1 = 0.01^2, C2 = 0.03^2, variances clamped at 0.
    padding="valid" drops a 5-pixel border of the SSIM map before the mean.
    ``train`` is accepted for API compatibility and ignored.
    """
    if padding not in ("same", "valid"):
        raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")
    C1, C2 = 0.01**2, 0.03**2
    C = img1.shape[1]
    g = _gaussian_window().to(device=img1.device, dtype=img1.dtype)
    w_h = g.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    w_v = g.view(1, 1, -1, 1).expand(C, 1, -1, 1)

    def blur(x):
        x = F.conv2d(x, w_h, padding=(0, 5), groups=C)
        return F.conv2d(x, w_v, padding=(5, 0), groups=C)

    mu1, mu2 = blur(img1), blur(img2)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = (blur(img1 * img1) - mu1_sq).clamp_min(0.0)
    sigma2_sq = (blur(img2 * img2) - mu2_sq).clamp_min(0.0)
    sigma12 = blur(img1 * img2) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if padding == "valid":
        ssim_map = ssim_map[:, :, 5:-5, 5:-5]
    return ssim_map.mean()


def _load_fused_ssim():
    if os.environ.get("DROPIN3D_SSIM_BACKEND", "").lower() == "torch":
        return torch_ssim, "torch"
    try:
        import fused_ssim as _fs

        # fused_ssim imports its kernel only when CUDA/MPS/XPU is available;
        # without one, ``fusedssim`` is undefined and calls would fail.
        if hasattr(_fs, "fusedssim"):
            return _fs.fused_ssim, "fused_ssim"
    except Exception:
        pass
    return torch_ssim, "torch"


fused_ssim, SSIM_BACKEND = _load_fused_ssim()
