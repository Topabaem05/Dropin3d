"""Device selection and per-device helpers.

DROPIN3D_DEVICE forces a device (e.g. "cuda", "mps", "cpu").
"""

from __future__ import annotations

import os

import torch


def mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def get_device(device: str | None = None) -> torch.device:
    """Return ``device``, else DROPIN3D_DEVICE, else cuda > mps > cpu.

    ``device="auto"`` is the same as ``None``.
    """
    if device and device != "auto":
        return torch.device(device)
    forced = os.environ.get("DROPIN3D_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache() -> None:
    """Release cached allocator memory (torch.cuda/torch.mps.empty_cache)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif mps_available():
        torch.mps.empty_cache()


def manual_seed_all(seed: int) -> None:
    """Seed accelerator RNGs (call torch.manual_seed separately)."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if mps_available():
        torch.mps.manual_seed(seed)
