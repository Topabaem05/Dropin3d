"""CuPy when it is usable, NumPy otherwise.

Write array code against the module returned by ``get_array_module`` (``xp``)
so the same function runs on CuPy (CUDA) and NumPy (Mac/CPU).
"""

from __future__ import annotations

import numpy as np
import torch

try:  # CuPy exists only on CUDA (or ROCm) machines.
    import cupy as _cp

    HAS_CUPY = bool(_cp.cuda.is_available())
except Exception:  # ImportError, or CUDA driver errors
    _cp = None
    HAS_CUPY = False


def to_array(x):
    """Convert a tensor/array to CuPy if CuPy is usable, else NumPy."""
    if isinstance(x, torch.Tensor):
        if HAS_CUPY:
            return _cp.asarray(x.detach())
        return x.detach().cpu().numpy()
    if HAS_CUPY:
        return _cp.asarray(x)
    return np.asarray(x)


def get_array_module(x):
    """Return ``cupy`` or ``numpy`` to match the type of ``x``."""
    if HAS_CUPY:
        return _cp.get_array_module(x)
    return np


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if HAS_CUPY and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return np.asarray(x)
