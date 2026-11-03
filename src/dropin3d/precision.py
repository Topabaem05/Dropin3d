"""Autocast dtype selection that matches the hardware.

* CUDA sm_80+ (A100, RTX 30xx+): bf16 (what most research code assumes).
* CUDA below sm_80 (V100, T4, ...): fp32. These GPUs have no bf16 units, and
  bf16 also makes PyTorch SDPA fall back to the O(N^2)-memory math kernel.
* MPS / CPU: fp32.

fp16 is an explicit opt-in only: its exponent range is smaller than bf16,
so validate it against a bf16 or fp32 run first.
"""

from __future__ import annotations

import contextlib

import torch

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_precision(device, precision: str = "auto") -> torch.dtype:
    """Map ``auto | bf16 | fp16 | fp32`` to a dtype for ``device``."""
    if precision not in ("auto", *_DTYPES):
        raise ValueError(f"unknown precision {precision!r}")
    if precision != "auto":
        return _DTYPES[precision]
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.get_device_capability(dev) >= (8, 0):
        return torch.bfloat16
    return torch.float32


def autocast_context(device, dtype: torch.dtype):
    """Autocast for ``dtype`` on ``device``; a no-op context for fp32."""
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(torch.device(device).type, dtype=dtype)


@contextlib.contextmanager
def no_autocast():
    """Disable autocast on every device type (cuda, mps, cpu).

    Replaces ``torch.autocast("cuda", enabled=False)``, which leaves MPS and
    CPU autocast active.
    """
    with contextlib.ExitStack() as stack:
        for dev_type in ("cuda", "mps", "cpu"):
            try:
                stack.enter_context(torch.autocast(dev_type, enabled=False))
            except (RuntimeError, ValueError):
                pass  # device type not supported by this torch build
        yield
