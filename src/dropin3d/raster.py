"""Wrapper around ``gsplat.rasterization`` for Apple Silicon.

On Mac, install the mpsplat fork of gsplat (it keeps the ``gsplat`` package
name). Its Metal rasterizer runs only for exactly 3 channels and
``backgrounds=None``; any other call falls back to a slow pure-PyTorch
rasterizer. Many methods render more channels (features, masks, 3D tracks,
depth) with a background, so on MPS this wrapper:

1. pads the channels (colors + depth channel) to a multiple of 3 and renders
   with ``channel_chunk=3``, so every chunk uses the Metal kernel;
2. composites the background itself: ``C + (1 - alpha) * bg``.

gsplat 1.5.3 applies backgrounds as ``pix_out + T * bg`` with
``alpha = 1 - T`` and uses a zero background for the depth channel, so step 2
gives the same result. On CUDA the call is passed through unchanged.

DROPIN3D_CHUNKED_RASTER: "1" (default) = on for MPS tensors, "0" = off,
"force" = on for every device (tests).
"""

from __future__ import annotations

import os

import torch

_DEPTH_MODES = ("RGB+D", "RGB+ED")


def _gsplat_rasterization():
    from gsplat.rendering import rasterization

    return rasterization


def rasterization(*, colors, backgrounds=None, render_mode="RGB", _impl=None, **kwargs):
    """Same keyword arguments and outputs as ``gsplat.rasterization``.

    ``_impl`` replaces the gsplat function (tests only).
    """
    impl = _impl or _gsplat_rasterization()
    mode = os.environ.get("DROPIN3D_CHUNKED_RASTER", "1")
    use_chunks = mode == "force" or (mode != "0" and colors.device.type == "mps")
    if not use_chunks or render_mode in ("D", "ED"):
        return impl(colors=colors, backgrounds=backgrounds, render_mode=render_mode, **kwargs)
    if "channel_chunk" in kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k != "channel_chunk"}

    D = colors.shape[-1]
    n_depth = 1 if render_mode in _DEPTH_MODES else 0
    pad = (-(D + n_depth)) % 3
    if pad:
        colors = torch.cat([colors, colors.new_zeros(*colors.shape[:-1], pad)], dim=-1)
    render_colors, alphas, info = impl(
        colors=colors,
        backgrounds=None,
        render_mode=render_mode,
        channel_chunk=3,
        **kwargs,
    )
    out = render_colors[..., :D]
    if backgrounds is not None:
        out = out + backgrounds[..., None, None, :] * (1.0 - alphas)
    if n_depth:
        out = torch.cat([out, render_colors[..., -1:]], dim=-1)
    return out, alphas, info
