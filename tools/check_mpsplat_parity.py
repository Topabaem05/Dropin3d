"""Check mpsplat on a Mac: Metal kernels vs. the PyTorch reference path.

Run on Apple Silicon after installing mpsplat (patched or not):
    python tools/check_mpsplat_parity.py

mpsplat runs its Metal rasterizer only for 3 channels and backgrounds=None.
Passing an all-zero background gives the same math but forces mpsplat's
pure-PyTorch rasterizer, so the two calls below compare the two paths on the
same inputs. After applying third_party/mpsplat-patches, both paths follow
gsplat 1.5.3's compositing rules and should agree to ~1e-5.

The script also renders 16 channels with a background through
dropin3d.raster.rasterization (Metal, 3-channel chunks) and compares it with
mpsplat's PyTorch path.
"""

from __future__ import annotations

import argparse
import math

import torch


def make_scene(n: int, device: torch.device, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g) * 0.6 + torch.tensor([0.0, 0.0, 3.0])
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    scales = torch.rand(n, 3, generator=g) * 0.05 + 0.01
    opacities = torch.rand(n, generator=g) * 0.98 + 0.01
    W, H = 96, 64
    f = 80.0
    Ks = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])[None]
    viewmats = torch.eye(4)[None]
    tensors = [t.to(device) for t in (means, quats, scales, opacities, viewmats, Ks)]
    return tensors, W, H


def render(rasterization, scene, colors, backgrounds, render_mode="RGB"):
    (means, quats, scales, opacities, viewmats, Ks), W, H = scene
    params = [t.clone().requires_grad_(True) for t in (means, scales, opacities, colors)]
    out, alphas, info = rasterization(
        means=params[0], quats=quats, scales=params[1], opacities=params[2],
        colors=params[3], viewmats=viewmats, Ks=Ks, width=W, height=H,
        backgrounds=backgrounds, packed=False, render_mode=render_mode,
    )
    g = torch.Generator().manual_seed(1)
    w_out = torch.randn(out.shape, generator=g).to(out.device)
    w_a = torch.randn(alphas.shape, generator=g).to(out.device)
    ((out * w_out).sum() + (alphas * w_a).sum()).backward()
    return out.detach(), alphas.detach(), [p.grad.detach() for p in params]


def report(name, a, b):
    err = (a - b).abs().max().item()
    scale = max(b.abs().max().item(), 1e-12)
    print(f"  {name:<12} max|diff| = {err:.3e}   (max|ref| = {scale:.3e})")
    return err / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=1e-4, help="relative tolerance")
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available: run this on Apple Silicon.")
    dev = torch.device("mps")
    from gsplat.rendering import rasterization as gsplat_rasterization

    from dropin3d.raster import rasterization as dropin_rasterization

    import gsplat.cuda._torch_rasterize as torch_raster

    patched = hasattr(torch_raster, "MAX_ALPHA_3DGS")
    print(f"mpsplat gsplat-1.5.3 parity patch applied: {patched}")
    if not patched:
        print("  (both paths still use mpsplat's own rules; see third_party/mpsplat-patches)")

    scene = make_scene(args.n, dev)
    worst = 0.0

    print("[1] 3 channels: Metal path vs. mpsplat PyTorch path")
    colors = torch.rand(args.n, 3, generator=torch.Generator().manual_seed(2)).to(dev)
    metal = render(gsplat_rasterization, scene, colors, None)
    torch_path = render(gsplat_rasterization, scene, colors, torch.zeros(1, 3, device=dev))
    worst = max(worst, report("colors", metal[0], torch_path[0]))
    worst = max(worst, report("alphas", metal[1], torch_path[1]))
    for name, a, b in zip(("d_means", "d_scales", "d_opacity", "d_colors"), metal[2], torch_path[2]):
        worst = max(worst, report(name, a, b))

    print("[2] 16 channels + depth + background: dropin3d chunks vs. PyTorch path")
    colors16 = torch.rand(args.n, 16, generator=torch.Generator().manual_seed(3)).to(dev)
    bg = torch.rand(1, 16, generator=torch.Generator().manual_seed(4)).to(dev)
    chunked = render(dropin_rasterization, scene, colors16, bg, render_mode="RGB+ED")
    ref = render(gsplat_rasterization, scene, colors16, bg, render_mode="RGB+ED")
    worst = max(worst, report("colors", chunked[0], ref[0]))
    worst = max(worst, report("alphas", chunked[1], ref[1]))
    for name, a, b in zip(("d_means", "d_scales", "d_opacity", "d_colors"), chunked[2], ref[2]):
        worst = max(worst, report(name, a, b))

    ok = math.isfinite(worst) and worst <= args.tol
    print(f"worst relative diff = {worst:.3e} -> {'PASS' if ok else 'FAIL'} (tol {args.tol:g})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
