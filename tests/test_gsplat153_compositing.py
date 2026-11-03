"""Checks the logic of third_party/mpsplat-patches against gsplat 1.5.3.

The Metal shaders cannot run here, so this test checks the *algorithm*:

1. ``_cuda_fwd_pixel``: line-by-line transcription of the per-pixel loop in
   gsplat 1.5.3 ``RasterizeToPixels3DGSFwd.cu`` (continue / break rules).
2. ``_vectorized_fwd``: the vectorized code used in the patched
   ``gsplat/cuda/_torch_rasterize.py`` of mpsplat.
3. ``_metal_bwd_pixel``: transcription of the patched
   ``rasterize_to_pixels_3dgs_bwd.metal`` loop, compared with autograd
   through (2).
"""

import torch

ALPHA_THRESHOLD = 1.0 / 255.0
MAX_ALPHA = 0.999
T_THRESHOLD = 1e-4


def _sigma(means2d, conics, px, py):
    dx = means2d[:, 0] - px
    dy = means2d[:, 1] - py
    return 0.5 * (conics[:, 0] * dx * dx + conics[:, 2] * dy * dy) + conics[:, 1] * dx * dy


def _cuda_fwd_pixel(means2d, conics, colors, opac, px, py):
    """gsplat 1.5.3 forward for one pixel. Returns (color, alpha, last_id)."""
    T, cur_idx = 1.0, 0
    out = torch.zeros(colors.shape[1], dtype=torch.float64)
    sig = _sigma(means2d, conics, px, py)
    for i in range(means2d.shape[0]):
        sigma = sig[i].item()
        alpha = min(MAX_ALPHA, opac[i].item() * torch.exp(-sig[i]).item())
        if sigma < 0.0 or alpha < ALPHA_THRESHOLD:
            continue
        next_T = T * (1.0 - alpha)
        if next_T <= T_THRESHOLD:
            break
        out += colors[i].double() * (alpha * T)
        cur_idx = i
        T = next_T
    return out, 1.0 - T, cur_idx


def _vectorized_fwd(means2d, conics, colors, opac, px, py):
    """Same code as the patched mpsplat torch rasterizer (one pixel column)."""
    sigma = _sigma(means2d, conics, px, py)[:, None]  # [k, 1]
    alpha = (opac[:, None] * torch.exp(-sigma)).clamp(max=MAX_ALPHA)
    valid = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    one_minus = 1.0 - alpha
    trans_after = one_minus.cumprod(dim=0)
    trans = torch.cat([torch.ones_like(one_minus[:1]), trans_after[:-1]], dim=0)
    stop = valid & (trans_after <= T_THRESHOLD)
    done = stop.cumsum(dim=0) > 0
    weights = torch.where(done, torch.zeros_like(alpha), alpha * trans)  # [k, 1]
    color = (weights * colors).sum(0)
    return color, weights.sum()


def _metal_bwd_pixel(means2d, conics, colors, opac, px, py, alpha_final, bin_final, v_c, v_a):
    """Transcription of the patched Metal backward for one pixel."""
    K = means2d.shape[0]
    g_means = torch.zeros(K, 2, dtype=torch.float64)
    g_conics = torch.zeros(K, 3, dtype=torch.float64)
    g_colors = torch.zeros(K, colors.shape[1], dtype=torch.float64)
    g_opac = torch.zeros(K, dtype=torch.float64)
    T_final = 1.0 - alpha_final
    T = T_final
    buf = torch.zeros(colors.shape[1], dtype=torch.float64)
    for idx in range(bin_final, -1, -1):
        dx = (means2d[idx, 0] - px).item()
        dy = (means2d[idx, 1] - py).item()
        cn = conics[idx].tolist()
        sigma = 0.5 * (cn[0] * dx * dx + cn[2] * dy * dy) + cn[1] * dx * dy
        vis = float(torch.exp(torch.tensor(-sigma, dtype=torch.float64)))
        o = opac[idx].item()
        alpha = min(MAX_ALPHA, o * vis)
        if sigma < 0.0 or alpha < ALPHA_THRESHOLD:
            continue
        ra = 1.0 / (1.0 - alpha)
        T *= ra
        fac = alpha * T
        g_colors[idx] += fac * v_c
        c = colors[idx].double()
        v_alpha = float(((c * T - buf * ra) * v_c).sum()) + T_final * ra * v_a
        if o * vis <= MAX_ALPHA:
            v_sigma = -o * vis * v_alpha
            g_conics[idx] += torch.tensor([0.5 * v_sigma * dx * dx, v_sigma * dx * dy, 0.5 * v_sigma * dy * dy], dtype=torch.float64)
            g_means[idx] += torch.tensor([v_sigma * (cn[0] * dx + cn[1] * dy), v_sigma * (cn[1] * dx + cn[2] * dy)], dtype=torch.float64)
            g_opac[idx] += vis * v_alpha
        buf += c * fac
    return g_means, g_conics, g_colors, g_opac


def _scene(seed, K=40):
    g = torch.Generator().manual_seed(seed)
    means2d = torch.rand(K, 2, generator=g, dtype=torch.float64) * 8
    a = torch.rand(K, generator=g, dtype=torch.float64) * 0.8 + 0.05
    conics = torch.stack([a, torch.zeros(K, dtype=torch.float64), a], -1)
    conics[:3, 1] = 3.0  # indefinite conics -> sigma < 0 for some pixels
    opac = torch.rand(K, generator=g, dtype=torch.float64)
    opac[::5] = 0.9999  # saturating Gaussians -> alpha cap and early stop
    opac[1::7] = 0.002  # below 1/255 -> skipped
    colors = torch.rand(K, 3, generator=g, dtype=torch.float64)
    return means2d, conics, colors, opac


def test_vectorized_forward_matches_cuda_loop():
    n_early_stop = 0
    for seed in range(5):
        means2d, conics, colors, opac = _scene(seed)
        for px in (0.5, 2.5, 4.5, 6.5):
            for py in (1.5, 3.5, 5.5):
                ref_c, ref_a, last = _cuda_fwd_pixel(means2d, conics, colors, opac, px, py)
                c, a = _vectorized_fwd(means2d, conics, colors, opac, px, py)
                torch.testing.assert_close(c, ref_c, rtol=1e-10, atol=1e-12)
                torch.testing.assert_close(a, torch.tensor(ref_a, dtype=torch.float64), rtol=1e-10, atol=1e-12)
                n_early_stop += ref_a > 1 - 1e-3
    assert n_early_stop > 0  # the scenes exercise the early-stop branch


def test_patched_metal_backward_matches_autograd():
    for seed in range(3):
        means2d, conics, colors, opac = _scene(seed)
        for px, py in ((2.5, 3.5), (4.5, 1.5), (6.5, 5.5)):
            ref_c, ref_a, last = _cuda_fwd_pixel(means2d, conics, colors, opac, px, py)
            g = torch.Generator().manual_seed(100 + seed)
            v_c = torch.randn(3, generator=g, dtype=torch.float64)
            v_a = float(torch.randn(1, generator=g, dtype=torch.float64))

            params = [t.clone().requires_grad_(True) for t in (means2d, conics, colors, opac)]
            c, a = _vectorized_fwd(*params, px, py)
            ((c * v_c).sum() + a * v_a).backward()

            got = _metal_bwd_pixel(means2d, conics, colors, opac, px, py, ref_a, last, v_c, v_a)
            for auto, metal in zip(params, got):
                torch.testing.assert_close(metal, auto.grad, rtol=1e-8, atol=1e-10)
