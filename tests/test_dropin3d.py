"""CPU tests: each fallback is checked against a reference implementation.

Run: pytest   (no GPU, gsplat, cuML, CuPy, pytorch3d or pointops2 needed)
"""

import numpy as np
import pytest
import torch

from dropin3d import cluster, knn, precision, raster, ssim

# ---------------------------------------------------------------------------
# knn_points (pytorch3d)
# ---------------------------------------------------------------------------


def _brute_knn(p1, p2, K):
    d2 = ((p1[:, :, None, :] - p2[:, None, :, :]) ** 2).sum(-1)
    vals, idx = torch.sort(d2, dim=-1)
    return vals[..., :K], idx[..., :K]


@pytest.mark.parametrize("P1,P2,K", [(1000, 70, 4), (5000, 300, 8), (37, 37, 4)])
def test_torch_knn_points_matches_bruteforce(P1, P2, K):
    g = torch.Generator().manual_seed(0)
    p1 = torch.randn(2, P1, 3, generator=g)
    p2 = torch.randn(2, P2, 3, generator=g)
    out = knn.torch_knn_points(p1, p2, K=K, chunk_size=777)
    ref_d, ref_i = _brute_knn(p1, p2, K)
    torch.testing.assert_close(out.dists, ref_d)
    assert out.idx.dtype == torch.int64
    # Indices may differ only between points at exactly the same distance
    # (tie order is unspecified, as in pytorch3d): check each returned index
    # really has the reference distance.
    d_at_idx = ((p1[:, :, None, :] - torch.gather(
        p2[:, None].expand(-1, P1, -1, -1), 2, out.idx[..., None].expand(-1, -1, -1, 3)
    )) ** 2).sum(-1)
    torch.testing.assert_close(d_at_idx, ref_d)
    assert (out.idx != ref_i).float().mean() < 1e-3


def test_torch_knn_points_gradients():
    g = torch.Generator().manual_seed(1)
    p1 = torch.randn(1, 200, 3, generator=g, requires_grad=True)
    p2 = torch.randn(1, 50, 3, generator=g, requires_grad=True)
    out = knn.torch_knn_points(p1, p2, K=5)
    w = torch.randn_like(out.dists)
    (out.dists * w).sum().backward()

    p1r = p1.detach().clone().requires_grad_(True)
    p2r = p2.detach().clone().requires_grad_(True)
    d = ((p1r[0][:, None] - p2r[0][out.idx[0]]) ** 2).sum(-1)
    (d * w[0]).sum().backward()
    torch.testing.assert_close(p1.grad, p1r.grad)
    torch.testing.assert_close(p2.grad, p2r.grad)


def test_knn_points_signature_and_return_nn(monkeypatch):
    monkeypatch.setenv("DROPIN3D_KNN_BACKEND", "torch")
    p = torch.randn(1, 64, 3)
    dists, idx, nn = knn.knn_points(p, p, None, None, K=4, return_nn=True)
    assert torch.equal(idx[0, :, 0], torch.arange(64))  # self is nearest
    assert torch.all(dists[0, :, 0] == 0)
    torch.testing.assert_close(nn[0, :, 0], p[0])
    _, _, nn2 = knn.knn_points(p, p, K=2)
    assert nn2 is None


# ---------------------------------------------------------------------------
# knnquery (pointops2)
# ---------------------------------------------------------------------------


def _ref_knnquery(nsample, xyz, new_xyz, offset, new_offset):
    m = new_xyz.shape[0]
    idx = torch.zeros(m, nsample, dtype=torch.int64)
    d2 = torch.full((m, nsample), 1e10)
    ends, q_ends = offset.tolist(), new_offset.tolist()
    starts, q_starts = [0] + ends[:-1], [0] + q_ends[:-1]
    for s, e, qs, qe in zip(starts, ends, q_starts, q_ends):
        for q in range(qs, qe):
            dd = ((new_xyz[q][None] - xyz[s:e]) ** 2).sum(-1)
            order = torch.argsort(dd, stable=True)[:nsample]
            k = len(order)
            idx[q, :k] = order + s
            d2[q, :k] = dd[order]
            idx[q, k:] = s
    return idx.int(), torch.sqrt(d2)


def test_torch_knnquery_matches_kernel_semantics():
    g = torch.Generator().manual_seed(0)
    ctx_sizes = [50, 3, 0, 40]  # a segment smaller than k and an empty one
    q_sizes = [7, 5, 4, 9]
    xyz = torch.randn(sum(ctx_sizes), 3, generator=g)
    new_xyz = torch.randn(sum(q_sizes), 3, generator=g)
    offset = torch.tensor(np.cumsum(ctx_sizes), dtype=torch.int32)
    new_offset = torch.tensor(np.cumsum(q_sizes), dtype=torch.int32)
    idx, dist = knn.knnquery(8, xyz, new_xyz, offset, new_offset)  # CPU -> torch path
    ref_idx, ref_dist = _ref_knnquery(8, xyz, new_xyz, offset, new_offset)
    assert idx.dtype == torch.int32 and idx.shape == (sum(q_sizes), 8)
    assert torch.equal(idx, ref_idx)
    torch.testing.assert_close(dist, ref_dist)
    # Chunking does not change the result.
    idx2, dist2 = knn.torch_knnquery(8, xyz, new_xyz, offset, new_offset, chunk_size=2)
    assert torch.equal(idx2, idx) and torch.equal(dist2, dist)


# ---------------------------------------------------------------------------
# Clustering (cuML)
# ---------------------------------------------------------------------------


def test_kmeans_labels_sklearn(monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(cluster, "USE_CUML", False)
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    x = np.concatenate([c + 0.1 * rng.standard_normal((100, 3)) for c in centers])
    labels = cluster.kmeans_labels(torch.from_numpy(x), n_clusters=3)
    assert labels.dtype == np.int64 and labels.shape == (300,)
    for k in range(3):
        assert len(set(labels[k * 100 : (k + 1) * 100])) == 1
    assert len(set(labels)) == 3
    assert np.array_equal(labels, cluster.kmeans_labels(torch.from_numpy(x), 3))


def test_hdbscan_labels_sklearn(monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(cluster, "USE_CUML", False)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.1, (60, 3)), rng.normal(5, 0.1, (60, 3))])
    labels = cluster.hdbscan_labels(x.astype(np.float32), min_cluster_size=20, max_cluster_size=60)
    assert labels.max() + 1 == 2


# ---------------------------------------------------------------------------
# SSIM (fused-ssim)
# ---------------------------------------------------------------------------


def test_torch_ssim_matches_pytorch_msssim():
    pytorch_msssim = pytest.importorskip("pytorch_msssim")
    g = torch.Generator().manual_seed(0)
    a = torch.rand(2, 3, 64, 80, generator=g)
    b = (a + 0.1 * torch.randn(a.shape, generator=g)).clamp(0, 1)
    ours = ssim.torch_ssim(a, b, padding="valid")
    ref = pytorch_msssim.ssim(a, b, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-5)
    assert ssim.torch_ssim(a, a, padding="same").item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Rasterization wrapper (gsplat / mpsplat)
# ---------------------------------------------------------------------------


def _fake_rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height,
    backgrounds=None, render_mode="RGB", packed=False, channel_chunk=32,
):
    """Linear stand-in that follows gsplat 1.5.3's output rules:
    pixel = sum_i w_i c_i + T * bg, alpha = 1 - T, depth appended for
    RGB+ED with zero background and divided by alpha."""
    C, N = viewmats.shape[0], means.shape[0]
    g = torch.Generator().manual_seed(42)
    w = torch.rand(C, height, width, N, generator=g)
    w = 0.9 * w / w.sum(-1, keepdim=True) * torch.rand(C, height, width, 1, generator=g)
    alphas = w.sum(-1, keepdim=True)
    feats, bg = colors, backgrounds
    if render_mode in ("RGB+D", "RGB+ED"):
        feats = torch.cat([colors, means[:, 2:3]], dim=-1)
        if bg is not None:
            bg = torch.cat([bg, torch.zeros(C, 1)], dim=-1)
    out = torch.einsum("chwn,nd->chwd", w, feats)
    if bg is not None:
        out = out + (1 - alphas) * bg[:, None, None, :]
    if render_mode == "RGB+ED":
        out = torch.cat([out[..., :-1], out[..., -1:] / alphas.clamp(min=1e-10)], dim=-1)
    return out, alphas, {"channel_chunk": channel_chunk, "n_channels": feats.shape[-1], "bg": backgrounds}


@pytest.mark.parametrize(
    "D,mode", [(3, "RGB"), (4, "RGB"), (16, "RGB+ED"), (17, "RGB+ED"), (5, "RGB+D")]
)
def test_chunked_raster_matches_gsplat_background(monkeypatch, D, mode):
    N, C, H, W = 20, 2, 6, 5
    g = torch.Generator().manual_seed(0)
    kw = dict(
        means=torch.randn(N, 3, generator=g),
        quats=torch.randn(N, 4, generator=g),
        scales=torch.randn(N, 3, generator=g),
        opacities=torch.rand(N, generator=g),
        viewmats=torch.eye(4)[None].repeat(C, 1, 1),
        Ks=torch.eye(3)[None].repeat(C, 1, 1),
        width=W,
        height=H,
        packed=False,
        _impl=_fake_rasterization,
    )
    colors = torch.randn(N, D, generator=g)
    bg = torch.rand(C, D, generator=g)

    monkeypatch.setenv("DROPIN3D_CHUNKED_RASTER", "0")
    ref, ref_a, ref_info = raster.rasterization(colors=colors, backgrounds=bg, render_mode=mode, **kw)
    monkeypatch.setenv("DROPIN3D_CHUNKED_RASTER", "force")
    out, a, info = raster.rasterization(colors=colors, backgrounds=bg, render_mode=mode, **kw)

    assert ref_info["bg"] is not None and info["bg"] is None
    assert info["channel_chunk"] == 3 and info["n_channels"] % 3 == 0
    assert out.shape == ref.shape == (C, H, W, D + (mode != "RGB"))
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(a, ref_a)


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


def test_precision_helpers():
    assert precision.resolve_precision("cpu") == torch.float32
    assert precision.resolve_precision("mps") == torch.float32
    assert precision.resolve_precision("cpu", "fp16") == torch.float16
    with pytest.raises(ValueError):
        precision.resolve_precision("cpu", "int8")
    x = torch.randn(4, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert (x @ x).dtype == torch.bfloat16
        with precision.no_autocast():
            assert (x @ x).dtype == torch.float32
    with precision.autocast_context("cpu", torch.float32):
        assert (x @ x).dtype == torch.float32


def test_device_override(monkeypatch):
    from dropin3d.device import get_device

    monkeypatch.setenv("DROPIN3D_DEVICE", "cpu")
    assert get_device() == torch.device("cpu")
    assert get_device("meta") == torch.device("meta")
