"""Exact k-nearest-neighbour queries.

Two drop-in APIs:

* ``knn_points``: same contract as ``pytorch3d.ops.knn_points`` (subset).
  Uses pytorch3d for CUDA/CPU tensors when it is installed, else a pure
  PyTorch implementation (needed for MPS tensors, which pytorch3d does not
  support).
* ``knnquery``: same contract as ``pointops2.knnquery`` (Point Transformer,
  TAPIP3D). Uses the CUDA extension when it can be imported and the input is
  on CUDA, else a pure PyTorch implementation.

DROPIN3D_KNN_BACKEND = "native" | "torch" forces a backend for both.
"""

from __future__ import annotations

import importlib
import os
from typing import NamedTuple, Optional

import torch


def _forced_backend() -> str:
    return os.environ.get("DROPIN3D_KNN_BACKEND", "").lower()


# ---------------------------------------------------------------------------
# pytorch3d.ops.knn_points
# ---------------------------------------------------------------------------


class KNN(NamedTuple):
    dists: torch.Tensor  # squared distances, as in pytorch3d
    idx: torch.Tensor
    knn: Optional[torch.Tensor]


try:
    from pytorch3d.ops import knn_points as _p3d_knn_points

    HAS_PYTORCH3D = True
except Exception:
    _p3d_knn_points = None
    HAS_PYTORCH3D = False


def torch_knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    K: int,
    return_nn: bool = False,
    chunk_size: int = 4096,
) -> KNN:
    """Brute-force kNN with the ``pytorch3d.ops.knn_points`` contract.

    p1: [N, P1, D], p2: [N, P2, D]. Returns squared distances [N, P1, K]
    and int64 indices [N, P1, K], sorted ascending. Gradients flow to p1 and
    p2 through the distances (and ``knn`` when ``return_nn``), as in
    pytorch3d. Ragged batches (``lengths1/lengths2``) are not supported.
    """
    assert p1.dim() == 3 and p2.dim() == 3 and p1.shape[0] == p2.shape[0]
    N, P1, D = p1.shape
    if K > p2.shape[1]:
        raise ValueError(f"K={K} is larger than the number of points {p2.shape[1]}")

    # 1) Search without autograd. Squared differences are computed
    #    explicitly (not the matmul trick of torch.cdist) to keep full fp32
    #    precision in the ranking.
    with torch.no_grad():
        a_all, b = p1.detach(), p2.detach()
        idx = torch.cat(
            [
                torch.topk(
                    ((a_all[:, s : s + chunk_size, None, :] - b[:, None]) ** 2).sum(-1),
                    K,
                    dim=-1,
                    largest=False,
                    sorted=True,
                ).indices
                for s in range(0, P1, chunk_size)
            ],
            dim=1,
        )

    # 2) Differentiable distances for the selected neighbours.
    nn = torch.gather(p2, 1, idx.reshape(N, P1 * K, 1).expand(-1, -1, D)).reshape(
        N, P1, K, D
    )
    dists = ((p1[:, :, None, :] - nn) ** 2).sum(-1)
    return KNN(dists=dists, idx=idx, knn=nn if return_nn else None)


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1=None,
    lengths2=None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
) -> KNN:
    """Drop-in for ``pytorch3d.ops.knn_points`` (same signature)."""
    forced = _forced_backend()
    use_p3d = HAS_PYTORCH3D and p1.device.type in ("cuda", "cpu")
    if forced == "torch":
        use_p3d = False
    elif forced == "native":
        if not HAS_PYTORCH3D:
            raise ImportError("DROPIN3D_KNN_BACKEND=native but pytorch3d is missing")
        use_p3d = True

    if use_p3d:
        out = _p3d_knn_points(
            p1,
            p2,
            lengths1,
            lengths2,
            norm=norm,
            K=K,
            version=version,
            return_nn=return_nn,
            return_sorted=return_sorted,
        )
        return KNN(dists=out.dists, idx=out.idx, knn=out.knn)

    if lengths1 is not None or lengths2 is not None:
        raise NotImplementedError("torch kNN fallback does not support lengths1/lengths2")
    if norm != 2:
        raise NotImplementedError("torch kNN fallback supports norm=2 only")
    # Always sorted: a superset of return_sorted=False.
    return torch_knn_points(p1, p2, K=K, return_nn=return_nn)


# ---------------------------------------------------------------------------
# pointops2.knnquery
# ---------------------------------------------------------------------------

# Module paths where the pointops2 CUDA extension is usually importable.
# DROPIN3D_POINTOPS_MODULE overrides the list with one module path.
_POINTOPS_CANDIDATES = (
    "third_party.pointops2.functions.pointops",  # TAPIP3D / AnchorTAP3D
    "pointops2.pointops",  # `python setup.py install` of pointops2
    "pointops2.functions.pointops",
)
_pointops = None
_pointops_error: Optional[Exception] = None


def _load_pointops():
    global _pointops, _pointops_error
    if _pointops is not None or _pointops_error is not None:
        return _pointops
    names = [os.environ["DROPIN3D_POINTOPS_MODULE"]] if os.environ.get(
        "DROPIN3D_POINTOPS_MODULE"
    ) else list(_POINTOPS_CANDIDATES)
    for name in names:
        try:
            _pointops = importlib.import_module(name)
            return _pointops
        except Exception as e:  # ImportError, or missing CUDA build
            _pointops_error = e
    return None


# Same sentinel as the CUDA kernel (best_dist initialised to 1e10).
_EMPTY_DIST2 = 1e10


@torch.no_grad()
def torch_knnquery(
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: Optional[torch.Tensor],
    offset: torch.Tensor,
    new_offset: torch.Tensor,
    chunk_size: int = 2048,
):
    """Exact kNN with the ``pointops2.knnquery`` contract.

    xyz: (n, 3) context points, new_xyz: (m, 3) queries. ``offset`` and
    ``new_offset`` (b,) are cumulative end indices of each batch segment.
    Returns idx (m, nsample) int32 with *global* indices into ``xyz`` and
    dist (m, nsample) = sqrt(squared distance), sorted ascending. If a
    segment has fewer than ``nsample`` points, the missing slots get
    idx = segment start and dist = sqrt(1e10), as in the CUDA kernel.
    """
    if new_xyz is None:
        new_xyz = xyz
    m = new_xyz.shape[0]
    device = new_xyz.device
    idx_out = torch.empty((m, nsample), dtype=torch.int64, device=device)
    d2_out = torch.empty((m, nsample), dtype=torch.float32, device=device)

    ends, q_ends = offset.tolist(), new_offset.tolist()
    starts, q_starts = [0] + ends[:-1], [0] + q_ends[:-1]

    for s, e, qs, qe in zip(starts, ends, q_starts, q_ends):
        if qe <= qs:
            continue
        k = min(nsample, e - s)
        ctx = xyz[s:e].float()
        for c0 in range(qs, qe, chunk_size):
            c1 = min(c0 + chunk_size, qe)
            if k > 0:
                q = new_xyz[c0:c1].float()
                d2 = ((q[:, None, :] - ctx[None, :, :]) ** 2).sum(-1)
                vals, ids = torch.topk(d2, k, dim=-1, largest=False, sorted=True)
                d2_out[c0:c1, :k] = vals
                idx_out[c0:c1, :k] = ids + s
            if k < nsample:
                d2_out[c0:c1, k:] = _EMPTY_DIST2
                idx_out[c0:c1, k:] = s

    return idx_out.to(torch.int32), torch.sqrt(d2_out)


def knnquery(nsample, xyz, new_xyz, offset, new_offset):
    """Drop-in for ``pointops2.knnquery`` (same arguments and outputs)."""
    forced = _forced_backend()
    if forced == "torch":
        return torch_knnquery(nsample, xyz, new_xyz, offset, new_offset)
    native = _load_pointops() if (xyz.is_cuda or forced == "native") else None
    if forced == "native" and native is None:
        raise ImportError(f"pointops2 is not available: {_pointops_error}")
    if native is not None and xyz.is_cuda:
        return native.knnquery(nsample, xyz, new_xyz, offset, new_offset)
    return torch_knnquery(nsample, xyz, new_xyz, offset, new_offset)
