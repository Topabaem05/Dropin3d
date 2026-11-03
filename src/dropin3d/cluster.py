"""cuML KMeans / HDBSCAN on CUDA, scikit-learn otherwise.

DROPIN3D_CLUSTER_BACKEND = "cuml" | "sklearn" forces a backend.
"""

from __future__ import annotations

import os

import numpy as np

from dropin3d.array import to_numpy


def _use_cuml() -> bool:
    choice = os.environ.get("DROPIN3D_CLUSTER_BACKEND", "").lower()
    if choice == "sklearn":
        return False
    try:
        import cuml  # noqa: F401
    except Exception:
        if choice == "cuml":
            raise
        return False
    return True


USE_CUML = _use_cuml()


def kmeans_labels(features, n_clusters: int) -> np.ndarray:
    """Fit KMeans and return labels as a NumPy int64 array.

    cuML path: ``cuml.KMeans(n_clusters)`` with cuML defaults.
    scikit-learn path: ``n_init=1, random_state=1`` to match cuML's defaults
    (one initialisation, fixed seed). sklearn's k-means++ is not cuML's
    scalable-k-means++, so labels are not identical to the CUDA run.
    """
    if USE_CUML:
        from cuml import KMeans

        model = KMeans(n_clusters=n_clusters)
        model.fit(features)
        return to_numpy(model.labels_).astype(np.int64)

    from sklearn.cluster import KMeans

    x = to_numpy(features).astype(np.float32, copy=False)
    model = KMeans(n_clusters=n_clusters, n_init=1, random_state=1)
    model.fit(x)
    return model.labels_.astype(np.int64)


def hdbscan_labels(
    features, min_cluster_size: int, max_cluster_size: int = 0
) -> np.ndarray:
    """Fit HDBSCAN and return labels (-1 = noise) as a NumPy int64 array."""
    if USE_CUML:
        from cuml import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size
        )
        model.fit(features)
        return to_numpy(model.labels_).astype(np.int64)

    from sklearn.cluster import HDBSCAN

    x = to_numpy(features).astype(np.float32, copy=False)
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size if max_cluster_size > 0 else None,
        copy=True,
    )
    model.fit(x)
    return model.labels_.astype(np.int64)
