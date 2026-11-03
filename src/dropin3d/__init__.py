"""dropin3d: drop-in replacements for CUDA-only libraries used in 3D vision.

Each module keeps the original library on NVIDIA GPUs and falls back to a
portable implementation (PyTorch / NumPy / scikit-learn) on Apple Silicon
(MPS) or CPU. See README.md for the list of covered APIs.
"""

__version__ = "0.1.0"

from dropin3d.device import empty_cache, get_device, manual_seed_all, mps_available

__all__ = ["__version__", "empty_cache", "get_device", "manual_seed_all", "mps_available"]
