"""
_seeding.py — deterministic-seed helper.

Call set_global_seed(seed) once at the top of any training entrypoint to
make Python, NumPy, and PyTorch deterministic across CPU and GPU.
"""

from __future__ import annotations
import os
import random
import logging

import numpy as np

log = logging.getLogger(__name__)


def set_global_seed(seed: int = 42, *, deterministic_torch: bool = True) -> int:
    """
    Seed Python, NumPy, PyTorch (CPU + CUDA), and the cuBLAS workspace.

    Args:
        seed: integer seed; use the same value across runs to reproduce.
        deterministic_torch: also force torch deterministic algorithms. Slower
            but eliminates non-determinism from cuDNN convolution kernels and
            similar. Set False if a kernel raises and you don't have a
            deterministic alternative.

    Returns: the seed actually applied.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS deterministic workspace (required for some matmul ops on Ampere+)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass

    log.info("seeded: %d (deterministic_torch=%s)", seed, deterministic_torch)
    return seed
