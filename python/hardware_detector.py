"""
hardware_detector.py — Detect available hardware and configure training tier.
Sets batch size, max bars, mixed precision, and worker count.
"""

import os
import logging
import psutil

log = logging.getLogger(__name__)


class HardwareConfig:
    device: str = "cpu"
    amp: bool = False
    batch_size: int = 256
    max_bars: int = 100_000
    workers: int = 1
    tier: str = "minimum"
    vram_gb: float = 0.0
    ram_gb: float = 0.0

    def __repr__(self):
        return (f"HardwareConfig(tier={self.tier}, device={self.device}, "
                f"amp={self.amp}, batch={self.batch_size}, "
                f"max_bars={self.max_bars:,}, workers={self.workers}, "
                f"vram={self.vram_gb:.1f}GB, ram={self.ram_gb:.1f}GB)")


def detect() -> HardwareConfig:
    """
    Probe hardware and return a fully populated HardwareConfig.
    Priority: CUDA > MPS > ROCm > CPU
    """
    import importlib
    cfg = HardwareConfig()

    ram_total = psutil.virtual_memory().total / 1e9
    ram_avail = psutil.virtual_memory().available / 1e9
    cfg.ram_gb = ram_total

    # Peak RAM during training: parquet stores raw OHLCV (9 cols), but
    # feature_engine expands each bar to ~1000 float32 features (4 kB/bar).
    # Peak = feature matrix (1000×4B) + working numpy copy (1000×4B) = 8 kB/bar.
    # Use 40% of available RAM so model weights, batch tensors, and OS fit in the rest.
    bytes_per_bar = 1000 * 4 * 2  # float32 feature matrix + numpy working copy
    cfg.max_bars = min(5_000_000_000, int(ram_avail * 0.40 * 1e9 / bytes_per_bar))

    # Per-row activation overhead heuristic (bytes / sample / forward+backward).
    # Direction features dominate; modify model adds 8 dims, exec adds 120 — same
    # order of magnitude. AMP halves activation bytes.
    feat_dim = 1180   # FEATURE_DIM_DIR; imported lazily to avoid config cycles
    try:
        from config import FEATURE_DIM_DIR as _fd
        feat_dim = max(feat_dim, int(_fd))
    except Exception:
        pass

    try:
        torch = importlib.import_module("torch")

        if torch.cuda.is_available():
            cfg.device = "cuda"
            cfg.amp    = True
            props      = torch.cuda.get_device_properties(0)
            vram       = props.total_memory / 1e9
            cfg.vram_gb = vram

            # Use 85% of VRAM as the activation budget — leaves headroom for
            # cuDNN workspaces, AMP scaler, optimizer state, and a margin.
            # Per-row footprint: feat_dim * 4B for input + ~6x for activations,
            # gradients, and optimizer state. AMP fp16 halves activations only.
            per_row_bytes = feat_dim * 4 * (5 if cfg.amp else 10)
            budget_bytes  = vram * 0.85 * 1e9
            est_bs        = int(budget_bytes // per_row_bytes)

            # Round to a power-of-two-ish for cuBLAS efficiency, cap at 262144.
            # Floor at 4096 so tiny VRAM still trains.
            ladder = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
            cfg.batch_size = max(b for b in ladder if b <= est_bs) if est_bs >= ladder[0] else ladder[0]

            cfg.workers = min(8, os.cpu_count() or 2)

        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            cfg.device = "mps"
            cfg.amp    = False
            # MPS unified memory: same heuristic, on RAM
            per_row_bytes = feat_dim * 4 * 10
            est_bs = int(ram_total * 0.50 * 1e9 // per_row_bytes)
            cfg.batch_size = min(8192, max(512, est_bs))
            cfg.workers    = min(4, os.cpu_count() or 2)

        else:
            # CPU — check for ROCm via env
            rocm = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
            cfg.device = "cpu"
            cfg.amp    = False
            # CPU stays in small-batch territory: gradient locality and L3 cache
            # matter more than throughput, and per-batch wall time grows ~linearly.
            cfg.batch_size = min(1024, int(ram_avail * 64))
            cfg.workers    = max(1, (os.cpu_count() or 2) - 1)
            if rocm:
                log.info("ROCm detected but PyTorch CPU fallback active")

            # Warn loudly when NVIDIA GPU is present but torch can't see it.
            # Most common cause: Python 3.12 has CPU-only torch while Python 3.13
            # has CUDA torch — wrong interpreter on PATH.
            try:
                import subprocess as _sp
                _nv = _sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
                if _nv.returncode == 0 and _nv.stdout.strip():
                    _gpu_name = _nv.stdout.strip().splitlines()[0]
                    log.warning(
                        "NVIDIA GPU detected (%s) but torch.cuda.is_available()=False!\n"
                        "  You are likely running the WRONG Python (CPU-only torch).\n"
                        "  Current interpreter: %s\n"
                        "  Fix — run training from the CUDA Python:\n"
                        "    py -3.13 train.py apex  (or whichever subcommand)\n"
                        "  OR install CUDA torch in this Python:\n"
                        "    python -m pip install torch --index-url "
                        "https://download.pytorch.org/whl/cu126",
                        _gpu_name, os.path.abspath(__import__('sys').executable)
                    )
            except Exception:
                pass

    except ImportError:
        log.warning("PyTorch not importable — using CPU defaults")

    # Hardware tier classification
    # Bars actually usable are further capped by available RAM above.
    if cfg.vram_gb >= 16 or cfg.ram_gb >= 64:
        cfg.tier = "enterprise"
        cfg.max_bars = min(5_000_000_000, cfg.max_bars)
    elif cfg.vram_gb >= 4 or cfg.ram_gb >= 16:
        cfg.tier = "mid"
        cfg.max_bars = min(1_000_000_000, cfg.max_bars)
    elif cfg.ram_gb >= 8:
        cfg.tier = "low"
        cfg.max_bars = min(500_000_000, cfg.max_bars)
    else:
        cfg.tier = "minimum"
        cfg.max_bars = min(100_000_000, cfg.max_bars)
        cfg.batch_size = min(256, cfg.batch_size)
        cfg.workers = 1

    # Final: env override always wins. Power users can pin batch size for
    # reproducibility benchmarks or to claw back VRAM for other workloads.
    env_bs = os.environ.get("HYDRA_BATCH_SIZE")
    if env_bs:
        try:
            cfg.batch_size = int(env_bs)
            log.info("HYDRA_BATCH_SIZE=%s — overriding auto-detected batch size", env_bs)
        except ValueError:
            log.warning("HYDRA_BATCH_SIZE=%r is not an int; ignoring", env_bs)

    log.info("Hardware: %s", cfg)
    return cfg


_cached: HardwareConfig | None = None


def get() -> HardwareConfig:
    global _cached
    if _cached is None:
        _cached = detect()
    return _cached


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(detect())
