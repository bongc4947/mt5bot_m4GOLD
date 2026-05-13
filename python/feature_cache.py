"""
feature_cache.py — Float16 memory-mapped feature cache.

Solves the RAM bottleneck without chunked training:
  1. Build features once for ALL bars, saved as float16 memmap files.
  2. Training loads them with np.memmap(mode='r') — OS pages on demand.
  3. DataLoader shuffles freely across all N bars; peak RAM = one batch (~4 MB).

Storage per symbol (float16):
  dir_feat  : N × 1000 × 2B  (e.g. 2M bars = 4 GB)
  exec_feat : N × 1120 × 2B  (e.g. 2M bars = 4.5 GB)
  labels    : stored as int8 .npy — negligible size

Cache is rebuilt when bars or symbol config changes.
Identified by a small .meta JSON with shape + mtime.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Overlap added to each block so rolling-window features are accurate at boundaries.
# Must be >= max rolling window used in feature_engine (EMA50, rolling-100, etc.)
_BLOCK_SIZE    = 200_000   # bars processed at once during build
_BLOCK_OVERLAP = 300       # context rows discarded after feature build


def _cache_dir() -> Path:
    from config import PARQUET_DIR
    d = PARQUET_DIR.parent / "feature_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _paths(symbol: str) -> Tuple[Path, Path, Path, Path, Path]:
    d = _cache_dir()
    return (
        d / f"{symbol}_dir.bin",     # float16 memmap  (N, FEATURE_DIM_DIR)
        d / f"{symbol}_exec.bin",    # float16 memmap  (N, FEATURE_DIM_EXEC)
        d / f"{symbol}_dlabels.npy", # int8            (N,)
        d / f"{symbol}_elabels.npy", # int8            (N, 5)
        d / f"{symbol}_cache.json",  # shape + mtime metadata
    )


def cache_valid(symbol: str, bars_mtime: float) -> bool:
    """True if a fresh cache exists for this symbol and bar file mtime."""
    _, _, _, _, meta_p = _paths(symbol)
    if not meta_p.exists():
        return False
    try:
        m = json.loads(meta_p.read_text())
        return m.get("bars_mtime") == bars_mtime
    except Exception:
        return False


def _preallocate_file(path: Path, size_bytes: int) -> None:
    """
    Create a correctly-sized binary file before np.memmap(mode='r+').

    Two Windows pitfalls this solves:
    1. np.memmap(mode='w+') on files >1 GB raises [Errno 22] Invalid argument.
    2. Partially-written files left by a crashed/killed training run also raise
       [Errno 22] when re-opened for truncation — Windows holds a CreateFileMapping
       handle on them even after the process exits.  Deleting and recreating is
       the only reliable recovery.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            raise OSError(
                f"Cannot remove stale cache file '{path.name}'. "
                f"It may be open in MT5 or another process. "
                f"Close MT5, delete it manually, then retrain.\n  ({exc})"
            ) from exc
    with open(str(path), "wb") as f:
        f.seek(size_bytes - 1)
        f.write(b"\x00")


def build_cache(bars,
                symbol: str,
                bars_mtime: float,
                pip_size: float,
                h1_df=None, h4_df=None, h8_df=None, d1_df=None) -> None:
    """
    Build float16 memmap feature cache for all bars.

    Processes bars in overlapping blocks so rolling features are accurate
    at every position. Writes directly to mmap files — peak RAM = one block.
    """
    from feature_engine  import build_feature_dataframe
    from labeler         import compute_direction_labels
    from labeler_exec    import compute_exec_labels
    from config          import FEATURE_DIM_DIR, FEATURE_DIM_EXEC, FEATURE_WARMUP_BARS

    N = len(bars)
    dir_p, exec_p, dl_p, el_p, meta_p = _paths(symbol)

    log.info("[%s] Building feature cache: %d bars → %s", symbol, N, _cache_dir())

    # Purge small metadata/label files first (never locked).
    # Binary .bin files are handled by _preallocate_file below, which raises a
    # clear error if Windows has them locked (e.g. a previous training session
    # left an open memmap handle — just restart Python and retry).
    for _stale in (dl_p, el_p, meta_p):
        if _stale.exists():
            _stale.unlink()

    # Pre-allocate binary files before memmap.  _preallocate_file handles both
    # the >1 GB [Errno 22] Windows limitation and stale locked files.
    _preallocate_file(dir_p,  N * FEATURE_DIM_DIR  * 2)
    _preallocate_file(exec_p, N * FEATURE_DIM_EXEC * 2)

    dir_mm  = np.memmap(dir_p,  dtype="float16", mode="r+", shape=(N, FEATURE_DIM_DIR))
    exec_mm = np.memmap(exec_p, dtype="float16", mode="r+", shape=(N, FEATURE_DIM_EXEC))
    dir_labels_all  = np.zeros(N, dtype=np.int8)
    exec_labels_all = np.zeros((N, 5), dtype=np.int8)

    written = 0
    block = 0
    while written < N:
        b_start_raw = written
        b_end_raw   = min(N, written + _BLOCK_SIZE)

        # Add overlap at the start for rolling windows
        ctx_start = max(0, b_start_raw - _BLOCK_OVERLAP)
        chunk     = bars.iloc[ctx_start:b_end_raw].copy()
        trim      = b_start_raw - ctx_start   # rows to discard (overlap)

        dir_feat, exec_feat = build_feature_dataframe(
            chunk, symbol, pip_size=pip_size,
            h1_df=h1_df, h4_df=h4_df, h8_df=h8_df, d1_df=d1_df)

        dir_labels_chunk, _ = compute_direction_labels(chunk)
        exec_labels_chunk   = compute_exec_labels(chunk, dir_labels_chunk,
                                                  pip_size=pip_size, symbol=symbol)

        # First block: discard FEATURE_WARMUP_BARS rows (rolling-window warmup).
        # Subsequent blocks: discard overlap rows added for rolling-window context.
        # actual_start mirrors bar position so mmap index == bar index.
        # Rows 0..(warmup-1) in the mmap are left as zeros with zero labels —
        # DirectionDataset excludes them via exclude_flat=True (label=0).
        first_trim = FEATURE_WARMUP_BARS if b_start_raw == 0 else trim
        actual_start = b_start_raw if b_start_raw > 0 else first_trim

        dir_feat      = dir_feat[first_trim:]
        exec_feat     = exec_feat[first_trim:]
        dir_labels_chunk  = dir_labels_chunk[first_trim:]
        exec_labels_chunk = exec_labels_chunk[first_trim:]

        rows = len(dir_feat)
        end  = min(actual_start + rows, N)
        rows = end - actual_start

        dir_mm [actual_start:end] = dir_feat[:rows].astype("float16")
        exec_mm[actual_start:end] = exec_feat[:rows].astype("float16")
        dir_labels_all [actual_start:end] = np.clip(dir_labels_chunk[:rows],  -1, 1).astype(np.int8)
        exec_labels_all[actual_start:end] = np.clip(exec_labels_chunk[:rows], -128, 127).astype(np.int8)

        block += 1
        written = b_end_raw
        log.info("[%s] Cache block %d: rows %d–%d written", symbol, block, actual_start, end)

        del chunk, dir_feat, exec_feat, dir_labels_chunk, exec_labels_chunk
        gc.collect()

    # Flush mmap files to disk
    dir_mm.flush()
    exec_mm.flush()
    del dir_mm, exec_mm

    # Save labels (small — load fully)
    np.save(dl_p, dir_labels_all)
    np.save(el_p, exec_labels_all)

    # Save metadata
    meta_p.write_text(json.dumps({
        "symbol":     symbol,
        "n_bars":     N,
        "dir_shape":  [N, FEATURE_DIM_DIR],
        "exec_shape": [N, FEATURE_DIM_EXEC],
        "bars_mtime": bars_mtime,
    }))
    log.info("[%s] Feature cache complete: %d bars  dir=%s exec=%s",
             symbol, N,
             _fmt_size(dir_p.stat().st_size),
             _fmt_size(exec_p.stat().st_size))


def load_cache(symbol: str):
    """
    Load cached features as memory-mapped float16 arrays + full labels.

    Returns (dir_feat, exec_feat, dir_labels, exec_labels)
      dir_feat   : np.memmap  (N, FEATURE_DIM_DIR)   float16 — disk-backed
      exec_feat  : np.memmap  (N, FEATURE_DIM_EXEC)  float16 — disk-backed
      dir_labels : np.ndarray (N,)                   int8
      exec_labels: np.ndarray (N, 5)                 int8
    """
    from config import FEATURE_DIM_DIR, FEATURE_DIM_EXEC

    dir_p, exec_p, dl_p, el_p, meta_p = _paths(symbol)
    meta = json.loads(meta_p.read_text())
    N    = meta["n_bars"]

    dir_feat    = np.memmap(dir_p,  dtype="float16", mode="r", shape=(N, FEATURE_DIM_DIR))
    exec_feat   = np.memmap(exec_p, dtype="float16", mode="r", shape=(N, FEATURE_DIM_EXEC))
    dir_labels  = np.load(dl_p)
    exec_labels = np.load(el_p)

    log.info("[%s] Loaded feature cache: %d bars  dir=%s  exec=%s",
             symbol, N,
             _fmt_size(dir_p.stat().st_size),
             _fmt_size(exec_p.stat().st_size))
    return dir_feat, exec_feat, dir_labels, exec_labels


def _fmt_size(b: int) -> str:
    return f"{b/1e9:.2f}GB" if b >= 1e9 else f"{b/1e6:.0f}MB"
