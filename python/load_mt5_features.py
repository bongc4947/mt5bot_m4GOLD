"""
load_mt5_features.py — read pre-computed feature vectors written by the
MT5-side exporter (ea/MT5_Bot_mk4_FeatureExport.mq5).

mk4.2.1: the canonical feature engine lives in MQL5 (FeatureEncoder.mqh).
Training reads what MT5 wrote — eliminating the parity drift between
two parallel feature implementations forever.

FILE FORMAT
-----------
For each (symbol, timeframe) the MT5 exporter writes two files into
$MT5_COMMON_DIR (or any directory you specify):

  HYDRA4_FEAT_{SYMBOL}_M5.bin       little-endian float32, row-major,
                                    shape (N, 1160). No header — pure floats.

  HYDRA4_FEAT_{SYMBOL}_M5.meta.json {
      "schema_version": "1.0",
      "symbol":         "EURUSD",
      "timeframe":      "M5",
      "n_rows":         2030129,
      "dim":            1160,
      "time_start":     "1971-01-03T23:00:00+00:00",
      "time_end":       "2026-04-21T01:25:00+00:00",
      "exported_by":    "HYDRA4_FeatureExport.mq5",
      "exported_at":    "2026-05-06T14:00:00+00:00",
      "ea_version":     "4.2.0",
      "ea_commit":      "<git short-sha at compile time, optional>",
      "feature_dim_dir":1160,
      "block_starts": {
          "M5":0, "H1":400, "H4":610, "H8":730, "D1":850,
          "Spectral":970, "Pattern":1030, "StatReg":1080, "XAsset":1140
      }
  }

  HYDRA4_FEAT_{SYMBOL}_M5_times.bin  optional: little-endian int64 array
                                     of bar-open epochs (UTC seconds),
                                     shape (N,). Lets the loader align
                                     the feature rows back to bars.

USAGE
-----
    from load_mt5_features import load_features
    features, meta = load_features("EURUSD", root="/path/to/Common/Files")
    print(features.shape, features.dtype)   # (N, 1160) float32

The trainer uses this when invoked as
    python train.py all --skip-extract --mt5-features
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MT5FeatureMeta:
    schema_version: str
    symbol:         str
    timeframe:      str
    n_rows:         int
    dim:            int
    time_start:     str
    time_end:       str
    exported_at:    str
    ea_version:     str
    block_starts:   dict[str, int]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _binfile(symbol: str, root: Path, timeframe: str = "M5") -> Path:
    return root / f"HYDRA4_FEAT_{symbol}_{timeframe}.bin"


def _metafile(symbol: str, root: Path, timeframe: str = "M5") -> Path:
    return root / f"HYDRA4_FEAT_{symbol}_{timeframe}.meta.json"


def _timesfile(symbol: str, root: Path, timeframe: str = "M5") -> Path:
    return root / f"HYDRA4_FEAT_{symbol}_{timeframe}_times.bin"


def default_root() -> Path:
    """Where MT5 writes feature files by default = MT5_COMMON_DIR."""
    from config import MT5_COMMON_DIR
    return MT5_COMMON_DIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_meta(symbol: str, root: Optional[Path] = None,
              timeframe: str = "M5") -> MT5FeatureMeta:
    """Read the JSON sidecar; raise if missing or schema mismatch."""
    root = root or default_root()
    p = _metafile(symbol, root, timeframe)
    if not p.exists():
        raise FileNotFoundError(
            f"MT5-exported metadata missing: {p}\n"
            f"Run the MQL5 exporter first: attach "
            f"ea/MT5_Bot_mk4_FeatureExport.mq5 to a {symbol} {timeframe} chart."
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    return MT5FeatureMeta(
        schema_version=str(raw.get("schema_version", "1.0")),
        symbol=str(raw["symbol"]),
        timeframe=str(raw.get("timeframe", timeframe)),
        n_rows=int(raw["n_rows"]),
        dim=int(raw["dim"]),
        time_start=str(raw.get("time_start", "")),
        time_end=str(raw.get("time_end", "")),
        exported_at=str(raw.get("exported_at", "")),
        ea_version=str(raw.get("ea_version", "")),
        block_starts=dict(raw.get("block_starts", {})),
    )


def load_features(
    symbol: str,
    *,
    root: Optional[Path] = None,
    timeframe: str = "M5",
    expected_dim: Optional[int] = None,
    mmap: bool = True,
) -> tuple[np.ndarray, MT5FeatureMeta]:
    """
    Load (features [N, dim], meta).

    Args:
        symbol       : e.g. "EURUSD"
        root         : directory containing the .bin/.meta.json. Defaults to
                       MT5_COMMON_DIR (where the EA exports).
        timeframe    : one of "M5","H1","H4","H8","D1". Default "M5".
        expected_dim : if set, raises if meta.dim mismatches.
        mmap         : if True, memory-map the binary (cheap; recommended for
                       2M+ bar files). If False, fully load into RAM.

    Returns: (features ndarray, meta dataclass).
    """
    root = root or default_root()
    meta = load_meta(symbol, root=root, timeframe=timeframe)

    if expected_dim is not None and meta.dim != expected_dim:
        raise ValueError(
            f"{symbol}: MT5-exported feature_dim={meta.dim} but caller "
            f"expected {expected_dim}. Did the EA recompile after a Defines.mqh "
            f"FEATURE_DIM bump?"
        )

    binp = _binfile(symbol, root, timeframe)
    if not binp.exists():
        raise FileNotFoundError(f"MT5-exported binary missing: {binp}")

    expected_bytes = meta.n_rows * meta.dim * 4   # float32
    actual_bytes = binp.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{symbol}: binary size {actual_bytes} != expected "
            f"{expected_bytes} (n_rows={meta.n_rows}, dim={meta.dim}, float32). "
            f"Truncated export, or meta drift."
        )

    if mmap:
        arr = np.memmap(binp, dtype=np.float32, mode="r",
                        shape=(meta.n_rows, meta.dim))
    else:
        arr = np.fromfile(binp, dtype=np.float32).reshape(meta.n_rows, meta.dim)

    log.info("[%s] MT5 features loaded: shape=%s  ea_version=%s  exported=%s",
             symbol, arr.shape, meta.ea_version, meta.exported_at)
    return arr, meta


def load_times(symbol: str, *, root: Optional[Path] = None,
               timeframe: str = "M5") -> Optional[np.ndarray]:
    """
    Optional: load the per-row UTC-epoch timestamps written alongside features.
    Returns int64 array of shape (N,), or None if the file isn't present.
    """
    root = root or default_root()
    p = _timesfile(symbol, root, timeframe)
    if not p.exists():
        return None
    arr = np.fromfile(p, dtype=np.int64)
    return arr


def list_available(root: Optional[Path] = None) -> list[str]:
    """Symbols that have a complete (bin + meta) export under root."""
    root = root or default_root()
    out = []
    for p in sorted(root.glob("HYDRA4_FEAT_*_M5.meta.json")):
        sym = p.stem.removesuffix(".meta")[len("HYDRA4_FEAT_"):-len("_M5")]
        if _binfile(sym, root).exists():
            out.append(sym)
    return out


# ---------------------------------------------------------------------------
# CLI: list / inspect
# ---------------------------------------------------------------------------

def _cli() -> int:
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None,
                   help="Directory containing .bin/.meta.json (default: MT5_COMMON_DIR).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="List symbols with a complete MT5 export.")
    sp_in = sub.add_parser("info", help="Print metadata + sanity stats for one symbol.")
    sp_in.add_argument("symbol")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.root or default_root()

    if args.cmd == "list":
        syms = list_available(root)
        if not syms:
            print(f"No HYDRA4_FEAT_*_M5.bin found under {root}")
            return 1
        for s in syms:
            try:
                m = load_meta(s, root=root)
                print(f"  {s:<14} N={m.n_rows:>10,}  dim={m.dim}  "
                      f"{m.time_start[:10]} -> {m.time_end[:10]}  "
                      f"v={m.ea_version}")
            except Exception as e:
                print(f"  {s:<14} BROKEN: {e}")
        return 0

    if args.cmd == "info":
        feats, meta = load_features(args.symbol, root=root, mmap=True)
        print(f"symbol      : {meta.symbol}")
        print(f"shape       : {feats.shape}  ({feats.dtype})")
        print(f"ea_version  : {meta.ea_version}")
        print(f"exported_at : {meta.exported_at}")
        print(f"time range  : {meta.time_start}  ->  {meta.time_end}")
        print(f"block_starts: {meta.block_starts}")
        # Cheap stats (skim only)
        sample = np.asarray(feats[::max(1, len(feats) // 10000)])
        print(f"sample mean : {sample.mean():.4f}")
        print(f"sample std  : {sample.std():.4f}")
        print(f"NaN cells   : {int(np.isnan(sample).sum())}")
        print(f"Inf cells   : {int(np.isinf(sample).sum())}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
