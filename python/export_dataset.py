"""
export_dataset.py - single entrypoint for packaging HYDRA mk4 training data
into a self-describing bundle ready for upload to Kaggle, Hugging Face Datasets,
Zenodo, OSF, or any open data bank.

What it bundles
---------------
  bars/{SYMBOL}_M5.parquet (+optional .csv) - M5 OHLCV bars per symbol
  labels/{SYMBOL}_M5_dir20.parquet          - direction label + forward return
                                              (-1/0/+1 per bar; 20-bar horizon)
  calendar/economic_calendar.parquet        - scheduled macro events
  macro/cot_sentiment.parquet               - CFTC COT-derived sentiment
  macro/fred_macro.parquet                  - FRED macro indicators
  schemas/*.schema.json                     - per-table column schemas
  manifest.json                             - integrity + provenance
  dataset-metadata.json                     - Kaggle CLI metadata
  dataset_card.md / README.md               - HF dataset card with YAML front-
                                              matter (also serves as Kaggle README)
  LICENSE                                   - chosen license text

USAGE
-----
    python export_dataset.py                       # all data, default paths
    python export_dataset.py --out my_dataset/     # explicit output dir
    python export_dataset.py --symbols EURUSD GOLD # subset
    python export_dataset.py --no-csv --no-labels  # parquet only, no labels
    python export_dataset.py --zip                 # produce a .zip bundle
    python export_dataset.py --license CC-BY-4.0 \
                             --kaggle-id "yourname/hydra-mk4-bars"

Upload paths after running
--------------------------
    Kaggle (CLI installed):
        kaggle datasets create -p dataset_export/
    Hugging Face (huggingface_hub installed):
        huggingface-cli upload <user>/<dataset> dataset_export/ . \
            --repo-type=dataset
    Zenodo / OSF: zip the dataset_export/ folder and upload via web UI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BASE_DIR, DATA_DIR, PARQUET_DIR,
    ALL_SYMBOLS, LABEL_FORWARD_BARS,
)

log = logging.getLogger(__name__)

DEFAULT_OUT  = BASE_DIR / "dataset_export"
SCHEMA_DIR   = "schemas"

KNOWN_LICENSES = {
    "CC-BY-4.0":     ("Creative Commons Attribution 4.0", "https://creativecommons.org/licenses/by/4.0/"),
    "CC-BY-NC-4.0":  ("Creative Commons Attribution-NonCommercial 4.0", "https://creativecommons.org/licenses/by-nc/4.0/"),
    "CC-BY-SA-4.0":  ("Creative Commons Attribution-ShareAlike 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
    "MIT":           ("MIT License", "https://opensource.org/licenses/MIT"),
    "Apache-2.0":    ("Apache License 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
    "ODbL-1.0":      ("Open Data Commons Open Database License 1.0", "https://opendatacommons.org/licenses/odbl/1-0/"),
    "CDLA-Permissive-2.0": ("CDLA-Permissive-2.0 (Linux Foundation)", "https://cdla.dev/permissive-2-0/"),
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256(p: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _pick_latest_per_symbol() -> dict[str, Path]:
    """Largest parquet per symbol under PARQUET_DIR (most-bars wins)."""
    by_sym: dict[str, Path] = {}
    for p in PARQUET_DIR.glob("HYDRA4_FEAT_*_*bars.parquet"):
        # filename: HYDRA4_FEAT_{SYMBOL}_{N}bars.parquet
        stem = p.stem
        try:
            _, _, *parts = stem.split("_")
            symbol = "_".join(parts[:-1])           # supports BRENT_OIL etc.
            n_bars = int(parts[-1].rstrip("bars"))
        except Exception:
            continue
        cur = by_sym.get(symbol)
        if cur is None or p.stat().st_size > cur.stat().st_size:
            by_sym[symbol] = p
        # n_bars unused but parsed to validate filename pattern
        _ = n_bars
    return by_sym


def _schema_from_df(df: pd.DataFrame, *, name: str, description: str) -> dict:
    return {
        "name":        name,
        "description": description,
        "n_rows":      int(len(df)),
        "columns": [
            {
                "name":  str(c),
                "dtype": str(df[c].dtype),
                "nullable": bool(df[c].isna().any()),
            }
            for c in df.columns
        ],
    }


def _write_schema(out: Path, schema: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def _record(manifest: list[dict], path: Path, root: Path,
            schema_ref: Optional[str], rows: Optional[int]) -> None:
    manifest.append({
        "path":       path.relative_to(root).as_posix(),
        "bytes":      path.stat().st_size,
        "rows":       rows,
        "sha256":     _sha256(path),
        "schema_ref": schema_ref,
    })


# ---------------------------------------------------------------------------
# Per-block exporters
# ---------------------------------------------------------------------------

def _export_bars(symbols: list[str], *, out_root: Path, write_csv: bool,
                 manifest: list[dict]) -> dict[str, pd.DataFrame]:
    bars_dir = out_root / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)

    src_map = _pick_latest_per_symbol()
    if not src_map:
        log.warning("no parquet bars found under %s", PARQUET_DIR)
        return {}

    bars_kept: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        src = src_map.get(sym)
        if src is None:
            log.warning("skip %s: no cached parquet", sym)
            continue
        df = pd.read_parquet(src)
        keep = [c for c in ("time","open","high","low","close",
                            "tick_volume","spread","real_volume","canonical")
                if c in df.columns]
        df = df[keep].copy()
        if "canonical" not in df.columns:
            df["canonical"] = sym

        out_pq = bars_dir / f"{sym}_M5.parquet"
        df.to_parquet(out_pq, index=False)
        log.info("bars: %s  rows=%d  -> %s", sym, len(df), out_pq.name)

        if write_csv:
            out_csv = bars_dir / f"{sym}_M5.csv"
            df.to_csv(out_csv, index=False)
            _record(manifest, out_csv, out_root, "schemas/bars.schema.json", len(df))

        _record(manifest, out_pq, out_root, "schemas/bars.schema.json", len(df))
        bars_kept[sym] = df

    if bars_kept:
        any_df = next(iter(bars_kept.values()))
        _write_schema(
            out_root / SCHEMA_DIR / "bars.schema.json",
            _schema_from_df(
                any_df,
                name="HYDRA mk4 M5 OHLCV bars",
                description=("5-minute bars for one symbol. Time is UTC. "
                             "tick_volume is the broker tick count; real_volume "
                             "may be 0 for non-exchange instruments."),
            ),
        )
    return bars_kept


def _export_labels(bars_kept: dict[str, pd.DataFrame], *, out_root: Path,
                   manifest: list[dict]) -> None:
    if not bars_kept:
        return
    from labeler import compute_direction_labels

    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    H = LABEL_FORWARD_BARS
    sample_df = None
    for sym, df in bars_kept.items():
        if "close" not in df.columns:
            log.warning("labels: %s missing 'close' column; skipping", sym)
            continue
        labels, regimes = compute_direction_labels(df)
        closes = df["close"].to_numpy()
        fwd = pd.Series(0.0, index=df.index, dtype="float64")
        if len(closes) > H:
            fwd.iloc[:-H] = (closes[H:] - closes[:-H]) / closes[:-H]
        out_df = pd.DataFrame({
            "time":               df["time"].values if "time" in df.columns else range(len(df)),
            "dir_label_20bar":    labels.astype("int8"),
            "regime_3state":      regimes.astype("int8"),
            "forward_return_20bar": fwd.values.astype("float32"),
        })
        out_pq = labels_dir / f"{sym}_M5_dir20.parquet"
        out_df.to_parquet(out_pq, index=False)
        _record(manifest, out_pq, out_root, "schemas/labels.schema.json", len(out_df))
        sample_df = out_df
        log.info("labels: %s  rows=%d  +1=%d  -1=%d  0=%d",
                 sym, len(out_df),
                 int((labels == 1).sum()), int((labels == -1).sum()),
                 int((labels == 0).sum()))
    if sample_df is not None:
        _write_schema(
            out_root / SCHEMA_DIR / "labels.schema.json",
            _schema_from_df(
                sample_df,
                name="Direction labels (20-bar horizon)",
                description=("Per-bar direction label and 20-bar forward return. "
                             "dir_label_20bar in {-1, 0, +1}; regime_3state in "
                             "{0=BULL, 1=SIDEWAYS, 2=BEAR}; forward_return_20bar is "
                             "(close[t+20] - close[t]) / close[t]. "
                             "Last 20 rows have forward_return_20bar = 0."),
            ),
        )


# mk4.2: calendar/macro export blocks removed — model no longer consumes
# those features. The dataset is now bars + labels only.


# ---------------------------------------------------------------------------
# Top-level metadata
# ---------------------------------------------------------------------------

def _write_kaggle_metadata(out_root: Path, *, kaggle_id: str, license_id: str,
                           title: str, subtitle: str) -> None:
    md = {
        "title":     title,
        "id":        kaggle_id,
        "subtitle":  subtitle,
        "licenses":  [{"name": license_id}],
        "resources": [],
    }
    (out_root / "dataset-metadata.json").write_text(json.dumps(md, indent=2),
                                                    encoding="utf-8")


def _write_license(out_root: Path, license_id: str) -> None:
    name, url = KNOWN_LICENSES.get(license_id, (license_id, ""))
    (out_root / "LICENSE").write_text(
        f"{name}\n{url}\n\n"
        "This dataset is derived from market data accessed via MetaTrader 5\n"
        "broker terminals and from public macro sources (FRED, CFTC). The\n"
        "underlying market data may be subject to broker-specific redistribution\n"
        "terms; verify with your broker before commercial reuse.\n",
        encoding="utf-8",
    )


def _write_card(out_root: Path, *, license_id: str, title: str, subtitle: str,
                kaggle_id: str, manifest: list[dict], symbols: list[str]) -> None:
    n_files = len(manifest)
    total_bytes = sum(m["bytes"] for m in manifest)
    total_rows  = sum(m["rows"] or 0 for m in manifest)
    yaml = (
        "---\n"
        f"license: {license_id.lower()}\n"
        "task_categories:\n  - time-series-forecasting\n  - tabular-classification\n"
        "tags:\n  - finance\n  - trading\n  - mt5\n  - ohlcv\n  - intraday\n"
        "size_categories:\n  - 10M<n<100M\n"
        "---\n"
    )
    body = f"""# {title}

> {subtitle}

Generated by `export_dataset.py` from the
[HYDRA mk4](https://github.com/bongc4947/mtbotmk1) trading-bot project.

## Contents

| Folder         | Files | Description |
|----------------|-------|-------------|
| `bars/`        | one `.parquet` per symbol | M5 OHLCV bars (UTC) |
| `labels/`      | one `.parquet` per symbol | Direction labels + 20-bar forward returns |
| `calendar/`    | `economic_calendar.parquet` | Scheduled macro releases |
| `macro/`       | `cot_sentiment.parquet`, `fred_macro.parquet` | Weekly COT + daily FRED |
| `schemas/`     | per-file `*.schema.json` | Column type + description |
| `manifest.json` | 1 file | sha256 / row count / byte size for every artifact |

**Symbols included:** {', '.join(symbols) if symbols else 'see manifest'}

**Totals:** {n_files} files, {total_rows:,} rows, {total_bytes/1e6:.1f} MB.

## Schema highlights

### `bars/{{SYMBOL}}_M5.parquet`
| column         | type     | notes |
|----------------|----------|-------|
| `time`         | datetime64[ns, UTC] | bar open time |
| `open/high/low/close` | float64 | OHLC |
| `tick_volume`  | uint64 | broker tick count |
| `spread`       | int32 | spread in points |
| `real_volume`  | uint64 | exchange volume (0 for OTC) |
| `canonical`    | string | symbol canonical name |

### `labels/{{SYMBOL}}_M5_dir20.parquet`
| column                  | type    | notes |
|-------------------------|---------|-------|
| `time`                  | datetime64[ns, UTC] | aligned with bars |
| `dir_label_20bar`       | int8 | -1=SHORT, 0=FLAT, +1=LONG (20-bar horizon, ATR + Sharpe filtered) |
| `regime_3state`         | int8 | 0=BULL, 1=SIDEWAYS, 2=BEAR |
| `forward_return_20bar`  | float32 | (close[t+20] - close[t]) / close[t] |

The last 20 rows of each symbol's labels are zero-padded (no future data
available).

## Time alignment / leakage
- `bars` and `labels` share `time` (UTC). Join with a simple left-merge.
- `calendar`, `macro/cot_sentiment`, and `macro/fred_macro` are coarser-grained
  (event / weekly / daily). Use `pd.merge_asof(direction='backward')` to align
  with bars without leaking future events.
- Labels look forward by exactly `LABEL_FORWARD_BARS = 20` bars (~100 minutes
  on M5). When training, use a chronological train/val split with a gap of
  at least 20 bars between the end of the train period and the start of val.

## How to load (Python)

```python
import pandas as pd
bars   = pd.read_parquet('bars/EURUSD_M5.parquet')
labels = pd.read_parquet('labels/EURUSD_M5_dir20.parquet')
df = bars.merge(labels, on='time', how='left')

cal    = pd.read_parquet('calendar/economic_calendar.parquet')
df_cal = pd.merge_asof(df.sort_values('time'),
                       cal.sort_values('datetime'),
                       left_on='time', right_on='datetime',
                       direction='backward')
```

## Provenance and license

- Bars: extracted from MetaTrader 5 broker terminal via the official Python API.
- Calendar: MetaTrader 5 economic calendar.
- COT: CFTC Commitments of Traders public reports.
- FRED: Federal Reserve Economic Data (St. Louis Fed).
- License: **{license_id}** (see `LICENSE`).

The underlying market data may carry broker-specific redistribution terms.
Verify before commercial reuse.

## Reproducibility

- Bars covered: `{ {','.join(symbols)} }` exported on
  {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')}.
- Generator commit: ``git rev-parse HEAD`` of the source repo.
- All file hashes recorded in `manifest.json`.

## Citation

If you use this dataset, please cite the source repository:

```
HYDRA mk4 — multi-agent ONNX trading bot. https://github.com/bongc4947/mtbotmk1
```

## Kaggle metadata

`dataset-metadata.json` is pre-filled for the Kaggle CLI:

```bash
kaggle datasets create -p {out_root.name}/
```

If `id` does not match an existing dataset, edit it before running the
command. Default placeholder: `{kaggle_id}`.
"""
    text = yaml + body
    (out_root / "README.md").write_text(text, encoding="utf-8")
    (out_root / "dataset_card.md").write_text(text, encoding="utf-8")


def _write_manifest(out_root: Path, manifest: list[dict], *,
                    license_id: str, title: str, kaggle_id: str) -> None:
    obj = {
        "name":        title,
        "kaggle_id":   kaggle_id,
        "license":     license_id,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exporter":    "python/export_dataset.py",
        "n_files":     len(manifest),
        "files":       sorted(manifest, key=lambda m: m["path"]),
    }
    (out_root / "manifest.json").write_text(json.dumps(obj, indent=2),
                                            encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output directory (default: {DEFAULT_OUT})")
    p.add_argument("--include", nargs="+",
                   choices=["bars", "labels", "all"],
                   default=["all"], help="data blocks to export (default: all)")
    p.add_argument("--symbols", nargs="+", default=None,
                   help=f"subset of symbols (default: all under {PARQUET_DIR})")
    p.add_argument("--license", dest="license_id", default="CC-BY-NC-4.0",
                   choices=list(KNOWN_LICENSES.keys()),
                   help="license identifier (default: CC-BY-NC-4.0)")
    p.add_argument("--kaggle-id", default="bongc4947/hydra-mk4-bars",
                   help='Kaggle dataset id "user/slug" (default: bongc4947/hydra-mk4-bars)')
    p.add_argument("--title", default="HYDRA mk4 Multi-Asset M5 Bars + Labels")
    p.add_argument("--subtitle",
                   default=("M5 OHLCV bars across forex / metals / indices / "
                            "crypto / energy with pre-computed direction labels, "
                            "macro calendar, and CFTC + FRED context."))
    p.add_argument("--no-csv", dest="write_csv", action="store_false",
                   help="don't duplicate parquet bars as csv (saves space)")
    p.add_argument("--no-labels", dest="write_labels", action="store_false",
                   help="don't compute direction labels (parquet bars only)")
    p.add_argument("--zip", dest="make_zip", action="store_true",
                   help="also produce <out>.zip alongside the directory")
    args = p.parse_args(argv)

    out_root: Path = args.out
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / SCHEMA_DIR).mkdir(exist_ok=True)

    blocks = set(args.include)
    if "all" in blocks:
        blocks = {"bars", "labels"}

    available = list(_pick_latest_per_symbol().keys())
    symbols = args.symbols or available
    bad = [s for s in symbols if s not in ALL_SYMBOLS]
    if bad:
        log.warning("ignoring symbols not in ALL_SYMBOLS: %s", bad)
        symbols = [s for s in symbols if s in ALL_SYMBOLS]

    manifest: list[dict] = []

    bars_kept: dict[str, pd.DataFrame] = {}
    if "bars" in blocks:
        bars_kept = _export_bars(symbols, out_root=out_root,
                                  write_csv=args.write_csv, manifest=manifest)
    if "labels" in blocks and args.write_labels:
        if not bars_kept:
            # Re-load just to label, in case user asked --no-bars + --include labels
            bars_kept = _export_bars(symbols, out_root=out_root,
                                      write_csv=False, manifest=manifest)
        _export_labels(bars_kept, out_root=out_root, manifest=manifest)
    _write_kaggle_metadata(out_root, kaggle_id=args.kaggle_id,
                           license_id=args.license_id,
                           title=args.title, subtitle=args.subtitle)
    _write_license(out_root, args.license_id)
    _write_card(out_root, license_id=args.license_id,
                title=args.title, subtitle=args.subtitle,
                kaggle_id=args.kaggle_id, manifest=manifest,
                symbols=sorted(bars_kept.keys()))
    _write_manifest(out_root, manifest,
                    license_id=args.license_id, title=args.title,
                    kaggle_id=args.kaggle_id)

    n_files = len(manifest) + 5   # + manifest, kaggle md, license, README, card
    total_mb = sum(m["bytes"] for m in manifest) / 1e6
    print()
    print(f"Wrote {n_files} files to {out_root}  ({total_mb:.1f} MB data payload)")

    if args.make_zip:
        zip_path = out_root.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for fp in sorted(out_root.rglob("*")):
                if fp.is_file():
                    z.write(fp, fp.relative_to(out_root.parent))
        print(f"Zip:  {zip_path}  ({zip_path.stat().st_size/1e6:.1f} MB)")

    print()
    print("Next steps:")
    print(f"  Kaggle:        kaggle datasets create -p {out_root}/")
    print(f"  HuggingFace:   huggingface-cli upload <user>/<repo> {out_root}/ . --repo-type=dataset")
    print(f"  Zenodo / OSF:  upload {out_root.name}.zip via web UI (use --zip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
