"""
cloud/kaggle/build_bars.py — one-cell helper: build the GOLD M5 bar file.

WHY
---
The Kaggle tick dataset ships raw ticks only. AURUM trains on M5 bars, so
the datamodule resamples 173M ticks -> M5 bars on every fresh session
(~15 min). Build that file ONCE with this helper, add it to your dataset,
and every future training run skips the tick build entirely.

HOW TO USE ON KAGGLE
--------------------
 1. New notebook (no GPU needed — this is pure pandas).
 2. Internet ON, attach the hydra4-tick-data-bundle dataset.
 3. Paste this whole file into one cell and run (~15 min).
 4. Download  /kaggle/working/HYDRA4_5MFROMTICKS_GOLD.parquet  from the
    Output tab.
 5. Add that file to your hydra4-tick-data-bundle dataset
    (dataset page -> New Version -> upload alongside the ticks).

From then on config.py auto-detects the prebuilt M5 parquet under
/kaggle/input and the datamodule loads it directly — no rebuild.
This needs only pandas / pyarrow / numpy (already in the Kaggle image).
"""

import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/bongc4947/mt5bot_m4GOLD.git"
REPO_DIR = "/kaggle/working/MT5bot_m4Gold"


def sh(cmd: str, cwd: str | None = None) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def main() -> int:
    # 1. clone (or refresh) the repo
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        sh(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
    else:
        sh("git fetch --depth 1 origin master && git reset --hard FETCH_HEAD",
           cwd=REPO_DIR)

    sys.path.insert(0, os.path.join(REPO_DIR, "python"))

    # 2. confirm the tick dataset is attached
    import glob
    hits = glob.glob("/kaggle/input/**/HYDRA4_TICKS_GOLD.parquet", recursive=True)
    print("GOLD ticks found at:", hits)
    if not hits:
        sys.exit("ABORT: no HYDRA4_TICKS_GOLD.parquet under /kaggle/input — "
                 "attach the hydra4-tick-data-bundle dataset (Add Input).")

    # 3. build M5 bars from ticks (config auto-detects the dataset path;
    #    load_or_build_bars resamples + caches the parquet)
    import config
    from strategies_common import load_or_build_bars
    print(f"TICKS_DIR  = {config.TICKS_DIR}")
    print(f"PARQUET_DIR = {config.PARQUET_DIR}")
    print("building GOLD M5 bars from ticks — this is the ~15 min step ...",
          flush=True)
    bars = load_or_build_bars("GOLD", "5min")
    if bars is None:
        sys.exit("ABORT: bar build returned nothing — check the tick parquet.")
    print(f"built {len(bars):,} M5 bars  "
          f"({bars['time'].iloc[0]} -> {bars['time'].iloc[-1]})")

    # 4. stage the file at the working-dir root for an easy download
    src = config.PARQUET_DIR / "HYDRA4_5MFROMTICKS_GOLD.parquet"
    dst = "/kaggle/working/HYDRA4_5MFROMTICKS_GOLD.parquet"
    shutil.copy(src, dst)
    size_mb = os.path.getsize(dst) / 1e6
    print(f"\nDONE — staged {dst}  ({size_mb:.1f} MB)")
    print("Download it from the Output tab and add it to your "
          "hydra4-tick-data-bundle dataset (New Version).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
