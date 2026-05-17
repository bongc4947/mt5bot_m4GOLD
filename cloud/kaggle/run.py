"""
cloud/kaggle/run.py — one-cell Kaggle entrypoint for MT5bot_m4Gold.

HOW TO USE ON KAGGLE
--------------------
 1. Create a new Kaggle Notebook.
 2. Notebook settings (right panel):
      - Accelerator : GPU T4 x2  (or P100)
      - Internet    : ON         (needed for git clone + pip)
 3. Add data -> your dataset  "hydra4-tick-data-bundle"
    It mounts read-only at /kaggle/input/hydra4-tick-data-bundle/ and
    config.py auto-detects it (any /kaggle/input/* dir holding
    HYDRA4_TICKS_GOLD.parquet works — the folder name is not hard-coded).
 4. Paste this whole file into one cell and run.

Outputs land in /kaggle/working/MT5bot_m4Gold/onnx_out/ — the AURUM ONNX
bundle + spec. Download them from the notebook's Output tab and drop
them into MT5's Common Files.

The pipeline: clone -> install -> train_aurum.py all (baseline ->
pretrain -> finetune -> meta -> conformal -> export).
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/bongc4947/mt5bot_m4GOLD.git"
REPO_DIR = "/kaggle/working/MT5bot_m4Gold"
# FULL-STRENGTH defaults. The conservative first-run caps (12 epochs /
# 90k bars) were only there to prove the pipeline; that succeeded, and
# fine-tune val_PF was still climbing steeply (1.2 -> 1.75) when the cap
# cut it off. These settings give AURUM a real shot at the deploy gate:
#   EPOCHS=0    -> each phase uses its tuned config default
#                  (SSL pretrain 50 epochs, fine-tune 60 w/ early-stop)
#   MAX_BARS=0  -> use ALL available GOLD history
# A full run is ~4-6 h on a Kaggle T4 — fits one GPU session. To do a
# fast smoke run again, set EPOCHS=12 and MAX_BARS=90000 in the env.
EPOCHS = int(os.environ.get("EPOCHS", "0"))
MAX_BARS = int(os.environ.get("MAX_BARS", "0"))


def sh(cmd: str, cwd: str | None = None) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def main() -> int:
    # 1. clone (or refresh) the repo. `git fetch origin master` only moves
    #    FETCH_HEAD — on a shallow clone the origin/master ref may not
    #    exist, so reset to FETCH_HEAD, not origin/master.
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        sh(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
    else:
        sh("git fetch --depth 1 origin master && git reset --hard FETCH_HEAD",
           cwd=REPO_DIR)

    # 2. dependencies (Kaggle already ships torch/xgboost/onnx; this tops up)
    sh(f"{sys.executable} -m pip install -q -r python/requirements-train.txt",
       cwd=REPO_DIR)

    # 3. confirm the tick dataset is visible
    sh(f"{sys.executable} -c \"import sys; sys.path.insert(0,'python'); "
       f"import config; print('TICKS_DIR =', config.TICKS_DIR); "
       f"print('PARQUET_DIR =', config.PARQUET_DIR)\"", cwd=REPO_DIR)

    # 4. full AURUM pipeline (--use-gpu accelerates the XGBoost baseline +
    #    meta gate; the torch parts auto-detect CUDA via hardware_detector)
    flags = "--use-gpu --batch-size 256"
    if EPOCHS > 0:
        flags += f" --epochs {EPOCHS}"      # else: per-phase config defaults
    if MAX_BARS > 0:
        flags += f" --max-bars {MAX_BARS}"  # else: all available history
    sh(f"{sys.executable} python/train_aurum.py all {flags}", cwd=REPO_DIR)

    # 5. surface the artifacts
    sh("ls -la onnx_out", cwd=REPO_DIR)
    print("\nDONE — download onnx_out/ from the Kaggle Output tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
