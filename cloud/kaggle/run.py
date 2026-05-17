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
# 20 epochs completes comfortably inside one Kaggle GPU session on a first
# run. Raise it (set the EPOCHS env var) once you've confirmed the pipeline
# runs end-to-end and you have session quota to spare.
EPOCHS = int(os.environ.get("EPOCHS", "20"))


def sh(cmd: str, cwd: str | None = None) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def main() -> int:
    # 1. clone (or refresh) the repo
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        sh(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
    else:
        sh("git fetch --depth 1 origin master && git reset --hard origin/master",
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
    sh(f"{sys.executable} python/train_aurum.py all "
       f"--epochs {EPOCHS} --use-gpu --batch-size 256",
       cwd=REPO_DIR)

    # 5. surface the artifacts
    sh("ls -la onnx_out", cwd=REPO_DIR)
    print("\nDONE — download onnx_out/ from the Kaggle Output tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
