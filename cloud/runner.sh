#!/usr/bin/env bash
# cloud/runner.sh — MT5bot_m4Gold universal training entrypoint.
#
# Same script runs:
#   - locally (Linux/macOS/Windows-WSL) on CPU or NVIDIA GPU
#   - in any cloud notebook (Kaggle, Colab, RunPod, vast.ai, Lambda, Modal)
#
# Configuration via env vars (all optional):
#   EPOCHS              training epochs for the AI head (default 60)
#   SEED                RNG seed (default 42)
#   STRATEGIES          subset of H1,H4,H5,H6 (default: all four)
#   TRAIN_MODE          strategies | ai | both (default: both)
#   FORCE_CPU=1         skip GPU detection, run on CPU
#   FORCE_GPU=1         require CUDA — fail if not available
#   REPO_URL / REPO_BRANCH / REPO_DIR  for `git clone` workflows
#
# Hardware auto-detection runs inside python/hardware_detector.py — no
# manual switch needed for the CPU-vs-GPU split. The runner just sets env
# defaults; the trainer picks batch size, AMP, and worker count from it.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
: "${REPO_URL:=}"
: "${REPO_BRANCH:=master}"
: "${REPO_DIR:=${PWD}/MT5bot_m4Gold}"
: "${OUT_DIR:=${PWD}/out}"
: "${EPOCHS:=60}"
: "${SEED:=42}"
: "${TRAIN_MODE:=both}"        # strategies | ai | both
: "${STRATEGIES:=H1,H4,H5,H6}"
: "${FORCE_CPU:=0}"
: "${FORCE_GPU:=0}"

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Repo bootstrap (only when REPO_URL is set, e.g. cloud runner)
# ---------------------------------------------------------------------------
if [[ -n "$REPO_URL" ]]; then
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "[runner] cloning $REPO_URL @ $REPO_BRANCH -> $REPO_DIR"
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
  else
    echo "[runner] reusing $REPO_DIR"
    (cd "$REPO_DIR" && git fetch --depth 1 origin "$REPO_BRANCH" && git reset --hard "origin/$REPO_BRANCH")
  fi
  cd "$REPO_DIR"
fi

# ---------------------------------------------------------------------------
# Python env
# ---------------------------------------------------------------------------
python -m pip install --upgrade pip
python -m pip install -r python/requirements-train.txt

# ---------------------------------------------------------------------------
# Hardware probe — never fatal; just informs the user what we're about to do.
# ---------------------------------------------------------------------------
if [[ "$FORCE_CPU" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  echo "[runner] FORCE_CPU=1 — masking CUDA devices"
fi
python -c "
import sys
sys.path.insert(0, 'python')
from hardware_detector import detect
hw = detect()
print(f'[hardware] tier={hw.tier} device={hw.device} amp={hw.amp} '
      f'batch={hw.batch_size} max_bars={hw.max_bars:,} workers={hw.workers} '
      f'vram={hw.vram_gb:.1f}GB ram={hw.ram_gb:.1f}GB')
if '${FORCE_GPU}' == '1' and hw.device != 'cuda':
    sys.exit('FORCE_GPU=1 set but no CUDA available')
"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
if [[ "$TRAIN_MODE" == "strategies" || "$TRAIN_MODE" == "both" ]]; then
  echo "[runner] === strategies ($STRATEGIES) ==="
  python python/train_strategies.py GOLD --strategies "$STRATEGIES" --seed "$SEED"
fi

if [[ "$TRAIN_MODE" == "ai" || "$TRAIN_MODE" == "both" ]]; then
  echo "[runner] === AI direction head ==="
  python python/train.py gold --epochs "$EPOCHS" --seed "$SEED" --skip-extract
fi

# ---------------------------------------------------------------------------
# Bundle ONNX + specs for download
# ---------------------------------------------------------------------------
ONNX_DIR="${ONNX_OUTPUT_DIR:-onnx_out}"
echo "[runner] bundling artifacts from $ONNX_DIR -> $OUT_DIR"
if [[ -d "$ONNX_DIR" ]]; then
  cp -r "$ONNX_DIR"/* "$OUT_DIR/" 2>/dev/null || true
fi
ls -la "$OUT_DIR"
echo "[runner] done."
