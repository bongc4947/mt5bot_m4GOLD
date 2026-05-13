# Cloud training — alternatives to Google Colab

The HYDRA mk4 training pipeline is a Linux/CUDA Python job that needs:
- ~13 GB RAM, ~6-15 GB VRAM (T4-class is enough; A100/L4/H100 finish faster)
- ~300 MB parquet bundle as input
- ~30-60 min to train all 4 directional agents + BCF on a T4
- ~50 MB ONNX outputs to recover

This directory has one universal runner ([`runner.sh`](runner.sh)) plus
thin per-platform wrappers, so the same training run reproduces on any of:

| Platform | Cost | Free tier? | Persistent storage | Idle disconnect | Setup difficulty | Verdict |
|---|---|---|---|---|---|---|
| **[Kaggle Notebooks](kaggle/)** | $0 | **Yes — 30 GPU hr/wk** | Yes (Datasets) | 12 h sessions, 20 min idle | ★ trivial | **Best free option.** Beats Colab on persistence — attach the parquet bundle as a Kaggle Dataset, never re-upload. |
| **[Modal](modal_app.py)** | ~$0.59/hr T4, ~$3.10/hr A100 | $30/mo free credit | Yes (Volumes) | None — runs to completion | ★★ pip install modal | Best for scheduled retrains; serverless, billed per second. |
| **[RunPod](runpod_quickstart.md)** | $0.20-0.40/hr RTX 3090, $2-3/hr A100 | None | Yes (Network Volume) | None | ★★ docker pull | Cheapest serious GPU per hour; SSH+Jupyter, you own the box. |
| **[vast.ai](runpod_quickstart.md#vastai)** | $0.15-0.50/hr RTX 3090 | None | Yes (per-instance) | None | ★★★ container yaml | Cheapest of all; spot-style market; instances can disappear. |
| Lightning.ai Studios | $0.40/hr L4 | 4 free GPU hr/mo | Yes (Studio) | Long-idle suspend | ★★ web UI | Persistent VS Code; nice UX. |
| Paperspace Gradient | ~$0.45-3/hr | Free M4000 sometimes | Yes | None | ★★ web UI | Decent but less price-competitive in 2026. |
| Google Colab | Free / $10-50/mo | Yes (T4) | **No** (Drive mount only) | 12 h, ~90 min idle | ★ trivial | Where you started; fine for ad-hoc, painful for scheduled. |
| Local CUDA box | One-off hardware | Free after purchase | Yes | Never | ★★★ NVIDIA drivers + CUDA | Best if you have a 3060+. Run `bash cloud/runner.sh` directly. |

## Recommendation

- **For "I want it free, less friction than Colab"** → **Kaggle Notebooks**.
  Upload the parquet bundle once as a private Kaggle Dataset, attach to a
  notebook, click Save & Run. Output ONNX is automatically saved as the
  notebook's output and downloadable.

- **For "I want this to retrain weekly without me babysitting"** → **Modal**.
  Define a cron schedule on the `@app.function` and walk away. Pay only
  for actual GPU seconds.

- **For "I need an A100/H100 cheap for an evening of experiments"** →
  **RunPod** (more reliable) or **vast.ai** (cheaper, more variance).
  Both pull our Docker image directly.

- **For "I have a 3060/3090/4090 at home"** → just run
  [`cloud/runner.sh`](runner.sh) on the local box. Cheapest of all.

## How the pipeline works

All wrappers below converge on **the same logical flow**:

```
1. Clone the repo (or update existing checkout)
2. Install requirements-train.txt (CPU+CUDA cross-platform)
3. Pull a parquet data bundle from a configurable source:
     a Kaggle Dataset, a URL (S3/HF/GCS), Google Drive, or a mounted volume
4. Verify the bundle (sha256 if manifest present)
5. Run audit_leakage.py against the cached symbols (cached after first run)
6. Dispatch on TRAIN_MODE:
     directional     -> python/train.py $TRAIN_AGENT ... --sampler ... --samples-per-epoch ...
     scalp           -> python/train_scalp.py $SCALP_SYMBOLS --epochs $SCALP_EPOCHS --window $SCALP_WINDOW
     hedge (auto)    -> python/train_hedge.py screen, then train across passing pairs
     hedge (explicit)-> one `train_hedge.py train --pair A B` per pair in $HEDGE_PAIRS
     scalp_and_hedge -> both scalp + hedge
7. Run python/train.py compliance --seed $SEED  (directional + TRAIN_AGENT=all only)
8. Copy onnx_out/*.onnx + meta.json to a configurable destination
```

`runner.sh` does steps 1-8 directly given env vars; each platform wrapper
just sets those env vars and calls `runner.sh`.

## Common environment variables

| Variable | Purpose | Default |
|---|---|---|
| `REPO_URL` | Source repo (https/ssh git url) | this repo |
| `REPO_BRANCH` | Branch to use | `master` |
| `BUNDLE_URL` | Where to fetch the parquet bundle (`http://...`, `s3://...`, `kaggle://user/slug`, or a local path) | (required) |
| `OUT_DIR` | Where to write ONNX + logs | `/workspace/out` |
| `EPOCHS` | Override training epochs | `60` |
| `SEED` | Random seed | `42` |
| `HYDRA_BATCH_SIZE` | Force a batch size (skip auto-detect) | (auto) |
| `RUN_AUDIT` | `1` to run leakage audit before training | `1` |
| `RUN_COMPLIANCE` | `1` to also train BCF (directional + agent=all only) | `1` |
| `SYMBOLS` | Space-separated symbols, or empty for all | (all) |
| `TRAIN_MODE` | `directional` / `scalp` / `hedge` / `scalp_and_hedge` | `directional` |
| `TRAIN_AGENT` | (directional) `all` / `prism` / `gnn` / `apex` / `ce` | `all` |
| `TRAIN_SAMPLER` | (directional) `chronological` / `random-window` | `chronological` |
| `SAMPLES_PER_EPOCH` | (directional, random-window) draws per epoch | `100000` |
| `SCALP_SYMBOLS` | (scalp) space-separated symbols | `EURUSD GBPUSD USDJPY GOLD` |
| `SCALP_EPOCHS` | (scalp) epochs per symbol | `30` |
| `SCALP_WINDOW` | (scalp) tick-bars per training sample | `64` |
| `SCALP_BATCH_SIZE` | (scalp) loader batch size | `1024` |
| `SCALP_SHOULD_TRADE_THRESHOLD` | (scalp) gate threshold | `0.55` |
| `HEDGE_PAIRS` | (hedge) `auto` or `"A/B C/D"` explicit list | `auto` |
| `HEDGE_EPOCHS` | (hedge) epochs per pair | `30` |
| `HEDGE_BATCH_SIZE` | (hedge) loader batch size | `512` |
| `COINT_P_THRESHOLD` | (hedge / auto) Engle-Granger p-value | `0.05` |
| `COINT_MIN_WINDOWS` | (hedge / auto) consecutive windows that must pass | `3` |
| `COINT_WINDOW_BARS` | (hedge / auto) bars per cointegration window | `10000` |
| `PARALLEL_TRAINING` | `1` to fire sub-trainings concurrently in one cell | `0` |
| `MAX_PARALLEL_WORKERS` | concurrent worker cap (T4: 4 fits in 16 GB VRAM) | `4` |

## Parallel training (in-cell concurrency)

Splitting work across 4 separate notebook cells does **not** speed
anything up — Jupyter runs cells sequentially in a single kernel. The
fix is `PARALLEL_TRAINING = True` in `notebook_run.py`, which spawns
sub-trainings concurrently *inside one cell*:

| TRAIN_MODE | What runs in parallel |
|---|---|
| `directional` + `TRAIN_GROUP=all` | `prism` + `gnn` + `apex` + `ce` simultaneously |
| `scalp` | every symbol in `SCALP_SYMBOLS`, capped at `MAX_PARALLEL_WORKERS` |
| `hedge` (explicit) | every pair in `HEDGE_PAIRS`, capped at `MAX_PARALLEL_WORKERS` |
| `hedge` (auto) | screen runs first sequentially; then all passing pairs in parallel |

Each parallel job's stdout/stderr is captured to
`MT5_bot_mk4/log_<agent_or_symbol>.txt` so the cell output stays
readable. Wall-clock savings on a T4 vs. fully sequential: ~2-3× for
directional, ~3-4× for scalp/hedge with many symbols.

**Memory guidance** — `MAX_PARALLEL_WORKERS=4` fits a T4 (16 GB VRAM)
because each PyTorch process holds ~1.5 GB of CUDA context plus model
+ batch state. If you OOM, drop to 3 or 2. On A100 (40 GB) try 6-8.

## Downloading the trained ONNX models

After training, `runner.sh` writes:
- `onnx_out/HYDRA4_*.onnx` — one per model
- `onnx_out/HYDRA4_*_meta.json` — calibration + skill metrics + deploy flag
- `onnx_out/run_manifest.txt` — commit hash, mode, completion time
- `onnx_out/HYDRA4_onnx_<mode>_<commit>_<timestamp>.zip` — **everything bundled into one zip for one-click download**

### Per-platform retrieval

| Platform | How to grab the files |
|---|---|
| **Kaggle** | After `Save Version → Save & Run All (Commit)` finishes, open the notebook viewer. Right sidebar → **Output** tab → expand `onnx_out/` → either download files individually or grab the single `HYDRA4_onnx_*.zip`. CLI alternative: `kaggle kernels output <user>/<kernel-slug> -p ./local_dir/`. |
| **Colab** | In a NEW cell after training: `from google.colab import files; files.download('/content/onnx_out/HYDRA4_onnx_*.zip')`. Or copy to Drive: `!cp -r /content/onnx_out /content/drive/MyDrive/HYDRA4_models/`. |
| **RunPod / vast.ai / Lambda** | `scp -r <user>@<host>:/workspace/onnx_out/  ./local_dir/`  or `rsync -avz <user>@<host>:/workspace/onnx_out/ ./local_dir/`. The Web Terminal also has a download button if you click on the file. |
| **Local** | Files are already on disk under `BASE_DIR/onnx_out/`. Drop them into the auto-detected `MT5_COMMON_DIR` (`%APPDATA%\MetaQuotes\Terminal\<id>\MQL5\Files\` on Windows). The EA hot-reloads ONNX without restart. |

The notebook's last cell also prints a per-platform reminder with the
exact command for *your* environment after every successful run.

## Quick start — pick your platform

- **Any notebook (Kaggle / Colab / Lightning / Paperspace / local Jupyter)**:
  paste [`cloud/notebook_run.py`](notebook_run.py) into a cell — it auto-
  detects the platform and self-adjusts paths, Drive mounting, and data
  lookup. Edit only the constant for *your* platform.
- Kaggle-specific notes: [`cloud/kaggle/README.md`](kaggle/README.md)
- Modal: [`cloud/modal_app.py`](modal_app.py) (`pip install modal && modal run cloud/modal_app.py`)
- RunPod / vast.ai: [`cloud/runpod_quickstart.md`](runpod_quickstart.md)
- Bare-metal local: `bash cloud/runner.sh` (set `BUNDLE_URL=file:///path/to/HYDRA4_data_bundle.zip`)
