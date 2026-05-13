# Kaggle Notebooks — training pipeline

Kaggle Notebooks give you **30 GPU hours/week of T4 (×2) or P100** for free,
plus persistent dataset attachment (no re-uploading bundles per session).

## One-time setup

1. **Make a private Kaggle Dataset for the parquet bundle.**
   - On your local Windows + MT5 box:
     ```
     # Default: M5 bars (chronological training)
     python python/extract_data.py
     python python/bundle_data.py --include cal --out HYDRA4_data_bundle.zip

     # OR — mk4.7 tick-mode (random-window training), single command:
     python python/extract_data.py --source ticks --max-size-mb 2048 --bundle
     ```
   - Go to https://www.kaggle.com/datasets and click *New Dataset*.
   - Upload `HYDRA4_data_bundle.zip`. Mark Private. Title it
     e.g. `<your-user>/hydra4-tick-data-bundle`.
   - From now on you upload a new version once a quarter (or whenever you
     pull fresh bars), instead of every Colab session.

2. **Or**: skip step 1 and use the public dataset built by `export_dataset.py`
   if you've already published one (see [`python/export_dataset.py`](../../python/export_dataset.py)).

## Per-run setup (takes 30 seconds)

3. Click *New Notebook*, then in the right sidebar:
   - **Settings → Accelerator → GPU T4×2** (or P100).
   - **Add Data → Search your dataset → Add**.
     The bundle now lives at `/kaggle/input/<dataset-slug>/`.

4. Paste the contents of [`cloud/notebook_run.py`](../notebook_run.py)
   (the universal entrypoint — auto-detects Kaggle, Colab, RunPod, or
   local) into the first cell. The Kaggle-specific copy at
   [`run.py`](run.py) is kept identical for paste-history compatibility.

   **All editable knobs** are in the EDIT block at the top — change only
   the ones for *your* TRAIN_MODE:
   - `TRAIN_MODE`              — `directional` / `scalp` / `hedge` / `scalp_and_hedge`
   - `KAGGLE_DATASET_SLUG`     — your dataset slug
   - directional: `TRAIN_GROUP`, `EPOCHS`, `TRAIN_SAMPLER`, `SAMPLES_PER_EPOCH`
   - scalp: `SCALP_SYMBOLS`, `SCALP_EPOCHS`, `SCALP_WINDOW`, `SCALP_BATCH_SIZE`
   - hedge: `HEDGE_PAIRS`, `HEDGE_EPOCHS`, `COINT_P_THRESHOLD`,
     `COINT_MIN_WINDOWS`, `COINT_WINDOW_BARS`

   See [`cloud/CELLS_TEMPLATE.md`](../CELLS_TEMPLATE.md) for the
   ready-to-paste recipes (directional, scalp-only, hedge-only,
   scalp+hedge in one run, four-cell parallel directional).

   Then *Save & Run All (Commit)*.
   - That commits a notebook version. Outputs in `/kaggle/working/onnx_out/`
     are preserved as the version's downloadable artifacts.
   - You can also click *Save Version → Save & Run All* on a schedule
     (Kaggle Pro) to retrain on a cron.

## Conserving the free 30 GPU-hours/week — parallel training in one cell

The mk4.7 default `PARALLEL_TRAINING = True` fires sub-trainings
concurrently inside a single cell. Jupyter cells in one notebook run
sequentially, so 4 training cells = 4× wall-clock. With
`PARALLEL_TRAINING=True` in **one** cell, all four directional groups
(or all scalp symbols, or all hedge pairs) train at once on the same
GPU — a T4 fits 4 concurrent PyTorch processes in its 16 GB VRAM.

```python
# In the EDIT block at the top of cloud/notebook_run.py:
TRAIN_MODE           = "directional"   # or "scalp" / "hedge" / "scalp_and_hedge"
TRAIN_GROUP          = "all"
PARALLEL_TRAINING    = True            # mk4.7 default
MAX_PARALLEL_WORKERS = 4               # lower if you OOM
```

Each parallel job's stdout/stderr is captured to
`MT5_bot_mk4/log_<agent_or_symbol>.txt`. The cell prints a one-line
"finished (rc=0)" per job as they complete.

## Downloading the trained ONNX

Outputs land in `/kaggle/working/onnx_out/`, including one zip with
everything bundled:

```
HYDRA4_onnx_<TRAIN_MODE>_<commit>_<timestamp>.zip
```

To get them off Kaggle:

1. **Save Version → Save & Run All (Commit)** so the version's outputs
   get persisted (hitting "Run All" alone doesn't preserve them).
2. Once the version completes, open the notebook viewer page.
3. Right sidebar → **Output** tab → expand `onnx_out/` → click the
   three-dot menu on the file you want → **Download**. Or grab the
   single `.zip` for everything in one click.
4. CLI alternative if you have the Kaggle API set up:
   ```bash
   kaggle kernels output <your-user>/<kernel-slug> -p ./onnx_out/
   ```

After download, drop the files into `%APPDATA%\MetaQuotes\Terminal\Common\Files\`
on the MT5 box. The EA hot-reloads ONNX without restart.

## Why this beats Colab for this workflow

|                  | Colab           | Kaggle          |
|------------------|------------------|-----------------|
| Free GPU         | T4 (variable)   | T4×2 or P100    |
| Free hours       | undocumented    | **30 / week**   |
| Re-upload data   | every session   | **one-time**    |
| Output recovery  | manual download | auto (versioned)|
| Scheduled retrain| Pro only ($)    | Pro only ($)    |
| Drive friction   | yes             | none            |

If you already have a Kaggle account, the migration is literally:
upload the bundle once + paste `run.py`. No Drive, no `from google.colab import drive`.
