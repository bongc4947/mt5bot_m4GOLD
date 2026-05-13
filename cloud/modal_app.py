"""
cloud/modal_app.py — serverless HYDRA mk4 training on Modal.

USAGE
-----
    pip install modal
    modal token new                                    # one-time auth

    # Set your secrets
    modal secret create hydra-bundle BUNDLE_URL=https://your.host/HYDRA4_data_bundle.zip

    # One-off run on T4
    modal run cloud/modal_app.py::train

    # One-off run on A100, 40 epochs
    modal run cloud/modal_app.py::train --gpu A100 --epochs 40

    # Schedule weekly retrains
    modal deploy cloud/modal_app.py
    # → 'hydra-mk4-train' app appears in the Modal dashboard with a cron
    #    that fires every Sunday at 02:00 UTC. Modify the schedule= line
    #    below and re-deploy if you want a different cadence.

WHY MODAL
---------
- Pay per second of GPU time (~$0.59/hr T4, ~$3.10/hr A100 in 2026).
- $30/mo free credit covers ~50 hours of T4 training.
- Persistent Volume so the parquet bundle is cached between runs.
- Built-in cron scheduling — set it once, walk away.
- No idle disconnect or 12 h session cap like Colab.

This file is a thin wrapper around cloud/runner.sh — same pipeline,
same env vars, just running inside a Modal Function.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Image: CUDA + Python + this repo (pulled fresh each cold start)
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
                              add_python="3.11")
    .apt_install("git", "curl", "ca-certificates")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
        "onnx", "onnxruntime", "onnxscript",
        "psutil", "requests", "yfinance", "fredapi", "pandas-datareader",
        "hmmlearn", "tqdm",
    )
)

app = modal.App("hydra-mk4-train", image=image)

# Volumes: keep the parquet bundle between runs so we don't re-download.
data_volume   = modal.Volume.from_name("hydra-mk4-data",   create_if_missing=True)
output_volume = modal.Volume.from_name("hydra-mk4-output", create_if_missing=True)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    gpu="T4",                                      # override at call time: --gpu A100
    timeout=60 * 60 * 6,                           # 6h hard cap — plenty for full pipeline
    volumes={
        "/data": data_volume,
        "/out":  output_volume,
    },
    secrets=[modal.Secret.from_name("hydra-bundle")],   # must define BUNDLE_URL
)
def train(
    epochs: int = 60,
    seed: int = 42,
    symbols: str = "",
    repo_branch: str = "master",
    batch_size: str = "",   # "" = auto
    skip_compliance: bool = False,
    skip_audit: bool = False,
):
    """Run the full HYDRA mk4 training pipeline inside this Modal container."""
    import os, subprocess, pathlib, shutil

    repo_dir = "/repo/MT5_bot_mk4"
    if not pathlib.Path(repo_dir, ".git").exists():
        subprocess.check_call([
            "git", "clone", "--branch", repo_branch, "--single-branch", "--depth", "1",
            "https://github.com/bongc4947/mtbotmk1.git", repo_dir,
        ])
    runner = f"{repo_dir}/cloud/runner.sh"
    subprocess.check_call(["chmod", "+x", runner])

    # Local cache for the bundle inside the persistent volume
    bundle_url = os.environ["BUNDLE_URL"]
    cached = pathlib.Path("/data/HYDRA4_data_bundle.zip")
    if bundle_url.startswith(("http://", "https://")) and not cached.exists():
        subprocess.check_call(["curl", "-fL", "--retry", "3",
                               "-o", str(cached), bundle_url])
        bundle_url = f"file://{cached}"
    elif cached.exists():
        bundle_url = f"file://{cached}"

    env = {
        **os.environ,
        "REPO_DIR":          repo_dir,
        "REPO_BRANCH":       repo_branch,
        "OUT_DIR":           "/out",
        "BUNDLE_URL":        bundle_url,
        "EPOCHS":            str(epochs),
        "SEED":              str(seed),
        "RUN_AUDIT":         "0" if skip_audit else "1",
        "RUN_COMPLIANCE":    "0" if skip_compliance else "1",
        "SYMBOLS":           symbols,
        "HYDRA_BATCH_SIZE":  batch_size,
        "PYBIN":             "python3",
    }
    subprocess.check_call(["bash", runner], env=env)

    # Surface artifacts list for the run log
    out = pathlib.Path("/out")
    print("\n=== Outputs ===")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:<60} {p.stat().st_size/1e6:>8.2f} MB")

    # Persist back to the volume (Modal does this automatically at function exit
    # because /out is mounted as a Volume, but commit makes the writes visible
    # immediately for downstream functions).
    output_volume.commit()


# ---------------------------------------------------------------------------
# Scheduled retrain — uncomment the schedule= line and `modal deploy` to enable.
# ---------------------------------------------------------------------------
@app.function(
    gpu="T4",
    timeout=60 * 60 * 6,
    volumes={"/data": data_volume, "/out": output_volume},
    secrets=[modal.Secret.from_name("hydra-bundle")],
    # schedule=modal.Cron("0 2 * * 0"),   # ← every Sunday 02:00 UTC
)
def scheduled_retrain():
    """Cron entry — calls train() with default args."""
    train.remote()


# ---------------------------------------------------------------------------
# Local helper: download trained ONNX from the persistent Volume to your laptop
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def download(target: str = "./onnx_out"):
    """
    modal run cloud/modal_app.py::download --target ./local_models

    Pulls the latest /out/ contents from the Modal Volume into a local dir.
    """
    import pathlib
    from modal import Volume
    out = pathlib.Path(target)
    out.mkdir(parents=True, exist_ok=True)
    vol = Volume.from_name("hydra-mk4-output")
    for f in vol.iterdir("/"):
        with open(out / f.path.split("/")[-1], "wb") as fh:
            for chunk in vol.read_file(f.path):
                fh.write(chunk)
    print(f"downloaded → {out.resolve()}")
