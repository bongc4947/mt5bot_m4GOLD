# RunPod / vast.ai / Lambda — quickstart

These platforms rent bare GPU instances by the hour. Cheapest serious GPUs
in 2026 (RTX 3090 ≈ $0.20-0.40/h, A100 ≈ $1.50-3/h). Both expect either an
SSH key or a Docker image. Our [`Dockerfile`](Dockerfile) is the path of
least resistance — same image, three providers.

## Build & publish the image once

```bash
# Build (from repo root)
docker build -t hydra-mk4-trainer cloud/

# Push to GitHub Container Registry (free, public or private)
echo "$GITHUB_PAT" | docker login ghcr.io -u <user> --password-stdin
docker tag  hydra-mk4-trainer ghcr.io/<user>/hydra-mk4-trainer:latest
docker push                    ghcr.io/<user>/hydra-mk4-trainer:latest
```

The image is ~6 GB; it bakes torch + cu121 + all numeric deps so the actual
run starts in seconds rather than minutes.

---

## RunPod (recommended for paid runs)

Web UI: https://runpod.io/console/pods

1. **Pods → Deploy → GPU Pod**.
2. Pick a GPU: `RTX 3090` ($0.34/h on-demand, $0.22/h spot) or `RTX 4090`
   for ~50% faster training at $0.50-0.70/h.
3. Storage: attach a **Network Volume** (e.g., 50 GB) — this is the data /
   output cache that survives across pods. Mount at `/workspace`.
4. **Container image**: `ghcr.io/<user>/hydra-mk4-trainer:latest`.
5. **Environment variables**:
   - `BUNDLE_URL` — `https://your.host/HYDRA4_data_bundle.zip` (or
     `s3://bucket/key` if you've baked AWS creds into the image).
   - `OUT_DIR` — `/workspace/out` (so the persistent volume keeps the ONNX).
   - `EPOCHS`, `SEED`, `HYDRA_BATCH_SIZE` — optional.
6. **Container start command** — leave blank to use the image's `ENTRYPOINT`,
   which is `runner.sh`.
7. Deploy. The pod runs `runner.sh` and exits when training finishes; you
   get billed for that wall time only.
8. Recover: `runpodctl receive <pod-id>:/workspace/out ./local_out`
   (or use the web file browser).

For a **persistent** dev pod (kept alive between training runs):
- Use the `runpod/pytorch:2.4.0-py3.11-cuda12.1.1` image directly.
- SSH in, `git clone` the repo, `pip install -r python/requirements-train.txt`,
  then `bash cloud/runner.sh`.

---

## <a name="vastai"></a>vast.ai

Cheapest of the three (~$0.15-0.25/h for an RTX 3090) but instances are
spot-style — they can disappear if the host accepts a higher-paying job.
Good for one-shot training, less good for unattended long-running jobs.

1. https://cloud.vast.ai/templates/ → New Template.
2. **Image**: `ghcr.io/<user>/hydra-mk4-trainer:latest`.
3. **Docker run options**:
   ```
   -e BUNDLE_URL=https://your.host/HYDRA4_data_bundle.zip
   -e EPOCHS=60
   -e OUT_DIR=/workspace/out
   ```
4. **On-start script** (defaults to image ENTRYPOINT — leave blank).
5. Save template, then *Create Instance* with that template, picking an
   RTX 3090 or 4090 with `Reliability ≥ 0.95` and `Verified` to avoid
   flaky hosts.
6. SSH in once with the provided command, then `tail -f /workspace/out/run_manifest.txt`
   to watch progress, or just check the instance log.

---

## Lambda Cloud

Less price-competitive in 2026, but reliable A10/A100/H100 if you need
the bigger card. Same Docker image; pick *Custom Image* on the Launch page.

`gpu_1x_a10` ≈ $0.50/h, `gpu_1x_a100` ≈ $1.10/h, `gpu_1x_h100` ≈ $2.50/h.

---

## Local CUDA box (free-after-purchase)

If you have an RTX 3060 or better at home, ignore the cloud entirely:

```bash
git clone https://github.com/bongc4947/mtbotmk1.git
cd mtbotmk1
BUNDLE_URL="file:///path/to/HYDRA4_data_bundle.zip" bash cloud/runner.sh
```

That's it. Trained ONNX lands in `./out/` plus `onnx_out/` in the repo.

---

## Cost rule of thumb (full pipeline, all 13 symbols, 60 epochs)

| GPU      | Wall time | Cost @ on-demand | Cost @ spot |
|----------|----------|------------------|-------------|
| T4       | ~50 min  | $0.40 (Modal)    | $0.30 (Kaggle = free) |
| RTX 3090 | ~25 min  | $0.15            | $0.09       |
| RTX 4090 | ~15 min  | $0.15            | $0.10       |
| A100 40G | ~10 min  | $0.50            | $0.30       |
| H100     | ~5 min   | $0.20            | n/a         |

Numbers approximate; varies with vendor and hour-of-day. **For weekly
retrains, RTX 3090 spot on RunPod ≈ $5/year.** Modal scheduled retrain on
T4 ≈ $20/year, fully hands-off.
