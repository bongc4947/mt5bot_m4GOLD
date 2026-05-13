# Dockerfile — minimal CUDA image with the HYDRA mk4 runner pre-baked.
#
# Build:
#   docker build -t hydra-mk4-trainer cloud/
#
# Run (with NVIDIA Container Toolkit):
#   docker run --gpus all --rm \
#     -e BUNDLE_URL="https://your.host/HYDRA4_data_bundle.zip" \
#     -e EPOCHS=60 \
#     -v $(pwd)/out:/workspace/out \
#     hydra-mk4-trainer
#
# Push to a registry once for RunPod / vast.ai:
#   docker tag  hydra-mk4-trainer ghcr.io/<user>/hydra-mk4-trainer:latest
#   docker push                    ghcr.io/<user>/hydra-mk4-trainer:latest

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git curl ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install heavy deps so the actual run doesn't pay for them.
# Pin torch + cu121 wheels for determinism. The runner re-runs `pip install`
# but those are no-ops if everything is already present.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision

RUN python3 -m pip install \
        numpy pandas pyarrow scipy scikit-learn \
        onnx onnxruntime onnxscript \
        psutil requests yfinance fredapi pandas-datareader \
        hmmlearn tqdm matplotlib

WORKDIR /workspace

COPY runner.sh /workspace/runner.sh
RUN chmod +x /workspace/runner.sh

ENTRYPOINT ["/workspace/runner.sh"]
