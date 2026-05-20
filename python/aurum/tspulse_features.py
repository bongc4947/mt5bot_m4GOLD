"""
aurum/tspulse_features.py — IBM Granite TSPulse-r1 as extra features
for the MetaTrend meta-gate.

Per-anchor, causal (uses only the past 512 M5 closes), four scalars:

  tspulse_fwd_logret   net log-return forecast over the next 16 bars
  tspulse_fwd_mag      magnitude of that forecast, in ATR-normalised units
  tspulse_recon_err    masked-reconstruction error (anomaly proxy)
  tspulse_fft_div      KL-divergence of forecast FFT vs observed FFT
                       (regime/structure deviation)

The model is 1M params, CPU-friendly. Results are cached to disk keyed by
(symbol, n_anchors, anchor_hash) so repeated trainer runs are instant.

Use:
    from aurum.tspulse_features import extract
    X_extra = extract(m5_df, anchors=anchor_index_array)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

_MODEL = None              # PyTorch model (research backend)
_SESS = None               # ONNX Runtime session (production-parity backend)
_CTX = 512                 # tspulse context length
_REPO = "ibm-granite/granite-timeseries-tspulse-r1"
_ONNX_PATH = (Path(__file__).parent.parent.parent
              / "onnx_out" / "M4GOLD_TSPULSE_GOLD.onnx")
TSPULSE_FEATURES = [
    "tspulse_fwd_logret",
    "tspulse_fwd_mag",
    "tspulse_recon_err",
    "tspulse_fft_div",
]
N_TSPULSE_FEATURES = len(TSPULSE_FEATURES)


def _disable_random_masking(base):
    """tspulse-r1 has random time + FFT masking active even in eval mode.
    Patch them out so the model is deterministic — mandatory for
    train/serve parity when the EA runs the exported ONNX live."""
    import types
    def _pt_time(self, inputs, past_observed_mask=None):
        return inputs, torch.zeros_like(inputs, dtype=torch.bool)
    def _pt_fft(self, fft_tensor):
        return fft_tensor, torch.zeros_like(fft_tensor)
    base.backbone.time_masker.forward = types.MethodType(
        _pt_time, base.backbone.time_masker)
    for _, sub in base.named_modules():
        if type(sub).__name__ == "TSPulseFFTMasker":
            sub.forward = types.MethodType(_pt_fft, sub)


def _get_model():
    global _MODEL
    if _MODEL is None:
        import warnings
        warnings.filterwarnings("ignore")
        from tsfm_public.models.tspulse import TSPulseForReconstruction
        log.info("[tspulse] loading %s ...", _REPO)
        _MODEL = TSPulseForReconstruction.from_pretrained(_REPO)
        _MODEL.eval()
        _disable_random_masking(_MODEL)
        log.info("[tspulse] loaded - %d params, ctx=%d (maskers patched)",
                 sum(p.numel() for p in _MODEL.parameters()),
                 _MODEL.config.context_length)
    return _MODEL


def _get_onnx_session():
    """Lazy-load the exported ONNX. Required for backend='onnx'."""
    global _SESS
    if _SESS is None:
        if not _ONNX_PATH.exists():
            raise FileNotFoundError(
                f"{_ONNX_PATH.name} not found. Run "
                "`python python/export_tspulse_onnx.py` first.")
        import onnxruntime as ort
        log.info("[tspulse] loading ONNX session: %s", _ONNX_PATH.name)
        _SESS = ort.InferenceSession(str(_ONNX_PATH),
                                      providers=["CPUExecutionProvider"])
    return _SESS


def _cache_path(symbol: str, anchors: np.ndarray,
                backend: str = "torch") -> Path:
    h = hashlib.sha1(anchors.astype(np.int64).tobytes()).hexdigest()[:16]
    base = Path(__file__).parent.parent.parent / "onnx_out"
    base.mkdir(parents=True, exist_ok=True)
    # v2: maskers patched out for determinism — invalidates v1 random-mask caches
    tag = "" if backend == "torch" else f"_{backend}"
    return base / f"tspulse_features_v2_{symbol}_{len(anchors)}_{h}{tag}.npy"


def _forward_batch(X: np.ndarray, backend: str):
    """
    Forward a [B, 512, 1] float32 batch through tspulse.

    Returns four numpy arrays:
        fcst    [B, 16]   forecast_output[:, :, 0]
        recon   [B, 512]  reconstruction_outputs[:, :, 0]
        obs_fft [B, 256]  original_fft_softmax[:, :, 0]
        prd_fft [B, 256]  fft_softmax_preds[:, :, 0]

    `backend='torch'` runs the original PyTorch model (research/maximum
    accuracy). `backend='onnx'` runs the exported M4GOLD_TSPULSE_GOLD.onnx
    — slightly different numerics due to dynamo export precision but
    bit-for-bit matches what the live MT5 EA will see (zero serve-time
    drift).
    """
    if backend == "torch":
        model = _get_model()
        with torch.no_grad():
            res = model(past_values=torch.from_numpy(X))
        return (res["forecast_output"].cpu().numpy()[:, :, 0],
                res["reconstruction_outputs"].cpu().numpy()[:, :, 0],
                res["original_fft_softmax"].cpu().numpy()[:, :, 0],
                res["fft_softmax_preds"].cpu().numpy()[:, :, 0])
    elif backend == "onnx":
        sess = _get_onnx_session()
        out = sess.run(None, {"past_values": X})
        return out[0][:, :, 0], out[1][:, :, 0], out[2][:, :, 0], out[3][:, :, 0]
    else:
        raise ValueError(f"backend must be 'torch' or 'onnx', got {backend!r}")


def extract(m5: pd.DataFrame, anchors: np.ndarray, symbol: str = "GOLD",
            batch_size: int = 512, backend: str = "onnx") -> np.ndarray:
    """
    Return float32[len(anchors), N_TSPULSE_FEATURES]. Result is cached
    per (symbol, anchor-hash, backend) so torch and onnx caches don't
    collide.

    Each anchor i uses ONLY past closes m5[i-511 : i+1] - strictly causal.
    Default backend is 'onnx' so training matches what the live EA sees.
    """
    cp = _cache_path(symbol, anchors, backend=backend)
    if cp.exists():
        log.info("[tspulse] cache hit (%s): %s", backend, cp.name)
        return np.load(cp)

    c = m5["close"].to_numpy(np.float64)
    n_total = len(c)
    n_anchors = len(anchors)
    if (anchors.min() < _CTX - 1):
        raise ValueError(
            f"first anchor {anchors.min()} < ctx-1 ({_CTX-1}); need 512 prior bars")
    if anchors.max() >= n_total:
        raise ValueError("anchor exceeds M5 length")

    out = np.zeros((n_anchors, N_TSPULSE_FEATURES), dtype=np.float32)
    log.info("[tspulse] extracting %d anchors in batches of %d (backend=%s) ...",
             n_anchors, batch_size, backend)
    eps = 1e-12
    for b0 in range(0, n_anchors, batch_size):
        b1 = min(b0 + batch_size, n_anchors)
        idx = anchors[b0:b1]
        X = np.stack(
            [c[i - _CTX + 1 : i + 1] for i in idx], axis=0
        ).astype(np.float32)[:, :, None]
        fcst, recon, obs_fft, prd_fft = _forward_batch(X, backend)
        last_close = X[:, -1, 0]
        last_fcst = fcst[:, -1]
        net_logret = np.log(np.clip(last_fcst, eps, None)
                            / np.clip(last_close, eps, None))
        d = np.diff(X[:, :, 0], axis=1)
        atr_proxy = np.std(d, axis=1) + eps
        fwd_mag = np.abs(last_fcst - last_close) / atr_proxy
        recon_err = ((recon - X[:, :, 0]) ** 2).mean(axis=1)
        kl = (obs_fft * (np.log(obs_fft + eps) - np.log(prd_fft + eps))
              ).sum(axis=1)
        out[b0:b1, 0] = net_logret
        out[b0:b1, 1] = fwd_mag
        out[b0:b1, 2] = recon_err
        out[b0:b1, 3] = kl
        if (b0 // batch_size) % 10 == 0:
            log.info("[tspulse]  %d / %d", b1, n_anchors)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(cp, out)
    log.info("[tspulse] cached -> %s", cp.name)
    return out
