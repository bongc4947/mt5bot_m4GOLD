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

_MODEL = None
_CTX = 512               # tspulse context length
_REPO = "ibm-granite/granite-timeseries-tspulse-r1"
TSPULSE_FEATURES = [
    "tspulse_fwd_logret",
    "tspulse_fwd_mag",
    "tspulse_recon_err",
    "tspulse_fft_div",
]
N_TSPULSE_FEATURES = len(TSPULSE_FEATURES)


def _get_model():
    global _MODEL
    if _MODEL is None:
        import warnings
        warnings.filterwarnings("ignore")
        from tsfm_public.models.tspulse import TSPulseForReconstruction
        log.info("[tspulse] loading %s ...", _REPO)
        _MODEL = TSPulseForReconstruction.from_pretrained(_REPO)
        _MODEL.eval()
        log.info("[tspulse] loaded — %d params, ctx=%d",
                 sum(p.numel() for p in _MODEL.parameters()),
                 _MODEL.config.context_length)
    return _MODEL


def _cache_path(symbol: str, anchors: np.ndarray) -> Path:
    h = hashlib.sha1(anchors.astype(np.int64).tobytes()).hexdigest()[:16]
    base = Path(__file__).parent.parent.parent / "onnx_out"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"tspulse_features_{symbol}_{len(anchors)}_{h}.npy"


def extract(m5: pd.DataFrame, anchors: np.ndarray, symbol: str = "GOLD",
            batch_size: int = 512) -> np.ndarray:
    """
    Return float32[len(anchors), N_TSPULSE_FEATURES]. Result is cached.

    Each anchor i uses ONLY past closes m5[i-511 : i+1] — strictly causal.
    """
    cp = _cache_path(symbol, anchors)
    if cp.exists():
        log.info("[tspulse] cache hit: %s", cp.name)
        return np.load(cp)

    c = m5["close"].to_numpy(np.float64)
    n_total = len(c)
    n_anchors = len(anchors)
    if (anchors.min() < _CTX - 1):
        raise ValueError(
            f"first anchor {anchors.min()} < ctx-1 ({_CTX-1}); need 512 prior bars")
    if anchors.max() >= n_total:
        raise ValueError("anchor exceeds M5 length")

    model = _get_model()
    out = np.zeros((n_anchors, N_TSPULSE_FEATURES), dtype=np.float32)

    log.info("[tspulse] extracting %d anchors in batches of %d ...",
             n_anchors, batch_size)
    eps = 1e-12
    with torch.no_grad():
        for b0 in range(0, n_anchors, batch_size):
            b1 = min(b0 + batch_size, n_anchors)
            idx = anchors[b0:b1]
            # build [B, ctx, 1] windows of CLOSES ending AT the anchor
            X = np.stack(
                [c[i - _CTX + 1 : i + 1] for i in idx], axis=0
            ).astype(np.float32)[:, :, None]
            past = torch.from_numpy(X)
            res = model(past_values=past)
            fcst = res["forecast_output"].cpu().numpy()[:, :, 0]   # [B, 16]
            recon = res["reconstruction_outputs"].cpu().numpy()[:, :, 0]
            obs_fft = res["original_fft_softmax"].cpu().numpy()[:, :, 0]
            prd_fft = res["fft_softmax_preds"].cpu().numpy()[:, :, 0]
            last_close = X[:, -1, 0]
            last_fcst = fcst[:, -1]
            net_logret = np.log(np.clip(last_fcst, eps, None)
                                / np.clip(last_close, eps, None))
            # rough ATR proxy for normalisation — std of 1-bar diffs in window
            d = np.diff(X[:, :, 0], axis=1)
            atr_proxy = np.std(d, axis=1) + eps
            fwd_mag = np.abs(last_fcst - last_close) / atr_proxy
            recon_err = ((recon - X[:, :, 0]) ** 2).mean(axis=1)
            # KL(obs || pred) on the FFT softmax — regime divergence
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
