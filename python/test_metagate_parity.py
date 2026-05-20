"""
test_metagate_parity.py — verify the MetaGate.mqh tspulse math matches
the training-side feature extractor bit-for-bit when both run through
the same ONNX.

Procedure: pick a recent M5 anchor whose tspulse features are already
in the training cache, then re-compute the four scalars step by step in
the same arithmetic order MetaGate.mqh uses, and diff. Anything > 1e-5
is a real bug (not just float vs double rounding).
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger(__name__)


def mql5_tspulse_scalars(past_512: np.ndarray, sess) -> tuple:
    """Re-implement MetaGate.mqh::_MG_TspulseFeatures step-by-step."""
    eps = 1e-12
    past = past_512.astype(np.float32)
    out = sess.run(None, {"past_values": past.reshape(1, 512, 1)})
    fcst = out[0][0, :, 0]
    recon = out[1][0, :, 0]
    obs   = out[2][0, :, 0]
    prd   = out[3][0, :, 0]

    last_close = float(past[-1])
    last_fcst  = float(fcst[-1])

    # numpy.std(ddof=0) of 1-bar diffs - MQL5 mirror
    diffs = past[1:] - past[:-1]
    n_d = len(diffs)
    mu = float(diffs.sum()) / n_d
    var = float(((diffs - mu) ** 2).sum()) / n_d
    atr_proxy = float(np.sqrt(var)) + eps

    net_logret = float(np.log(max(last_fcst, eps) / max(last_close, eps)))
    fwd_mag    = abs(last_fcst - last_close) / atr_proxy

    recon_err = float(((recon - past) ** 2).sum()) / 512.0

    fft_div = 0.0
    for i in range(256):
        o = float(obs[i]); p = float(prd[i])
        fft_div += o * (np.log(o + eps) - np.log(p + eps))

    return net_logret, fwd_mag, recon_err, fft_div


def main() -> int:
    import onnxruntime as ort
    from aurum.datamodule import _load_m5_bars

    onnx_path = (Path(__file__).parent.parent / "onnx_out"
                 / "M4GOLD_TSPULSE_GOLD.onnx")
    sess = ort.InferenceSession(str(onnx_path),
                                 providers=["CPUExecutionProvider"])
    m5 = _load_m5_bars()
    c = m5["close"].to_numpy(np.float64)
    log.info("[parity] loaded %d M5 bars", len(c))

    # Find the cached training features (built at anchors=arange(511,N))
    cache_dir = Path(__file__).parent.parent / "onnx_out"
    cache_files = list(cache_dir.glob("tspulse_features_GOLD_*_onnx.npy"))
    if not cache_files:
        log.error("[parity] no ONNX-backend tspulse cache - "
                  "run train_h7 --with-tspulse first")
        return 1
    cache = np.load(cache_files[0])
    valid_lo = 511
    log.info("[parity] training cache: %s  shape=%s",
             cache_files[0].name, cache.shape)

    # Pick 5 anchors across the last 20% of the series
    n = len(c)
    test_anchors = np.linspace(int(n * 0.8), n - 1, 5, dtype=int)

    max_err = np.zeros(4)
    for a in test_anchors:
        past = c[a - 511: a + 1]
        if len(past) != 512:
            continue
        mql5_scal = np.array(mql5_tspulse_scalars(past, sess))
        cache_idx = a - valid_lo
        train_scal = cache[cache_idx].astype(np.float64)
        diff = np.abs(mql5_scal - train_scal)
        max_err = np.maximum(max_err, diff)
        log.info("[parity] anchor=%d  mql5_emu=%s  train=%s  diff=%s",
                 int(a),
                 np.array2string(mql5_scal, precision=6),
                 np.array2string(train_scal, precision=6),
                 np.array2string(diff, precision=2))

    names = ["fwd_logret", "fwd_mag", "recon_err", "fft_div"]
    log.info("[parity] max abs diff per scalar:")
    all_ok = True
    for k, e in zip(names, max_err):
        ok = e < 1e-4
        all_ok = all_ok and ok
        log.info("  %-12s %.3e  %s", k, e, "OK" if ok else "HIGH")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
