"""
export_tspulse_onnx.py — export IBM Granite TSPulse-r1 to ONNX so the
live MT5 EA can compute the 4 tspulse meta-gate features without needing
a PyTorch runtime.

The PyTorch model is a heavy multi-output transformer (~21 named tensors
out of forward). For deployment we wrap it in a *minimal* nn.Module that
returns only the four tensors the EA actually needs:

    forecast       float32[B, 16, 1]
    reconstruction float32[B, 512, 1]
    obs_fft        float32[B, 256, 1]
    pred_fft       float32[B, 256, 1]

That keeps the ONNX file small, deterministic, and trivial to run on
ONNX Runtime under MT5's OnnxRun. The 4 scalars themselves are computed
in MetaGate.mqh from those tensors — same math as
aurum/tspulse_features.py so train/serve parity holds.

Run:
    python python/export_tspulse_onnx.py

Writes:
    onnx_out/M4GOLD_TSPULSE_GOLD.onnx
    + a parity report against the PyTorch model (max-err on 8 random
      inputs across all 4 outputs)
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger(__name__)

_REPO = "ibm-granite/granite-timeseries-tspulse-r1"
_OUT = Path(__file__).parent.parent / "onnx_out" / "M4GOLD_TSPULSE_GOLD.onnx"


def _disable_random_masking(base):
    """
    tspulse-r1's reconstruction-head architecture applies RANDOM time and
    FFT masking *at every forward call, even in eval mode* (it is part of
    the self-supervised masked-reconstruction design, not training-only
    dropout). That makes the raw model non-deterministic — running it
    twice on the same input gives different outputs by ~$1 on a $2000
    GOLD price.

    For deployment we need bit-stable features. Replace both maskers with
    pass-through. The cost: `recon_err` becomes "how well does the model
    reconstruct a fully-visible input" rather than "how well from a 70%-
    masked view". Whether that variant still lifts PF is an empirical
    question we re-test by retraining the meta-gate on patched features.
    """
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


class TSPulseExportWrapper(nn.Module):
    """Strip the multi-output module down to the 4 tensors the EA needs."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, past_values: torch.Tensor):
        out = self.base(past_values=past_values, return_loss=False)
        return (
            out.forecast_output,            # [B, 16,  1]
            out.reconstruction_outputs,     # [B, 512, 1]
            out.original_fft_softmax,       # [B, 256, 1]
            out.fft_softmax_preds,          # [B, 256, 1]
        )


def main() -> int:
    from tsfm_public.models.tspulse import TSPulseForReconstruction
    log.info("[export] loading %s ...", _REPO)
    base = TSPulseForReconstruction.from_pretrained(_REPO)
    base.eval()
    _disable_random_masking(base)
    log.info("[export] random maskers patched out -> deterministic forward")
    wrapper = TSPulseExportWrapper(base).eval()
    _OUT.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input — batch=1 is what MT5 will use; opset 17 has all the
    # tspulse ops (FFT, layernorm, gather).
    dummy = torch.zeros(1, 512, 1, dtype=torch.float32)
    log.info("[export] tracing -> %s ...", _OUT.name)
    # The legacy TorchScript exporter can't lower aten::fft_rfft at
    # opset 17 — tspulse's FFT branch needs the dynamo path. Dynamo's
    # tensor-level absolute error is loose on price-scale outputs (~1e-3
    # relative) but the FOUR SCALARS the EA actually computes downstream
    # are tight; the parity check below verifies that explicitly.
    torch.onnx.export(
        wrapper, (dummy,), str(_OUT),
        input_names=["past_values"],
        output_names=["forecast", "reconstruction", "obs_fft", "pred_fft"],
        dynamic_axes={"past_values": {0: "batch"},
                      "forecast": {0: "batch"},
                      "reconstruction": {0: "batch"},
                      "obs_fft": {0: "batch"},
                      "pred_fft": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    log.info("[export] wrote %s (%.1f MB)", _OUT.name,
             _OUT.stat().st_size / 1e6)

    # ---- parity check vs the PyTorch model ----
    # We care about parity on the 4 SCALARS the EA will use, not on the
    # raw tensors. Compute both end-to-end and compare — using REAL GOLD
    # M5 windows so atr_proxy reflects actual deployment scale.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(_OUT), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    eps = 1e-12

    sys.path.insert(0, str(Path(__file__).parent))
    from aurum.datamodule import _load_m5_bars
    m5 = _load_m5_bars()
    c_all = m5["close"].to_numpy(np.float64)
    # 8 random non-overlapping 512-bar windows from the back half
    anchors = rng.choice(np.arange(len(c_all) // 2, len(c_all)),
                          size=8, replace=False)

    tensor_max = {"forecast": 0.0, "reconstruction": 0.0,
                  "obs_fft": 0.0, "pred_fft": 0.0}
    scalar_max = {"fwd_logret": 0.0, "fwd_mag": 0.0,
                  "recon_err": 0.0, "fft_div": 0.0}
    for a in anchors:
        x = c_all[a - 511: a + 1].astype(np.float32).reshape(1, 512, 1)
        with torch.no_grad():
            tf = [t.cpu().numpy() for t in wrapper(torch.from_numpy(x))]
        of = sess.run(None, {"past_values": x})
        for k, t, o in zip(tensor_max.keys(), tf, of):
            tensor_max[k] = max(tensor_max[k], float(np.abs(t - o).max()))
        for tag, outs in (("torch", tf), ("onnx", of)):
            last_close = float(x[0, -1, 0])
            last_fcst = float(outs[0][0, -1, 0])
            d = np.diff(x[0, :, 0]); atr_proxy = float(np.std(d)) + eps
            fwd_logret = float(np.log(max(last_fcst, eps) /
                                       max(last_close, eps)))
            fwd_mag = abs(last_fcst - last_close) / atr_proxy
            recon = outs[1][0, :, 0]
            recon_err = float(((recon - x[0, :, 0]) ** 2).mean())
            obs = outs[2][0, :, 0]; prd = outs[3][0, :, 0]
            fft_div = float((obs * (np.log(obs + eps) - np.log(prd + eps))).sum())
            scalars = (fwd_logret, fwd_mag, recon_err, fft_div)
            if tag == "torch":
                t_sc = scalars
            else:
                for k, ts, os_ in zip(scalar_max.keys(), t_sc, scalars):
                    scalar_max[k] = max(scalar_max[k], abs(ts - os_))
    log.info("[export] tensor max-err (absolute, price-scale):")
    for k, e in tensor_max.items():
        log.info("  %-15s %.3e", k, e)
    log.info("[export] SCALAR max-err (the 4 EA features) — this is what matters:")
    all_ok = True
    for k, e in scalar_max.items():
        ok = e < 1e-3
        all_ok = all_ok and ok
        log.info("  %-12s %.3e %s", k, e, "OK" if ok else "HIGH")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
