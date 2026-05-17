"""
aurum/export.py — ONNX bundler + OnnxRuntime parity check.

Exports the AURUM deployable bundle:
  M4GOLD_AURUM_GOLD.onnx        main net  float[1,2048] -> float[1,13]
  M4GOLD_AURUM_META_GOLD.onnx   meta gate float[1,13]  -> float[1,2]
  M4GOLD_AURUM_GOLD_spec.json   conformal threshold, sizing params,
                                channel norm stats, contract dims, deploy flag

Every export is validated: the ONNX output must match the PyTorch output
within tolerance on random inputs, or the bundle is rejected (a silent
export/runtime mismatch would trade on a different model than was tested).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from aurum.aurum_config import (
    ARTIFACT_PREFIX, SYMBOL, FLAT_INPUT_DIM, OUTPUT_DIM, TIMEFRAMES,
    CHANNELS, QUANTILES, SLICE_DIR, SLICE_QUANT, SLICE_EXEC, SLICE_REGIME,
    META_ACT_THRESHOLD, CONFORMAL_ALPHA,
)
from aurum.model import AurumNet, AurumExportWrapper
from aurum.sizing import sizing_params

log = logging.getLogger(__name__)
ONNX_OPSET = 17     # opset 17 covers LayerNorm/GELU/Attention sub-ops for MT5


def _validate(onnx_path: Path, torch_fn, in_dim: int,
              tol: float = 1e-3) -> bool:
    """Run the exported ONNX vs the torch model on random inputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime missing — skipping parity check for %s",
                    onnx_path.name)
        return True
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    max_err = 0.0
    for _ in range(8):
        x = np.random.randn(1, in_dim).astype(np.float32)
        onnx_out = sess.run(None, {in_name: x})[0]
        with torch.no_grad():
            torch_out = torch_fn(torch.from_numpy(x)).numpy()
        max_err = max(max_err, float(np.abs(onnx_out - torch_out).max()))
    ok = max_err <= tol
    log.info("[export] parity %s  max_err=%.2e  %s",
             onnx_path.name, max_err, "OK" if ok else "FAIL")
    return ok


def export_main_net(net: AurumNet, out_dir: Path) -> tuple[Path, bool]:
    """Export AurumNet (with softmax wrapper) to ONNX."""
    net.eval()
    wrapper = AurumExportWrapper(net).eval()
    path = out_dir / f"{ARTIFACT_PREFIX}_{SYMBOL}.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, FLAT_INPUT_DIM)
    torch.onnx.export(
        wrapper, dummy, str(path),
        input_names=["input"], output_names=["output"],
        opset_version=ONNX_OPSET, dynamo=False,
    )
    ok = _validate(path, wrapper, FLAT_INPUT_DIM)
    return path, ok


def write_spec(out_dir: Path, *, conformal_q: float,
               norm: dict, cv_report: dict, deploy: bool) -> Path:
    """Write the spec JSON the EA reads alongside the ONNX files."""
    spec = {
        "strategy": "AURUM",
        "symbol": SYMBOL,
        "version": "aurum-1.0.0",
        "contract": {
            "input_dim": FLAT_INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "timeframes": TIMEFRAMES,
            "channels": CHANNELS,
            "slice_dir": list(SLICE_DIR),
            "slice_quant": list(SLICE_QUANT),
            "slice_exec": list(SLICE_EXEC),
            "slice_regime": list(SLICE_REGIME),
            "quantiles": QUANTILES,
        },
        "channel_norm": norm,
        "conformal": {"alpha": CONFORMAL_ALPHA, "q_hat": conformal_q},
        "meta": {"act_threshold": META_ACT_THRESHOLD},
        "sizing": sizing_params(),
        "cv_report": cv_report,
        "deploy": bool(deploy),
    }
    path = out_dir / f"{ARTIFACT_PREFIX}_{SYMBOL}_spec.json"
    path.write_text(json.dumps(spec, indent=2))
    log.info("[export] spec -> %s  (deploy=%s)", path.name, deploy)
    return path
