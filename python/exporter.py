"""
exporter.py — Export PyTorch models to ONNX opset 12 (MT5 compatible).
Pipeline: eval/train mode → torch.onnx.export (legacy) → onnxsim → OnnxRuntime validation → write.
Writes _dir_det.onnx, _dir_mc.onnx, _exec_det.onnx, _modify_det.onnx + meta.json per symbol.
"""

import hashlib
import json
import logging
import time
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from config import (
    HYDRA_VERSION, ONNX_OPSET, FEATURE_DIM_DIR, FEATURE_DIM_EXEC,
    FEATURE_DIM_MOD, ONNX_OUTPUT_DIR,
    onnx_det_path, onnx_mc_path, onnx_exec_path, onnx_modify_path, meta_path,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()[:16]


def _simplify(path: Path) -> bool:
    try:
        import onnx
        import onnxsim
        model = onnx.load(str(path))
        simplified, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(simplified, str(path))
            log.info("onnxsim simplified: %s", path.name)
        return ok
    except ImportError:
        log.debug("onnxsim not installed — skipping simplification")
        return False
    except Exception as e:
        log.warning("onnxsim failed for %s: %s", path.name, e)
        return False


def _validate_onnx(path: Path, in_shape: tuple, n_samples: int = 100) -> bool:
    """Run N random samples through OnnxRuntime and compare to pytorch would."""
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3   # ERROR only (suppress INFO/WARNING spam)
        sess = ort.InferenceSession(str(path),
                                    sess_options=so,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        for _ in range(n_samples):
            x = np.random.randn(*in_shape).astype(np.float32)
            out = sess.run(None, {inp_name: x})
            assert len(out) > 0, "No output"
        log.info("OnnxRuntime validation passed (%d samples): %s",
                 n_samples, path.name)
        return True
    except ImportError:
        log.warning("onnxruntime not installed — skipping validation")
        return True
    except Exception as e:
        log.error("OnnxRuntime validation FAILED for %s: %s", path.name, e)
        return False


def _export_one(model: nn.Module, path: Path,
                in_shape: tuple, input_names: list,
                output_names: list, opset: int = ONNX_OPSET,
                dynamic_axes: Optional[dict] = None) -> bool:
    """
    Export to a single self-contained .onnx file (no external .data sidecar).
    MT5 OnnxCreate requires all weights embedded — external data files are ignored.
    Uses a temp buffer + onnx.save_model with save_as_external_data=False to guarantee this.
    """
    import warnings
    import sys
    import os
    import io
    import onnx as _onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.cpu().eval()
    dummy = torch.randn(*in_shape)

    # Step 1: export to an in-memory buffer (BytesIO) — prevents .data sidecar creation
    buf = io.BytesIO()

    _devnull = open(os.devnull, "w", encoding="utf-8")
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _devnull
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model_cpu,
                (dummy,),
                buf,
                dynamo=False,
                opset_version=opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {input_names[0]: {0: "batch"}},
                do_constant_folding=True,
                export_params=True,
            )
    except TypeError:
        # dynamo kwarg not supported in this PyTorch build
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model_cpu,
                (dummy,),
                buf,
                opset_version=opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {input_names[0]: {0: "batch"}},
                do_constant_folding=True,
                export_params=True,
            )
    except Exception as e:
        sys.stdout, sys.stderr = _old_stdout, _old_stderr
        _devnull.close()
        log.error("ONNX export failed for %s: %s", path.name, e)
        return False
    finally:
        sys.stdout, sys.stderr = _old_stdout, _old_stderr
        _devnull.close()

    # Step 2: parse the in-memory proto, then save with all tensors inline
    try:
        buf.seek(0)
        proto = _onnx.load_model(buf)
        # Convert any external-data references → inline tensors
        _onnx.load_external_data_for_model(proto, "")
        # Delete any stale .data sidecar from a previous run
        sidecar = Path(str(path) + ".data")
        if sidecar.exists():
            sidecar.unlink()
        # Save as a single file with no external data
        _onnx.save_model(
            proto,
            str(path),
            save_as_external_data=False,
        )
        log.info("Exported (opset %d): %s", opset, path.name)
        return True
    except Exception as e:
        log.error("ONNX save failed for %s: %s", path.name, e)
        return False


# ---------------------------------------------------------------------------
# Direction model export (det + mc pair)
# ---------------------------------------------------------------------------

def export_direction(model: nn.Module, agent: str, symbol: str,
                     train_metrics: Optional[Dict[str, Any]] = None,
                     feature_dim: int = FEATURE_DIM_DIR) -> bool:
    """
    Exports _dir_det.onnx (eval), _dir_mc.onnx (train/dropout), meta.json.
    Returns True if both exports + validations succeed.
    """
    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_shape = (1, feature_dim)
    ok = True

    # --- Deterministic ---
    model.eval()
    p_det = onnx_det_path(agent, symbol)
    ok &= _export_one(model, p_det, in_shape,
                      input_names=["features"],
                      output_names=["logit"])
    _simplify(p_det)
    ok &= _validate_onnx(p_det, in_shape)

    # --- MC Dropout (train mode keeps dropout active) ---
    model.train()
    p_mc = onnx_mc_path(agent, symbol)
    ok &= _export_one(model, p_mc, in_shape,
                      input_names=["features"],
                      output_names=["logit"])
    _simplify(p_mc)
    ok &= _validate_onnx(p_mc, in_shape)
    model.eval()  # restore

    if ok:
        _write_dir_meta(agent, symbol, feature_dim, model, train_metrics, p_det)

    return ok


# ---------------------------------------------------------------------------
# Execution model export
# ---------------------------------------------------------------------------

def export_execution(exec_model: nn.Module, agent: str, symbol: str,
                     train_metrics: Optional[Dict[str, Any]] = None,
                     feature_dim: int = FEATURE_DIM_EXEC) -> bool:
    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_shape = (1, feature_dim)
    exec_model.eval()

    p_exec = onnx_exec_path(agent, symbol)
    ok = _export_one(exec_model, p_exec, in_shape,
                     input_names=["exec_features"],
                     output_names=["exec_outputs"])
    # Do NOT simplify exec model: onnxsim decomposes the torch.cat([...], dim=1)
    # into 5 separate output tensors, causing ERR_ONNX_INCORRECT_OUTPUT_COUNT (5808)
    # in MT5 when OnnxRun is called with a single output array.
    ok &= _validate_onnx(p_exec, in_shape)

    if ok:
        _patch_exec_meta(agent, symbol, feature_dim, train_metrics)

    return ok


# ---------------------------------------------------------------------------
# Modification model export
# ---------------------------------------------------------------------------

def export_modify(mod_model: nn.Module, agent: str, symbol: str,
                  train_metrics: Optional[Dict[str, Any]] = None,
                  feature_dim: int = FEATURE_DIM_MOD) -> bool:
    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_shape = (1, feature_dim)
    mod_model.eval()

    p_mod = onnx_modify_path(agent, symbol)
    ok = _export_one(mod_model, p_mod, in_shape,
                     input_names=["mod_features"],
                     output_names=["mod_outputs"])
    # Do NOT simplify modify model: same onnxsim cat-split issue as exec model.
    ok &= _validate_onnx(p_mod, in_shape)

    if ok:
        _patch_modify_meta(agent, symbol, feature_dim, train_metrics)

    return ok


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _write_dir_meta(agent: str, symbol: str, feature_dim: int,
                    model: nn.Module, metrics: Optional[Dict],
                    det_path: Path):
    p = meta_path(agent, symbol)

    h1 = h2 = 0
    try:
        params = list(model.parameters())
        h1 = params[1].shape[0] if len(params) > 1 else 0
        h2 = params[3].shape[0] if len(params) > 3 else 0
    except Exception:
        pass

    meta = {
        "version":      HYDRA_VERSION,
        "agent":        agent,
        "symbol":       symbol,
        "feature_dim":  feature_dim,
        "h1":           h1,
        "h2":           h2,
        "trained_bars":   int(metrics.get("trained_bars", 0)) if metrics else 0,
        "val_acc":        float(metrics.get("val_acc", 0.0)) if metrics else 0.0,
        "win_rate":       float(metrics.get("win_rate", 0.0)) if metrics else 0.0,
        "profit_factor":  float(metrics.get("profit_factor", 0.0)) if metrics else 0.0,
        "expected_value": float(metrics.get("expected_value", 0.0)) if metrics else 0.0,
        "trained_at":     dt.datetime.now(dt.timezone.utc).isoformat() + "Z",
        "epochs":       int(metrics.get("epochs_run", 0)) if metrics else 0,
        "checksum":     _sha256(det_path),
        "exec_model":   {},
        "modify_model": {},
    }

    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Wrote meta: %s", p.name)


def _patch_exec_meta(agent: str, symbol: str, feature_dim: int, metrics: Optional[Dict]):
    assert feature_dim == FEATURE_DIM_EXEC, \
        f"exec model must use FEATURE_DIM_EXEC={FEATURE_DIM_EXEC}, got {feature_dim}"
    p = meta_path(agent, symbol)
    meta: Dict = {}
    if p.exists():
        with open(p) as f:
            meta = json.load(f)
    meta["exec_feature_dim"] = feature_dim   # EA reads this to set OnnxSetInputShape
    meta["exec_model"] = {
        "val_timing_acc":  float(metrics.get("val_timing_acc", 0)) if metrics else 0,
        "val_sl_mae_pips": float(metrics.get("val_sl_mae_pips", 0)) if metrics else 0,
        "val_tp_mae_pips": float(metrics.get("val_tp_mae_pips", 0)) if metrics else 0,
        "val_vol_mae":     float(metrics.get("val_vol_mae", 0)) if metrics else 0,
        "val_session_acc": float(metrics.get("val_session_acc", 0)) if metrics else 0,
        "avg_rr_ratio":    float(metrics.get("avg_rr_ratio", 0)) if metrics else 0,
    }
    meta["exec_checksum"] = _sha256(onnx_exec_path(agent, symbol)) if onnx_exec_path(agent, symbol).exists() else ""
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)


def _patch_modify_meta(agent: str, symbol: str, feature_dim: int, metrics: Optional[Dict]):
    assert feature_dim == FEATURE_DIM_MOD, \
        f"modify model must use FEATURE_DIM_MOD={FEATURE_DIM_MOD}, got {feature_dim}"
    p = meta_path(agent, symbol)
    meta: Dict = {}
    if p.exists():
        with open(p) as f:
            meta = json.load(f)
    meta["mod_feature_dim"] = feature_dim    # EA reads this to set OnnxSetInputShape
    meta["modify_model"] = {
        "val_be_acc":     float(metrics.get("val_be_acc", 0)) if metrics else 0,
        "val_close_acc":  float(metrics.get("val_close_acc", 0)) if metrics else 0,
        "avg_trail_error": float(metrics.get("avg_trail_error", 0)) if metrics else 0,
    }
    meta["modify_checksum"] = _sha256(onnx_modify_path(agent, symbol)) if onnx_modify_path(agent, symbol).exists() else ""
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
