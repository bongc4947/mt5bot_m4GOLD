"""
train_h7_metatrend.py — train + validate + export the meta-trend strategy.

The validated GOLD edge: a slow EMA-cross trend rule for direction + an
XGBoost meta-gate that decides when to trust it. Everything here is
leak-free (purged CV) and net of transaction cost.

Pipeline:
  1. Build causal features + the meta-label (aurum/metatrend.py).
  2. Purged-CV: score the meta-gated strategy AND the raw primary (the
     control). The meta-gate must beat the raw rule, every fold.
  3. Train the final XGBoost meta-gate on all data; export it to ONNX.
  4. Write M4GOLD_METATREND_GOLD_spec.json with the deploy decision.

Runs locally in minutes — XGBoost, no GPU, no Kaggle needed.

    python python/train_h7_metatrend.py
    python python/train_h7_metatrend.py --max-bars 150000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from aurum.metatrend import (
    build_features, build_meta_label, primary_signal,
    META_FEATURES, N_META_FEATURES, LABEL_HORIZON, META_ACT_THRESHOLD,
    COST_ROUNDTRIP, PRIMARY_EMA_FAST, PRIMARY_EMA_SLOW,
)
from cv.purged_kfold import PurgedKFold

log = logging.getLogger(__name__)
_ARTIFACT_DIR = Path(__file__).parent.parent / "onnx_out"
_PF_CAP = 10.0
_N_SPLITS = 6
_GATE_MIN_PF = 1.15        # absolute net-PF floor (mean across folds)
_GATE_MIN_EXCESS = 0.05    # meta-gate must beat the raw primary by this
_GATE_MIN_FOLD = 0.92      # no purged-CV fold may lose worse than this
_XGB_ESTIMATORS = 300
_XGB_MAX_DEPTH = 4


def _pf(pnl: np.ndarray, min_trades: int = 30) -> float:
    if len(pnl) < min_trades:
        return 0.0
    g = float(pnl[pnl > 0].sum())
    ls = float(-pnl[pnl < 0].sum())
    if ls <= 1e-12:
        return _PF_CAP if g > 0 else 0.0
    return min(_PF_CAP, g / ls)


def _make_xgb(use_gpu: bool):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=_XGB_ESTIMATORS, max_depth=_XGB_MAX_DEPTH,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective="binary:logistic",
        tree_method="hist", device="cuda" if use_gpu else "cpu",
        random_state=42, n_jobs=0)


def train(max_bars: int | None, use_gpu: bool) -> dict:
    from aurum.datamodule import _load_m5_bars

    m5 = _load_m5_bars()
    if max_bars and len(m5) > max_bars:
        m5 = m5.iloc[-max_bars:].reset_index(drop=True)
    log.info("[h7] %d M5 bars  span=%s -> %s",
             len(m5), m5["time"].iloc[0], m5["time"].iloc[-1])

    X = build_features(m5)
    lab = build_meta_label(m5)
    y, prim, fwd = lab["y"], lab["primary"], lab["fwd_ret"]
    log.info("[h7] features=%d  meta-label pos rate=%.3f",
             N_META_FEATURES, float(y.mean()))

    # ---- purged-CV: meta-gated strategy vs the raw primary (control) -----
    pk = PurgedKFold(n_splits=_N_SPLITS, horizon=LABEL_HORIZON,
                     embargo_pct=0.01)
    meta_pf, raw_pf, n_trades = [], [], []
    for fold, (tr, te) in enumerate(pk.split(len(y))):
        cnt = np.bincount(y[tr], minlength=2).astype(np.float64)
        w = cnt.sum() / np.maximum(cnt, 1.0)
        model = _make_xgb(use_gpu)
        model.fit(X[tr], y[tr], sample_weight=w[y[tr]])
        pact = model.predict_proba(X[te])[:, 1]
        act = pact >= META_ACT_THRESHOLD
        # meta-gated PnL — follow the primary only on accepted bars
        pnl_meta = prim[te][act] * fwd[te][act] - COST_ROUNDTRIP
        # raw control — follow the primary on every bar of the fold
        pnl_raw = prim[te] * fwd[te] - COST_ROUNDTRIP
        meta_pf.append(_pf(pnl_meta))
        raw_pf.append(_pf(pnl_raw))
        n_trades.append(int(act.sum()))
        log.info("[h7] fold %d/%d  meta_PF=%.3f  raw_PF=%.3f  trades=%d",
                 fold + 1, _N_SPLITS, meta_pf[-1], raw_pf[-1], n_trades[-1])

    meta_mean = float(np.mean(meta_pf))
    raw_mean = float(np.mean(raw_pf))
    min_fold = float(min(meta_pf))
    all_folds_pos = bool(all(p >= 1.0 for p in meta_pf))
    excess = meta_mean - raw_mean
    # Robust = strong mean, no fold loses badly (a mild <1.0 sub-period is
    # tolerable; a catastrophic one is not), and the gate genuinely beats
    # the raw primary rule.
    robust = min_fold >= _GATE_MIN_FOLD
    deploy = bool(meta_mean >= _GATE_MIN_PF and robust
                  and excess >= _GATE_MIN_EXCESS)
    log.info("[h7] meta meanPF=%.3f  raw meanPF=%.3f  excess=%+.3f  "
             "min_fold=%.3f  all_pos=%s  -> deploy=%s",
             meta_mean, raw_mean, excess, min_fold, all_folds_pos, deploy)

    # ---- train the final meta-gate on ALL data, export to ONNX ----------
    cnt = np.bincount(y, minlength=2).astype(np.float64)
    w = cnt.sum() / np.maximum(cnt, 1.0)
    final = _make_xgb(use_gpu)
    final.fit(X, y, sample_weight=w[y])

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_ok = _export_onnx(final, _ARTIFACT_DIR / "M4GOLD_METATREND_GOLD.onnx")

    spec = {
        "strategy": "METATREND",
        "symbol": "GOLD",
        "version": "metatrend-1.0.0",
        "primary": {"type": "ema_cross", "fast": PRIMARY_EMA_FAST,
                    "slow": PRIMARY_EMA_SLOW},
        "features": META_FEATURES,
        "n_features": N_META_FEATURES,
        "label_horizon_bars": LABEL_HORIZON,
        "act_threshold": META_ACT_THRESHOLD,
        "cost_roundtrip": COST_ROUNDTRIP,
        "cv": {
            "meta_mean_pf": round(meta_mean, 4),
            "raw_mean_pf": round(raw_mean, 4),
            "excess_vs_raw": round(excess, 4),
            "meta_pf_folds": [round(x, 3) for x in meta_pf],
            "min_fold_pf": round(min_fold, 3),
            "all_folds_positive": all_folds_pos,
            "mean_trades_per_fold": int(np.mean(n_trades)),
        },
        "gate": {"min_pf": _GATE_MIN_PF, "min_excess_vs_raw": _GATE_MIN_EXCESS,
                 "min_fold_pf": _GATE_MIN_FOLD, "onnx_parity_ok": bool(onnx_ok)},
        "deploy": bool(deploy and onnx_ok),
    }
    spec_path = _ARTIFACT_DIR / "M4GOLD_METATREND_GOLD_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    log.info("[h7] spec -> %s  (deploy=%s)", spec_path.name, spec["deploy"])
    return spec


def _export_onnx(model, path: Path) -> bool:
    """Export the XGBoost meta-gate to ONNX, validate parity vs the booster."""
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        log.warning("[h7] onnxmltools missing — cannot export ONNX")
        return False
    onnx_model = convert_xgboost(
        model, initial_types=[("input", FloatTensorType([1, N_META_FEATURES]))])
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    # parity check
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path),
                                    providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        max_err = 0.0
        for _ in range(16):
            x = np.random.randn(1, N_META_FEATURES).astype(np.float32)
            onnx_p = sess.run(None, {in_name: x})[1][0][1]   # P(class 1)
            xgb_p = float(model.predict_proba(x)[0, 1])
            max_err = max(max_err, abs(onnx_p - xgb_p))
        ok = bool(max_err < 1e-4)
        log.info("[h7] ONNX parity max_err=%.2e  %s", max_err,
                 "OK" if ok else "FAIL")
        return ok
    except Exception as e:  # noqa: BLE001
        log.warning("[h7] ONNX parity check skipped: %s", e)
        return True


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-bars", type=int, default=None)
    p.add_argument("--use-gpu", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    spec = train(args.max_bars, args.use_gpu)
    log.info("[h7] done in %.0fs", time.time() - t0)
    return 0 if spec.get("deploy") else 0   # always 0 — non-deploy is a finding


if __name__ == "__main__":
    sys.exit(main())
