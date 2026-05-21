"""
train_h8_quantile.py - train 7 quantile-regression heads on the MetaTrend
18-feature set, producing the distributional forecast for the v1.20 EA.

Quantiles: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
Target:    forward 240-bar log-return (same horizon as the meta-label)
Engine:    XGBoost 3.x reg:quantileerror, one model per quantile
           (XGBoost 3.1.1 multi-output multi-quantile is bugged on this
            release, so we train 7 separate single-output regressors)
Validation: purged 6-fold CV with embargo, pinball-loss per quantile + an
           integrated PF backtest of the v1.20 decision logic.
Export:    7 ONNX files staged in onnx_out/ + a manifest spec.

Decision logic the EA will run (see MetaGate.mqh + MetaTrend.mq5 v1.20):
  - q10 tail veto:   refuse trade if q10 < -InpQ10VetoAtr * ATR
  - q50 sign check:  median must agree with primary trend
  - q05 dynamic SL:  initial SL placed at q05 (in points) + small buffer
  - Kelly sizing:    lot = InpBaseLot * clip(mu/sigma^2, 0.5, 3.0)
                     mu = q50,  sigma ~ (q90 - q10) / 2.56
  - q90 trail loosen: when price reaches q90, widen trail-stop multiplier

Run:
    python python/train_h8_quantile.py
    python python/train_h8_quantile.py --max-bars 80000   # smoke
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
    META_FEATURES, N_META_FEATURES, LABEL_HORIZON, COST_ROUNDTRIP,
)
from cv.purged_kfold import PurgedKFold

log = logging.getLogger(__name__)
_ARTIFACT_DIR = Path(__file__).parent.parent / "onnx_out"
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
Q_NAMES = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
_N_SPLITS = 6
_PF_CAP = 10.0

# decision-logic knobs (mirrored in MetaTrend.mq5 v1.20)
_Q10_VETO_ATR    = 1.5      # refuse trade if q10 < -1.5 * ATR
_KELLY_FRACTION  = 0.25     # quarter-Kelly (drawdown protection)
_KELLY_MIN       = 0.5      # lot multiplier floor
_KELLY_MAX       = 3.0      # lot multiplier ceiling
_ATR_PROXY_BARS  = 14       # use rv24 * sqrt(48) / sqrt(N) as an ATR proxy


def _pf(pnl: np.ndarray, min_trades: int = 30) -> float:
    if len(pnl) < min_trades: return 0.0
    g = float(pnl[pnl > 0].sum())
    ls = float(-pnl[pnl < 0].sum())
    if ls <= 1e-12: return _PF_CAP if g > 0 else 0.0
    return min(_PF_CAP, g / ls)


def _pinball(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss at level alpha."""
    err = y - q_pred
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def _make_qmodel(alpha: float, use_gpu: bool):
    from xgboost import XGBRegressor
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, n_jobs=0,
    )


def _isotonize_row(q: np.ndarray) -> np.ndarray:
    """Sort each row so quantile order is preserved (cheap monotonic fix)."""
    return np.sort(q, axis=-1)


def _v120_decision(qrow: np.ndarray, prim: int, atr_proxy: float,
                   meta_pact: float, meta_thr: float, cost: float,
                   fwd: float) -> float:
    """
    Replay the v1.20 EA decision and return the per-bar PnL.

    qrow = [q05, q10, q25, q50, q75, q90, q95] forecast of forward log-return.
    All other inputs in matching units.

    Returns 0.0 if the gate vetoes; otherwise the cost-adjusted realised
    return times the Kelly-scaled lot multiplier.
    """
    # 1) baseline meta-gate
    if meta_pact < meta_thr: return 0.0
    if prim == 0: return 0.0
    # 2) q50 sign agrees with primary
    if (qrow[3] > 0 and prim < 0) or (qrow[3] < 0 and prim > 0):
        return 0.0
    # 3) tail veto
    if qrow[1] < -_Q10_VETO_ATR * atr_proxy: return 0.0
    # 4) Kelly sizing
    mu = float(qrow[3])
    sigma = max(1e-9, float(qrow[5] - qrow[1]) / 2.56)
    kelly = mu / (sigma * sigma + 1e-12) * _KELLY_FRACTION
    lot_mult = float(np.clip(abs(kelly), _KELLY_MIN, _KELLY_MAX))
    # signed PnL on this bar over the LABEL_HORIZON
    return lot_mult * (prim * fwd - cost)


def train(max_bars: int | None, use_gpu: bool) -> dict:
    from aurum.datamodule import _load_m5_bars

    m5 = _load_m5_bars()
    if max_bars and len(m5) > max_bars:
        m5 = m5.iloc[-max_bars:].reset_index(drop=True)
    log.info("[h8] %d M5 bars  span=%s -> %s",
             len(m5), m5["time"].iloc[0], m5["time"].iloc[-1])

    X = build_features(m5)
    lab = build_meta_label(m5)
    y_fwd  = lab["fwd_ret"].astype(np.float32)
    prim_v = lab["primary"]

    # baseline meta-gate P(act) computed by the existing classifier so we can
    # replay the gated decision; here we approximate by running a fresh
    # quick classifier fit per fold inside the CV loop.
    from xgboost import XGBClassifier
    meta_thr = 0.55

    # ATR proxy in log-return units, per bar (rv24 column is index 5)
    atr_proxy = X[:, 5].astype(np.float64) * np.sqrt(24.0)
    atr_proxy = np.clip(atr_proxy, 1e-6, None)

    # ---- purged 6-fold CV ----
    pk = PurgedKFold(n_splits=_N_SPLITS, horizon=LABEL_HORIZON,
                     embargo_pct=0.01)
    pinball_folds = {qn: [] for qn in Q_NAMES}
    pf_baseline_folds, pf_v120_folds = [], []
    cross_rate_folds = []

    for fold, (tr, te) in enumerate(pk.split(len(y_fwd))):
        # train baseline classifier for the meta-gate replay
        y_cls = ((prim_v * y_fwd - COST_ROUNDTRIP) > 0).astype(np.int64)
        cnt = np.bincount(y_cls[tr], minlength=2).astype(np.float64)
        w_cls = cnt.sum() / np.maximum(cnt, 1.0)
        cls = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", tree_method="hist",
            device="cuda" if use_gpu else "cpu", random_state=42, n_jobs=0,
        )
        cls.fit(X[tr], y_cls[tr], sample_weight=w_cls[y_cls[tr]])
        pact_te = cls.predict_proba(X[te])[:, 1]

        # train 7 quantile heads
        q_te = np.zeros((len(te), len(QUANTILES)), dtype=np.float64)
        for k, alpha in enumerate(QUANTILES):
            m = _make_qmodel(alpha, use_gpu)
            m.fit(X[tr], y_fwd[tr])
            q_te[:, k] = m.predict(X[te])
            pinball_folds[Q_NAMES[k]].append(
                _pinball(y_fwd[te], q_te[:, k], alpha))

        # measure quantile-crossing rate BEFORE isotonisation
        cross = (np.diff(q_te, axis=1) < 0).any(axis=1).mean()
        cross_rate_folds.append(float(cross))
        q_te = _isotonize_row(q_te)

        # ---- integrated PF backtest -----------------------------
        # baseline: meta-gate-only
        baseline_act = pact_te >= meta_thr
        pnl_base = prim_v[te][baseline_act] * y_fwd[te][baseline_act] - COST_ROUNDTRIP
        pf_baseline_folds.append(_pf(pnl_base))

        # v1.20: meta-gate + q50 sign + q10 tail veto + Kelly sizing
        pnl_v120 = np.array([
            _v120_decision(q_te[i], int(prim_v[te][i]), atr_proxy[te][i],
                           pact_te[i], meta_thr, COST_ROUNDTRIP,
                           float(y_fwd[te][i]))
            for i in range(len(te))
        ])
        pnl_v120 = pnl_v120[pnl_v120 != 0]    # drop bars where the gate vetoed
        pf_v120_folds.append(_pf(pnl_v120))

        log.info("[h8] fold %d/%d  baselinePF=%.3f  v1.20_PF=%.3f  "
                 "trades_base=%d  trades_v120=%d  cross=%.2f%%",
                 fold + 1, _N_SPLITS, pf_baseline_folds[-1],
                 pf_v120_folds[-1], int(baseline_act.sum()), len(pnl_v120),
                 100 * cross)

    base_mean = float(np.mean(pf_baseline_folds))
    v120_mean = float(np.mean(pf_v120_folds))
    log.info("[h8] baseline meanPF=%.3f  v1.20 meanPF=%.3f  "
             "delta=%+.3f  cross_rate=%.2f%%",
             base_mean, v120_mean, v120_mean - base_mean,
             100 * float(np.mean(cross_rate_folds)))

    # ---- train final 7 quantile heads on ALL data + export ONNX ----
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    onnx_paths = {}
    pf_byq = {}
    for k, alpha in enumerate(QUANTILES):
        log.info("[h8] training final head q%.2f on full data ...", alpha)
        m = _make_qmodel(alpha, use_gpu)
        m.fit(X, y_fwd)
        path = _ARTIFACT_DIR / f"M4GOLD_QUANTILE_{Q_NAMES[k]}_GOLD.onnx"
        onnx = convert_xgboost(m, initial_types=[("input",
                FloatTensorType([1, N_META_FEATURES]))])
        path.write_bytes(onnx.SerializeToString())
        onnx_paths[Q_NAMES[k]] = path.name
        pf_byq[Q_NAMES[k]] = round(float(np.mean(pinball_folds[Q_NAMES[k]])), 6)
        log.info("[h8]  -> %s  (pinball CV mean = %.4f)",
                 path.name, pf_byq[Q_NAMES[k]])

    # parity check on a random anchor
    import onnxruntime as ort
    log.info("[h8] ONNX parity test on 4 random anchors ...")
    rng = np.random.default_rng(7)
    test_idx = rng.choice(len(X), 4, replace=False)
    max_err = 0.0
    for k, qn in enumerate(Q_NAMES):
        sess = ort.InferenceSession(str(_ARTIFACT_DIR / onnx_paths[qn]),
                                     providers=["CPUExecutionProvider"])
        m = _make_qmodel(QUANTILES[k], use_gpu)
        m.fit(X, y_fwd)
        for i in test_idx:
            xi = X[i:i+1].astype(np.float32)
            xgb_p = float(m.predict(xi)[0])
            onnx_p = float(sess.run(None, {"input": xi})[0].ravel()[0])
            max_err = max(max_err, abs(xgb_p - onnx_p))
    log.info("[h8] parity max-err across all 7 heads: %.3e %s",
             max_err, "OK" if max_err < 1e-4 else "HIGH")

    spec = {
        "strategy": "QUANTILE_FORECAST",
        "symbol": "GOLD",
        "version": "quantile-1.0.0",
        "features": META_FEATURES,
        "n_features": N_META_FEATURES,
        "quantiles": QUANTILES,
        "q_names": Q_NAMES,
        "onnx_files": onnx_paths,
        "label_horizon_bars": LABEL_HORIZON,
        "cost_roundtrip": COST_ROUNDTRIP,
        "decision_knobs": {
            "q10_veto_atr":   _Q10_VETO_ATR,
            "kelly_fraction": _KELLY_FRACTION,
            "kelly_min":      _KELLY_MIN,
            "kelly_max":      _KELLY_MAX,
        },
        "cv": {
            "baseline_mean_pf": round(base_mean, 4),
            "v120_mean_pf":     round(v120_mean, 4),
            "delta_pf":         round(v120_mean - base_mean, 4),
            "baseline_folds":   [round(x, 3) for x in pf_baseline_folds],
            "v120_folds":       [round(x, 3) for x in pf_v120_folds],
            "cross_rate_pct":   round(100 * float(np.mean(cross_rate_folds)), 2),
            "pinball_per_q":    pf_byq,
        },
        "parity_max_err": float(max_err),
    }
    spec_path = _ARTIFACT_DIR / "M4GOLD_QUANTILE_GOLD_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    log.info("[h8] spec -> %s", spec_path.name)
    return spec


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--max-bars", type=int, default=None)
    p.add_argument("--use-gpu", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    spec = train(args.max_bars, args.use_gpu)
    log.info("[h8] done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
