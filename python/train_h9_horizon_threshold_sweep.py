"""
train_h9_horizon_threshold_sweep.py — find the (label_horizon, threshold)
combo that maximises trade frequency without breaking PF.

The user wants:
  (a) improved edge classification on the meta-gate
  (b) higher trade frequency

These are in tension. Lowering the threshold or shortening the horizon
both INCREASE trade count and usually DECREASE per-trade PF. This script
maps the frontier so we can pick the best tradeoff.

Experiment design:
  - features: 18 baseline + 6 sr_fib (the +0.039 PF lift from train_h7)
  - label horizons tested: [120, 180, 240]   (10h / 15h / 20h)
  - thresholds scored: [0.50, 0.52, 0.55, 0.58, 0.60]
  - purged 6-fold CV at each (horizon, threshold), single XGBoost fit
    per (fold, horizon), THEN score all 5 thresholds on the same model

Selection rule: pick (horizon, threshold) that maximises
   PF * sqrt(trades / 1000)
   subject to: meanPF >= 1.15, min_fold_pf >= 0.92
This rewards both edge AND frequency without one dominating.
"""
from __future__ import annotations
import logging, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from aurum.metatrend import (
    build_features, build_meta_label, primary_signal,
    META_FEATURES, N_META_FEATURES, COST_ROUNDTRIP,
)
from aurum.sr_fib_features import extract as sr_extract, \
    SR_FIB_FEATURES, N_SR_FIB_FEATURES
from cv.purged_kfold import PurgedKFold

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s", stream=sys.stdout, force=True)
log = logging.getLogger(__name__)

HORIZONS    = [120, 180, 240]
THRESHOLDS  = [0.50, 0.52, 0.55, 0.58, 0.60]
N_SPLITS    = 6
PF_CAP      = 10.0
SR_WARMUP   = 24 * 21 * 12   # 21 days of M5 bars


def _pf(pnl: np.ndarray, min_trades: int = 30) -> float:
    if len(pnl) < min_trades: return 0.0
    g  = float(pnl[pnl > 0].sum())
    ls = float(-pnl[pnl < 0].sum())
    if ls <= 1e-12: return PF_CAP if g > 0 else 0.0
    return min(PF_CAP, g / ls)


def _make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", tree_method="hist",
        device="cpu", random_state=42, n_jobs=0)


def main() -> int:
    from aurum.datamodule import _load_m5_bars
    m5 = _load_m5_bars()
    log.info("[h9] loaded %d M5 bars  span=%s -> %s",
             len(m5), m5["time"].iloc[0], m5["time"].iloc[-1])

    # Feature matrix (one-time computation, shared across all horizons)
    X_full = build_features(m5)
    anchors_full = np.arange(SR_WARMUP, len(m5), dtype=np.int64)
    X_sr = sr_extract(m5, anchors_full)
    X = np.concatenate([X_full[SR_WARMUP:], X_sr], axis=1).astype(np.float32)
    prim_full = primary_signal(m5["close"].to_numpy(np.float64))
    log.info("[h9] feature matrix: %d rows x %d cols", X.shape[0], X.shape[1])

    # Results storage: results[H][thr] = {pf_folds, trades_folds, min_fold, mean_pf, mean_trades}
    results: dict[int, dict[float, dict]] = {}

    for H in HORIZONS:
        log.info("="*60)
        log.info("HORIZON = %d M5 bars (~%.1f h)", H, H * 5 / 60)
        lab = build_meta_label(m5, horizon=H)
        y_full = lab["y"]; fwd_full = lab["fwd_ret"]
        y    = y_full[SR_WARMUP:]
        prim = prim_full[SR_WARMUP:]
        fwd  = fwd_full[SR_WARMUP:]

        pk = PurgedKFold(n_splits=N_SPLITS, horizon=H, embargo_pct=0.01)
        # per-threshold accumulator
        per_thr_pf: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
        per_thr_tr: dict[float, list[int]]   = {t: [] for t in THRESHOLDS}

        for fold, (tr, te) in enumerate(pk.split(len(y))):
            cnt = np.bincount(y[tr], minlength=2).astype(np.float64)
            w = cnt.sum() / np.maximum(cnt, 1.0)
            model = _make_xgb()
            model.fit(X[tr], y[tr], sample_weight=w[y[tr]])
            pact = model.predict_proba(X[te])[:, 1]

            for thr in THRESHOLDS:
                act = pact >= thr
                pnl = prim[te][act] * fwd[te][act] - COST_ROUNDTRIP
                pf = _pf(pnl)
                per_thr_pf[thr].append(pf)
                per_thr_tr[thr].append(int(act.sum()))

            log.info("  fold %d/%d  trades@0.50=%d  PF@0.55=%.3f  PF@0.60=%.3f",
                     fold + 1, N_SPLITS,
                     int((pact >= 0.50).sum()),
                     per_thr_pf[0.55][-1], per_thr_pf[0.60][-1])

        results[H] = {}
        for thr in THRESHOLDS:
            pf_folds = per_thr_pf[thr]
            tr_folds = per_thr_tr[thr]
            results[H][thr] = {
                "mean_pf":    float(np.mean(pf_folds)),
                "min_fold":   float(min(pf_folds)),
                "folds_pos":  sum(1 for p in pf_folds if p >= 1.0),
                "mean_trades": int(np.mean(tr_folds)),
                "pf_folds":   [round(p, 3) for p in pf_folds],
            }

    # ---- print full table ----
    log.info("="*92)
    log.info("HORIZON-THRESHOLD SWEEP RESULTS")
    log.info("="*92)
    log.info("%-6s %-6s %-9s %-9s %-9s %-10s %-7s",
             "H", "thr", "meanPF", "minFold", "folds+", "trd/fold", "score")
    best = None
    for H in HORIZONS:
        for thr in THRESHOLDS:
            r = results[H][thr]
            # composite score: PF * sqrt(trades/1000); penalize if fails gate
            score = r["mean_pf"] * np.sqrt(max(r["mean_trades"], 1) / 1000.0)
            gate_ok = r["mean_pf"] >= 1.15 and r["min_fold"] >= 0.92
            note = "" if gate_ok else " (gate fail)"
            log.info("%-6d %-6.2f %-9.3f %-9.3f %-9d %-10d %-7.3f%s",
                     H, thr, r["mean_pf"], r["min_fold"], r["folds_pos"],
                     r["mean_trades"], score, note)
            if gate_ok and (best is None or score > best[0]):
                best = (score, H, thr, r)

    log.info("="*92)
    if best:
        score, H, thr, r = best
        log.info("BEST  H=%d  thr=%.2f  meanPF=%.3f  trd/fold=%d  composite=%.3f",
                 H, thr, r["mean_pf"], r["mean_trades"], score)
        log.info("       folds: %s", r["pf_folds"])
        # vs baseline (h=240, thr=0.55, no sr_fib was 1.406 with ~10800 trades/fold)
        base = results[240][0.55]
        log.info("BASELINE 240/0.55: meanPF=%.3f trd/fold=%d  ->  best lifts PF %+.3f and trades %+d",
                 base["mean_pf"], base["mean_trades"],
                 r["mean_pf"] - base["mean_pf"], r["mean_trades"] - base["mean_trades"])
    else:
        log.info("NO config cleared the deploy gate (meanPF >= 1.15, min_fold >= 0.92)")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    log.info("done in %.0fs", time.time() - t0)
    sys.exit(rc)
