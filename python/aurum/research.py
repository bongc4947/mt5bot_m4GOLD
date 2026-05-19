"""
aurum/research.py — fast edge-search harness.

Honest premise: retraining the transformer is a ~6 h Kaggle job, far too
slow to iterate on. But the question "does this target / feature set
carry any causal edge?" is answered just as well by XGBoost under purged
cross-validation in ~1 minute. If a gradient-boosted model on causal
features cannot beat a coin flip, the transformer will not either.

So this harness is the search engine: it builds CAUSAL tabular features
from GOLD M5 bars, builds a configurable TARGET, and scores it with
XGBoost under PurgedKFold — reporting profit factor, accuracy and edge
over a passive baseline. Run a battery of hypotheses, keep what shows
edge, discard what doesn't.

    python python/aurum/research.py            # run the default battery
    python python/aurum/research.py --max-bars 120000

Everything here is strictly backward-looking — no leak. That is the
whole point: measure the HONEST edge.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from cv.purged_kfold import PurgedKFold

log = logging.getLogger(__name__)
_PF_CAP = 10.0


# ===========================================================================
# Causal features — every column uses only past/current closed bars
# ===========================================================================
def build_features(m5: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    o = m5["open"].to_numpy(np.float64)
    h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64)
    c = m5["close"].to_numpy(np.float64)
    v = m5["volume"].to_numpy(np.float64)
    t = pd.to_datetime(m5["time"], utc=True)
    eps = 1e-12
    logc = np.log(np.clip(c, eps, None))

    def ret(n):
        r = np.zeros_like(c)
        r[n:] = logc[n:] - logc[:-n]
        return r

    def roll(a, n, fn):
        return pd.Series(a).rolling(n, min_periods=max(2, n // 2)).agg(fn).to_numpy()

    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = roll(tr, 14, "mean")
    atr48 = roll(tr, 48, "mean")
    r1 = ret(1)

    feats, names = [], []

    def add(name, col):
        feats.append(np.nan_to_num(col.astype(np.float64), nan=0.0,
                                   posinf=0.0, neginf=0.0))
        names.append(name)

    for n in (1, 3, 6, 12, 24, 48, 96):
        add(f"ret{n}", ret(n))
    add("atr14_norm", atr14 / np.clip(c, eps, None))
    add("atr48_norm", atr48 / np.clip(c, eps, None))
    add("hl_rng12", roll((h - l) / np.clip(c, eps, None), 12, "mean"))
    add("rv24", roll(r1, 24, "std"))
    add("rv96", roll(r1, 96, "std"))
    add("rv_ratio", roll(r1, 24, "std") / np.clip(roll(r1, 96, "std"), eps, None))
    # position of close within recent ranges (0 = low, 1 = high)
    for n in (24, 96, 288):
        hi_n = roll(h, n, "max")
        lo_n = roll(l, n, "min")
        add(f"pos{n}", (c - lo_n) / np.clip(hi_n - lo_n, eps, None))
    # distance from EMAs, in ATR units
    for n in (20, 50, 200):
        ema = pd.Series(c).ewm(span=n, adjust=False).mean().to_numpy()
        add(f"ema{n}_dist", (c - ema) / np.clip(atr14, eps, None))
    add("vol_ratio", v / np.clip(roll(v, 50, "mean"), eps, None))
    add("vol_z", (v - roll(v, 96, "mean")) / np.clip(roll(v, 96, "std"), eps, None))
    # session / calendar
    hod = t.dt.hour.to_numpy() + t.dt.minute.to_numpy() / 60.0
    add("hod_sin", np.sin(2 * np.pi * hod / 24.0))
    add("hod_cos", np.cos(2 * np.pi * hod / 24.0))
    dow = t.dt.dayofweek.to_numpy()
    add("dow_sin", np.sin(2 * np.pi * dow / 7.0))
    add("dow_cos", np.cos(2 * np.pi * dow / 7.0))

    X = np.column_stack(feats).astype(np.float32)
    return X, names


# ===========================================================================
# Targets — each returns (y, kind) where kind drives the metric
# ===========================================================================
def _atr(m5: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64)
    c = m5["close"].to_numpy(np.float64)
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()


def _primary_signal(m5: pd.DataFrame, prim_type: str,
                    a: int, b: int) -> np.ndarray:
    """Deterministic trend signal — +1 long / -1 short. The 'direction'
    half of a meta-labelled strategy (ML never predicts direction)."""
    c = m5["close"].to_numpy(np.float64)
    if prim_type == "ema":
        ef = pd.Series(c).ewm(span=a, adjust=False).mean().to_numpy()
        es = pd.Series(c).ewm(span=b, adjust=False).mean().to_numpy()
        return np.where(ef > es, 1, -1).astype(np.int64)
    if prim_type == "mom":
        prev = np.concatenate([np.full(a, c[0]), c[:-a]])
        return np.where(c > prev, 1, -1).astype(np.int64)
    if prim_type == "donchian":
        hi = pd.Series(m5["high"]).rolling(a, min_periods=a).max().shift(1)
        lo = pd.Series(m5["low"]).rolling(a, min_periods=a).min().shift(1)
        mid = ((hi + lo) / 2.0).to_numpy()
        return np.where(c > np.nan_to_num(mid, nan=c[0]), 1, -1).astype(np.int64)
    raise ValueError(f"unknown primary: {prim_type}")


def make_target(m5: pd.DataFrame, kind: str, horizon: int,
                tp_atr: float = 2.0, sl_atr: float = 1.0,
                prim_type: str = "ema", prim_a: int = 20,
                prim_b: int = 50) -> dict:
    """
    Returns {y, fwd_ret, kind}. y is the label; fwd_ret is the realised
    forward return over `horizon` bars (for PF scoring).
    """
    c = m5["close"].to_numpy(np.float64)
    h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64)
    n = len(c)
    eps = 1e-12
    fwd = np.zeros(n, dtype=np.float64)
    fwd[:n - horizon] = np.log(np.clip(c[horizon:], eps, None)
                               / np.clip(c[:n - horizon], eps, None))

    if kind in ("tb", "tb_sym"):
        atr = _atr(m5)
        tp_m = tp_atr if kind == "tb" else sl_atr   # symmetric uses sl_atr both
        y = np.full(n, 1, dtype=np.int64)            # 1 = flat
        for i in range(n - horizon):
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            tp = c[i] + tp_m * a
            sl = c[i] - sl_atr * a
            win = slice(i + 1, i + 1 + horizon)
            hw, lw = h[win], l[win]
            ht = np.argmax(hw >= tp) if (hw >= tp).any() else horizon + 1
            hs = np.argmax(lw <= sl) if (lw <= sl).any() else horizon + 1
            if ht < hs:
                y[i] = 2
            elif hs < ht:
                y[i] = 0
        return {"y": y, "fwd_ret": fwd, "kind": "direction"}

    if kind == "sign":
        # plain sign of the forward return — 2 long / 0 short (no flat)
        y = np.where(fwd > 0, 2, 0).astype(np.int64)
        return {"y": y, "fwd_ret": fwd, "kind": "direction"}

    if kind == "volexp":
        # will the next-`horizon` realised range exceed its rolling median?
        rng = np.zeros(n)
        for i in range(n - horizon):
            w = slice(i + 1, i + 1 + horizon)
            rng[i] = (h[w].max() - l[w].min()) / max(c[i], eps)
        med = pd.Series(rng).rolling(2000, min_periods=200).median().to_numpy()
        y = (rng > med).astype(np.int64)
        return {"y": y, "fwd_ret": fwd, "kind": "binary"}

    if kind == "metalabel":
        # primary = a deterministic trend rule; label = 1 if following it
        # over `horizon` bars would have profited. ML learns ONLY the
        # accept/reject filter, never the direction.
        prim = _primary_signal(m5, prim_type, prim_a, prim_b)
        y = ((prim * fwd) > 0).astype(np.int64)
        return {"y": y, "fwd_ret": fwd, "kind": "binary", "primary": prim}

    raise ValueError(f"unknown target kind: {kind}")


# ===========================================================================
# Scoring
# ===========================================================================
def _pf(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    g = float(pnl[pnl > 0].sum())
    ls = float(-pnl[pnl < 0].sum())
    if ls <= 1e-12:
        return _PF_CAP if g > 0 else 0.0
    return min(_PF_CAP, g / ls)


def evaluate(X: np.ndarray, tgt: dict, horizon: int, n_splits: int = 5,
             use_gpu: bool = False, cost: float = 1.5e-4,
             act_threshold: float = 0.5) -> dict:
    """
    Purged-CV XGBoost on the target. `cost` is the round-trip transaction
    cost (spread) in log-return units, subtracted from every trade — so
    the reported PF is NET, not gross. `act_threshold` (meta targets) is
    the P(act) cut: higher = fewer, higher-conviction trades.
    """
    from xgboost import XGBClassifier
    y = tgt["y"]
    fwd = tgt["fwd_ret"]
    classes = sorted(np.unique(y).tolist())
    n_class = len(classes)
    remap = {cl: i for i, cl in enumerate(classes)}
    y_ix = np.array([remap[v] for v in y], dtype=np.int64)

    pk = PurgedKFold(n_splits=n_splits, horizon=horizon, embargo_pct=0.01)
    pfs, accs, n_tr, trades = [], [], [], []
    for tr, te in pk.split(len(y)):
        counts = np.bincount(y_ix[tr], minlength=n_class).astype(np.float64)
        w = counts.sum() / np.maximum(counts, 1.0)
        model = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="multi:softprob" if n_class > 2 else "binary:logistic",
            num_class=n_class if n_class > 2 else None,
            tree_method="hist", device="cuda" if use_gpu else "cpu",
            random_state=42, n_jobs=0)
        model.fit(X[tr], y_ix[tr], sample_weight=w[y_ix[tr]])
        pred_ix = model.predict(X[te])
        pred = np.array([classes[p] for p in pred_ix])
        accs.append(float((pred == y[te]).mean()))
        # PF: trade only on direction targets (classes include 0/2 = short/long)
        if tgt["kind"] == "direction":
            sign = np.where(pred == 2, 1.0, np.where(pred == 0, -1.0, 0.0))
            pnl = (sign * fwd[te])[sign != 0] - cost
            pfs.append(_pf(pnl))
        elif "primary" in tgt:
            # meta: act on the primary when P(act) clears the threshold
            proba1 = model.predict_proba(X[te])[:, 1]
            act = proba1 >= act_threshold
            prim = tgt["primary"][te]
            pnl = prim[act] * fwd[te][act] - cost
            pfs.append(_pf(pnl))
            trades.append(int(act.sum()))
        else:
            pfs.append(float("nan"))
        n_tr.append(int(len(te)))
    return {"pf": float(np.nanmean(pfs)),
            "pf_folds": [round(x, 3) for x in pfs],
            "acc": float(np.mean(accs)),
            "n_test": int(np.mean(n_tr)),
            "trades": int(np.mean(trades)) if trades else 0}


# ===========================================================================
# Battery
# ===========================================================================
def run_battery(hypotheses: list[dict], max_bars: int | None,
                use_gpu: bool) -> list[dict]:
    from aurum.datamodule import _load_m5_bars
    m5 = _load_m5_bars()
    if max_bars and len(m5) > max_bars:
        m5 = m5.iloc[-max_bars:].reset_index(drop=True)
    log.info("[research] %d M5 bars", len(m5))
    X, names = build_features(m5)
    log.info("[research] %d causal features: %s", len(names), names)

    results = []
    for hyp in hypotheses:
        t0 = time.time()
        tgt = make_target(m5, hyp["kind"], hyp["horizon"],
                          tp_atr=hyp.get("tp_atr", 2.0),
                          sl_atr=hyp.get("sl_atr", 1.0),
                          prim_type=hyp.get("prim_type", "ema"),
                          prim_a=hyp.get("prim_a", 20),
                          prim_b=hyp.get("prim_b", 50))
        m = evaluate(X, tgt, hyp["horizon"], use_gpu=use_gpu,
                     cost=hyp.get("cost", 1.5e-4),
                     act_threshold=hyp.get("act_threshold", 0.5))
        row = {**hyp, **m, "secs": round(time.time() - t0, 1)}
        results.append(row)
        log.info("[research] %-30s PF=%.3f  acc=%.3f  trades=%d  folds=%s",
                 hyp["name"], m["pf"], m["acc"], m["trades"], m["pf_folds"])
    return results


_DEFAULT_BATTERY = [
    {"name": "tb_dir_h20 (control)",  "kind": "tb",        "horizon": 20},
    {"name": "tb_dir_h48",           "kind": "tb",        "horizon": 48},
    {"name": "tb_dir_h96",           "kind": "tb",        "horizon": 96},
    {"name": "tb_dir_h240",          "kind": "tb",        "horizon": 240},
    {"name": "tb_sym_h96",           "kind": "tb_sym",    "horizon": 96,
     "sl_atr": 1.5},
    {"name": "sign_h96",             "kind": "sign",      "horizon": 96},
    {"name": "sign_h240",            "kind": "sign",      "horizon": 240},
    {"name": "volexp_h48",           "kind": "volexp",    "horizon": 48},
    {"name": "metalabel_emacross_h96", "kind": "metalabel", "horizon": 96},
]


def _meta(name, horizon, prim_type, a, b=0, thr=0.5):
    return {"name": name, "kind": "metalabel", "horizon": horizon,
            "prim_type": prim_type, "prim_a": a, "prim_b": b,
            "act_threshold": thr}


# Iteration 2 — refine the meta-label lead from iteration 1. PF is NET
# of a 1.5e-4 round-trip cost. Varies the trend primary and the horizon.
_BATTERY_2 = [
    _meta("ml_ema10/30_h48",   48,  "ema", 10, 30),
    _meta("ml_ema10/30_h96",   96,  "ema", 10, 30),
    _meta("ml_ema20/50_h48",   48,  "ema", 20, 50),
    _meta("ml_ema20/50_h96",   96,  "ema", 20, 50),
    _meta("ml_ema20/50_h144",  144, "ema", 20, 50),
    _meta("ml_ema50/200_h96",  96,  "ema", 50, 200),
    _meta("ml_ema50/200_h240", 240, "ema", 50, 200),
    _meta("ml_mom60_h96",      96,  "mom", 60),
    _meta("ml_mom120_h96",     96,  "mom", 120),
    _meta("ml_mom120_h240",    240, "mom", 120),
    _meta("ml_donch48_h96",    96,  "donchian", 48),
    _meta("ml_donch96_h144",   144, "donchian", 96),
]


# Iteration 3 — lock the best meta-label config: P(act) threshold sweep
# on the iteration-2 winners + a few even-slower primaries.
_BATTERY_3 = [
    _meta("ema50_200_h240_t50", 240, "ema", 50, 200, 0.50),
    _meta("ema50_200_h240_t55", 240, "ema", 50, 200, 0.55),
    _meta("ema50_200_h240_t60", 240, "ema", 50, 200, 0.60),
    _meta("ema50_200_h240_t65", 240, "ema", 50, 200, 0.65),
    _meta("donch96_h144_t50",   144, "donchian", 96, 0, 0.50),
    _meta("donch96_h144_t58",   144, "donchian", 96, 0, 0.58),
    _meta("donch96_h144_t65",   144, "donchian", 96, 0, 0.65),
    _meta("mom120_h240_t55",    240, "mom", 120, 0, 0.55),
    _meta("mom120_h240_t62",    240, "mom", 120, 0, 0.62),
    _meta("ema100_300_h240_t55",240, "ema", 100, 300, 0.55),
    _meta("mom200_h288_t55",    288, "mom", 200, 0, 0.55),
    _meta("donch120_h240_t58",  240, "donchian", 120, 0, 0.58),
]


# Iteration 4 — cost stress test. Re-run the two every-fold-positive
# winners at pessimistic round-trip costs to confirm the edge is real.
_BATTERY_4 = [
    {**_meta("ema50_200_h240 cost1.5", 240, "ema", 50, 200, 0.50), "cost": 1.5e-4},
    {**_meta("ema50_200_h240 cost2.5", 240, "ema", 50, 200, 0.50), "cost": 2.5e-4},
    {**_meta("ema50_200_h240 cost4.0", 240, "ema", 50, 200, 0.50), "cost": 4.0e-4},
    {**_meta("donch96_h144 cost1.5",   144, "donchian", 96, 0, 0.50), "cost": 1.5e-4},
    {**_meta("donch96_h144 cost2.5",   144, "donchian", 96, 0, 0.50), "cost": 2.5e-4},
    {**_meta("donch96_h144 cost4.0",   144, "donchian", 96, 0, 0.50), "cost": 4.0e-4},
]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        stream=sys.stdout, force=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-bars", type=int, default=None)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--battery", type=int, default=1, choices=(1, 2, 3, 4),
                   help="1=target search 2=meta refine 3=threshold 4=cost test")
    p.add_argument("--out", default=None, help="results JSON path")
    args = p.parse_args(argv)

    battery = {1: _DEFAULT_BATTERY, 2: _BATTERY_2,
               3: _BATTERY_3, 4: _BATTERY_4}[args.battery]
    results = run_battery(battery, args.max_bars, args.use_gpu)

    print("\n" + "=" * 78)
    print(f"  {'hypothesis':<28}{'PF':>8}{'acc':>8}{'edge':>10}")
    print("  " + "-" * 74)
    for r in sorted(results, key=lambda x: -x["pf"]):
        edge = "EDGE" if r["pf"] >= 1.15 else ("weak" if r["pf"] >= 1.05 else "—")
        print(f"  {r['name']:<28}{r['pf']:>8.3f}{r['acc']:>8.3f}{edge:>10}")
    print("=" * 78)

    out = Path(args.out) if args.out else \
        Path(__file__).parent.parent.parent / "onnx_out" / "AURUM_research.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
