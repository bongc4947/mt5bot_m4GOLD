"""
_train_agent.py — shared training routine used by all per-agent scripts.
Not meant to be run directly.
"""

import sys
import time
import logging
from pathlib import Path
from typing import List

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pip_size lookup by symbol
# ---------------------------------------------------------------------------

_PIP = {
    "GOLD": 0.01,   "SILVER": 0.01,     "PLATINUM": 0.01,  "COPPER": 0.001,
    "USDJPY": 0.01, "BTCUSD": 1.0,      "ETHUSD": 0.1,     "LTCUSD": 0.01,
    "CrudeOIL": 0.01, "BRENT_OIL": 0.01, "NATURAL_GAS": 0.001,
    "US_500": 0.01, "UK_100": 0.01,
}

def pip_size(symbol: str) -> float:
    if symbol in _PIP:
        return _PIP[symbol]
    return 0.01 if "JPY" in symbol else 0.0001


def _finite(x: float) -> float:
    """Coerce inf/nan to 0.0 for arithmetic comparisons."""
    return float(x) if (x is not None and np.isfinite(x)) else 0.0


def _evaluate_pnl_on_val(dir_model, bars, dir_feat, dir_labels, pip):
    """
    Run the trained direction model on the same chronological val slice the
    trainer used and return PnLMetrics (profit_factor, sharpe, sortino, MDD,
    win_rate, expected_value).

    Mirrors dataset.train_val_split + DirectionDataset(exclude_flat=True) layout
    so val rows align with what was used for early-stopping.
    """
    import torch
    from config import VAL_SPLIT, LABEL_FORWARD_BARS, CONF_THRESHOLD
    from eval_harness import compute_pnl_metrics

    non_flat = np.where(dir_labels != 0)[0]
    n_eff    = len(non_flat)
    n_val    = max(1, int(n_eff * VAL_SPLIT))
    n_tr     = max(1, n_eff - n_val - LABEL_FORWARD_BARS)
    val_idx_in_nonflat = np.arange(n_tr + LABEL_FORWARD_BARS, n_eff)
    val_bars_idx       = non_flat[val_idx_in_nonflat]
    if val_bars_idx.size < 50:
        raise RuntimeError(f"val slice too small ({val_bars_idx.size})")

    # Forward returns at val bar indices
    closes = bars["close"].to_numpy(dtype=np.float64)
    H = LABEL_FORWARD_BARS
    fwd = np.zeros_like(closes)
    fwd[:-H] = (closes[H:] - closes[:-H]) / closes[:-H]

    # Predictions
    dir_model.eval()
    device = next(dir_model.parameters()).device
    feats_val = np.asarray(dir_feat[val_bars_idx], dtype=np.float32)
    with torch.no_grad():
        out = []
        for s in range(0, feats_val.shape[0], 8192):
            chunk = torch.from_numpy(feats_val[s:s+8192]).to(device)
            out.append(torch.sigmoid(dir_model(chunk)).cpu().numpy())
        probs = np.concatenate(out, axis=0).reshape(-1)
    fwd_val = fwd[val_bars_idx]

    # 1 round-trip = open + close ≈ 2 × pip_size in slippage; commission 1 pip
    cost_kw = dict(bar="M5", commission=pip, slippage=pip)

    # Diagnostic: the BTCUSD/Gold/Indices runs in the 2026-05-06 Kaggle log
    # showed val_acc=0.60 alongside WR=0.000 PF=0.0 — physically impossible
    # unless the model is just playing the val-period class balance (regime
    # shift between train and val). Surface the model's actual prediction
    # distribution and the no-cost ceiling so the user can spot it.
    pred_long_frac = float((probs > 0.5).mean())
    val_long_frac  = float((dir_labels[val_bars_idx] > 0).mean())
    log_warn_msg = ""
    if abs(pred_long_frac - 0.5) > 0.40:
        log_warn_msg = (f"  [WARN] model predicts LONG on {pred_long_frac:.1%} of val bars — "
                        f"likely degenerate; val_acc may be tracking class imbalance, not skill.")
    if abs(val_long_frac - 0.5) > 0.20 and abs(pred_long_frac - val_long_frac) < 0.05:
        log_warn_msg = (f"  [WARN] val LONG-fraction={val_long_frac:.1%} matches model "
                        f"prediction LONG-fraction={pred_long_frac:.1%} — model may be "
                        f"learning val-period bias rather than directional signal.")
    log.info("  pred LONG-frac=%.3f  val LONG-frac=%.3f  prob mean=%.3f  prob std=%.3f",
             pred_long_frac, val_long_frac, float(probs.mean()), float(probs.std()))
    if log_warn_msg:
        log.warning(log_warn_msg)

    # (1) "all-bars" PF — what the model achieves if you trade every val bar.
    pred_all = (probs > 0.5).astype(np.int8)
    pnl_all  = compute_pnl_metrics(pred=pred_all, forward_returns=fwd_val, **cost_kw)

    # (1b) No-cost baseline — the directional ceiling. If WR(no-cost) is
    #      ~0.5 then the cost model isn't the issue, the model is.
    pnl_zero = compute_pnl_metrics(pred=pred_all, forward_returns=fwd_val,
                                    bar="M5", commission=0.0, slippage=0.0)
    log.info("  no-cost  : n=%-6d WR=%.3f  PF=%s  EV=%.6f  [directional ceiling]",
             pnl_zero.n_trades, pnl_zero.win_rate, pnl_zero.profit_factor,
             pnl_zero.expected_value)

    # (1c) Passive baselines — "always long" and "always short" on the same
    #      val slice with the same cost. The MAX of these two is the trend
    #      a passive trader would have ridden for free; the model has to
    #      beat it to claim directional skill. On bull-trending assets
    #      (BTC/SPX/Metals) the always-long PF can be > 1 just from drift —
    #      a model that ties this number adds zero alpha. On FX (no drift),
    #      both passive PFs hover near 1.0, so any model PF > 1 is real
    #      signal. FX is the skill canary.
    ones  = np.ones_like(pred_all)
    zeros = np.zeros_like(pred_all)
    pnl_long  = compute_pnl_metrics(pred=ones,  forward_returns=fwd_val, **cost_kw)
    pnl_short = compute_pnl_metrics(pred=zeros, forward_returns=fwd_val, **cost_kw)
    passive_pf = max(_finite(pnl_long.profit_factor), _finite(pnl_short.profit_factor))
    excess_pf  = _finite(pnl_all.profit_factor) - passive_pf
    log.info("  passive  : long_PF=%.3f  short_PF=%.3f  -> baseline=%.3f",
             _finite(pnl_long.profit_factor), _finite(pnl_short.profit_factor),
             passive_pf)
    log.info("  SKILL    : excess_PF = model_PF - passive_PF = %+.3f%s",
             excess_pf,
             "  [no skill — model just rides trend]" if excess_pf < 0.30 else "")

    # (2) "conf-thresholded" PF — what the EA *actually* trades. Live, the
    #     bot only fires when |prob - 0.5| crosses the conviction band.
    #     CONF_THRESHOLD is the canonical gate; pred=1 if prob>CONF_THRESHOLD,
    #     pred=-1 if prob<(1-CONF_THRESHOLD), else 0 (no trade).
    pred_conf = np.where(probs > CONF_THRESHOLD, 1,
                         np.where(probs < (1.0 - CONF_THRESHOLD), -1, 0)).astype(np.int8)
    n_traded = int(np.count_nonzero(pred_conf))
    used_threshold = CONF_THRESHOLD
    if n_traded < 50:
        # Mode-collapsed model: probs all hugging 0.5, so CONF_THRESHOLD never
        # fires. Fall back to "top 10% by absolute distance from 0.5" — gives
        # a meaningful diagnostic even on saturated outputs (otherwise the
        # eval reports n=0 / nan and the user learns nothing).
        abs_dev = np.abs(probs - 0.5)
        if abs_dev.size >= 50:
            cutoff = np.quantile(abs_dev, 0.90)        # top 10%
            top_mask = abs_dev >= cutoff
            pred_conf = np.where(top_mask & (probs > 0.5), 1,
                         np.where(top_mask & (probs < 0.5), -1, 0)).astype(np.int8)
            n_traded = int(np.count_nonzero(pred_conf))
            # Effective threshold the cutoff implies (for logging).
            used_threshold = float(0.5 + cutoff)

    if n_traded >= 50:
        pnl_conf = compute_pnl_metrics(pred=pred_conf, forward_returns=fwd_val, **cost_kw)
    else:
        pnl_conf = None

    # Stash both in pnl_all so the meta.json carries both views; expose
    # primary keys = the conf-thresholded numbers (closer to live behaviour).
    pnl_all_dict = pnl_all.as_dict()
    out_dict = dict(pnl_all_dict)
    if pnl_conf is not None:
        for k, v in pnl_conf.as_dict().items():
            out_dict[f"conf_{k}"] = v
    out_dict["conf_threshold"]            = CONF_THRESHOLD
    out_dict["conf_threshold_effective"]  = used_threshold
    out_dict["passive_long_pf"]           = _finite(pnl_long.profit_factor)
    out_dict["passive_short_pf"]          = _finite(pnl_short.profit_factor)
    out_dict["passive_pf_baseline"]       = passive_pf
    out_dict["excess_pf"]                 = excess_pf

    # Return a thin shim with .as_dict + the primary attrs we log.
    class _PnL:
        def __init__(self, base, conf, conf_dict, used_thresh, excess):
            self._base = base
            self._conf = conf
            self._conf_dict = conf_dict
            self._used_thresh = used_thresh
            self._excess = excess
        @property
        def used_threshold(self): return self._used_thresh
        @property
        def n_trades(self): return self._base.n_trades
        @property
        def win_rate(self): return self._base.win_rate
        @property
        def profit_factor(self): return self._base.profit_factor
        @property
        def sharpe(self): return self._base.sharpe
        @property
        def max_drawdown(self): return self._base.max_drawdown
        @property
        def excess_pf(self): return self._excess
        @property
        def conf_n_trades(self): return self._conf.n_trades if self._conf else 0
        @property
        def conf_win_rate(self): return self._conf.win_rate if self._conf else float("nan")
        @property
        def conf_profit_factor(self): return self._conf.profit_factor if self._conf else float("nan")
        @property
        def conf_sharpe(self): return self._conf.sharpe if self._conf else float("nan")
        def as_dict(self): return self._conf_dict

    return _PnL(pnl_all, pnl_conf, out_dict, used_threshold, excess_pf)


# ---------------------------------------------------------------------------
# Core per-symbol training function
# ---------------------------------------------------------------------------

def train_symbol(symbol: str, agent: str, create_dir_fn, epochs: int,
                 skip_extract: bool = False,
                 mt5_features: bool = False,
                 mt5_features_root=None) -> dict:
    """
    Full pipeline for one symbol:
      extract -> features -> labels -> train dir/exec/mod -> export ONNX

    If mt5_features=True, skip the Python feature engine entirely and
    load pre-computed feature vectors written by
    ea/MT5_Bot_mk4_FeatureExport.mq5. The MQL5 FeatureEncoder.mqh becomes
    the single source of feature truth (training + live), eliminating
    Python/MQL5 parity drift.

    Returns result dict with val_acc, bars, epochs_run.
    """
    import numpy as np
    from data_pipeline    import run_full_pipeline, fetch_h1_bars, fetch_h4_bars
    from labeler          import compute_direction_labels
    from labeler_exec     import compute_exec_labels
    from labeler_modify   import compute_modify_labels
    from trainer          import HardwareAwareTrainer
    from exporter         import export_direction, export_execution, export_modify
    from models.exec_net  import create_exec_net
    from models.modify_net import create_modify_net
    from hardware_detector import get as get_hw

    hw = get_hw()

    # 1. Data
    if not skip_extract:
        bars_dict = run_full_pipeline(symbols=[symbol], max_bars=hw.max_bars)
    else:
        # Find the largest cached parquet for this symbol. Look for tick-bar
        # caches (HYDRA4_TBARS_<sym>_<n>tpb.parquet from extract_data.py
        # --source ticks) AND M5 bar caches (HYDRA4_FEAT_<sym>_<n>bars.*).
        # Tick-bar caches preferred when present — they're closer to the
        # live tick stream the EA actually sees.
        from config import PARQUET_DIR
        tbar_cached = sorted(PARQUET_DIR.glob(f"HYDRA4_TBARS_{symbol}_*.parquet"),
                              key=lambda p: p.stat().st_size, reverse=True)
        bar_cached  = sorted(PARQUET_DIR.glob(f"HYDRA4_FEAT_{symbol}_*.parquet"),
                              key=lambda p: p.stat().st_size, reverse=True)
        cached = tbar_cached + bar_cached
        if not cached:
            print(f"  [{symbol}] No cached parquet found. "
                  f"Run without --skip-extract first.")
            return {}
        import pandas as pd
        df = pd.read_parquet(cached[0])
        bars_dict = {symbol: df}
        kind = "tick-bars" if cached[0].name.startswith("HYDRA4_TBARS_") else "M5"
        print(f"  [{symbol}] Loaded cache: {cached[0].name}  [{kind}]")

    bars = bars_dict.get(symbol)

    # mk4.4 #3: drop synthesized pre-electronic-trading history.
    if bars is not None and len(bars) > 0:
        from config import MIN_BAR_DATE
        if MIN_BAR_DATE and "time" in bars.columns:
            import pandas as pd
            cutoff = pd.Timestamp(MIN_BAR_DATE, tz="UTC")
            n_before = len(bars)
            bars = bars[bars["time"] >= cutoff].reset_index(drop=True)
            n_after = len(bars)
            if n_after < n_before:
                print(f"  [{symbol}] Dropped {n_before - n_after:,} bars older "
                      f"than {MIN_BAR_DATE} (broker-synthesized).")
    if bars is None or len(bars) == 0:
        print(f"  [{symbol}] ERROR: no bars returned.")
        return {}

    n_bars = len(bars)
    print(f"  [{symbol}] {n_bars:,} M5 bars loaded.")

    # 2. Features
    ps = pip_size(symbol)

    if mt5_features:
        # MT5-canonical feature path: MQL5 FeatureEncoder.mqh produced the
        # binary; we just load it. Skips Python feature_engine entirely.
        from load_mt5_features import load_features as _load_mt5
        from config import FEATURE_DIM_DIR, EXEC_CTX_DIM
        dir_feat, _ = _load_mt5(symbol, root=mt5_features_root,
                                expected_dim=FEATURE_DIM_DIR, mmap=False)
        if len(dir_feat) != n_bars:
            print(f"  [{symbol}] WARN: MT5 features rows={len(dir_feat):,} "
                  f"!= bars rows={n_bars:,} — aligning by truncation.")
            m = min(len(dir_feat), n_bars)
            dir_feat = dir_feat[:m]
            bars = bars.iloc[:m].copy()
            n_bars = m
        # Exec features: dir_feat ++ zero-padded exec context. Microstructure
        # slots and EA-injected position slots are written by FeatureEncoder
        # at inference time, not training time — matching live distribution.
        exec_feat = np.concatenate(
            [dir_feat, np.zeros((n_bars, EXEC_CTX_DIM), dtype=np.float32)],
            axis=1,
        )
        print(f"  [{symbol}] MT5 features: shape={dir_feat.shape}")
    else:
        # Legacy Python feature path. feature_engine.py mirrors
        # FeatureEncoder.mqh; parity drift is the user's risk to manage.
        from feature_engine import build_feature_dataframe
        h1_bars = h4_bars = None
        if not skip_extract:
            h1_bars = fetch_h1_bars(symbol)
            h4_bars = fetch_h4_bars(symbol)
            if h1_bars is not None:
                print(f"  [{symbol}] {len(h1_bars):,} H1 bars loaded.")
            if h4_bars is not None:
                print(f"  [{symbol}] {len(h4_bars):,} H4 bars loaded.")
        dir_feat, exec_feat = build_feature_dataframe(
            bars, symbol, pip_size=ps, h1_df=h1_bars, h4_df=h4_bars)

    dir_labels, _       = compute_direction_labels(bars)
    exec_labels         = compute_exec_labels(bars, dir_labels, pip_size=ps, symbol=symbol)
    mod_labels          = compute_modify_labels(bars, dir_labels,
                                                pip_size=ps,
                                                exec_sl_labels=exec_labels[:, 1])

    n_long = int((dir_labels > 0).sum())
    n_shrt = int((dir_labels < 0).sum())
    print(f"  [{symbol}] Labels: {n_long:,} long / {n_shrt:,} short / "
          f"{int((dir_labels==0).sum()):,} flat")

    trainer = HardwareAwareTrainer(hw)

    # 3. Direction model
    print(f"  [{symbol}] Training direction model...")
    dir_model = create_dir_fn()
    dir_metrics = trainer.train_direction(dir_model, dir_feat, dir_labels,
                                          epochs=epochs, symbol=symbol)
    val_acc = dir_metrics.get("val_acc", 0.0)
    ep_run  = dir_metrics.get("epochs_run", epochs)

    # Production PnL metrics on the chronological val slice. Two views:
    #   "all-bars"  : trade every val bar (population baseline).
    #   "conf-only" : trade only when prob crosses CONF_THRESHOLD (live EA).
    # The conf-thresholded number is what actually goes to live trading.
    excess_pf = None
    try:
        pnl = _evaluate_pnl_on_val(dir_model, bars, dir_feat, dir_labels, ps)
        dir_metrics.update({f"pnl_{k}": v for k, v in pnl.as_dict().items()})
        excess_pf = pnl.excess_pf
        log.info("[%s] Direction val_acc=%.4f  epochs=%d", symbol, val_acc, ep_run)
        log.info("  all-bars : n=%-6d WR=%.3f  PF=%s  Sharpe=%s  MDD=%.4f",
                 pnl.n_trades, pnl.win_rate, pnl.profit_factor, pnl.sharpe,
                 pnl.max_drawdown)
        from config import CONF_THRESHOLD as _cfg_ct
        thresh_label = (f"thresh={pnl.used_threshold:.3f}"
                        if abs(pnl.used_threshold - _cfg_ct) < 1e-6
                        else f"adaptive top-10% (eff thresh={pnl.used_threshold:.3f}; "
                             f"CONF_THRESHOLD={_cfg_ct} fired n=0)")
        log.info("  conf-only: n=%-6d WR=%.3f  PF=%s  Sharpe=%s   [%s]",
                 pnl.conf_n_trades, pnl.conf_win_rate,
                 pnl.conf_profit_factor, pnl.conf_sharpe, thresh_label)
    except Exception as _e:
        log.warning("[%s] PnL eval failed: %s", symbol, _e)
        log.info("[%s] Direction  val_acc=%.4f  epochs=%d", symbol, val_acc, ep_run)

    # 4. Execution model
    print(f"  [{symbol}] Training execution model...")
    exec_model   = create_exec_net()
    exec_metrics = trainer.train_execution(exec_model, exec_feat, exec_labels,
                                           epochs=epochs, symbol=symbol)

    # 5. Modification model
    # ModifyNet expects FEATURE_DIM_MOD = FEATURE_DIM_DIR + 8 dims (the trailing
    # 8 are EA-fed position context: open_pnl, sl_dist, tp_dist, age_bars, etc.,
    # zero at training time). Without padding, BatchNorm dim-mismatches.
    print(f"  [{symbol}] Training modification model...")
    from config import FEATURE_DIM_DIR as _FDIR, FEATURE_DIM_MOD as _FMOD
    pos_pad = np.zeros((dir_feat.shape[0], _FMOD - _FDIR), dtype=np.float32)
    mod_feat = np.concatenate([dir_feat, pos_pad], axis=1)
    mod_model   = create_modify_net()
    mod_metrics = trainer.train_modify(mod_model, mod_feat, mod_labels,
                                       epochs=epochs, symbol=symbol)

    # 6a. Walk-forward summary — informational only, logs median PF and
    #     fold consistency across 5 folds of an "always long" baseline so
    #     the user can compare the trained model's single-split PF against
    #     a multi-fold passive number. Doesn't gate (skill_gate already
    #     handles that), but adds robustness signal to the meta JSON.
    try:
        from validation import walk_forward_pf_summary
        from config import LABEL_FORWARD_BARS
        closes_full = bars["close"].to_numpy(dtype=np.float64)
        H = LABEL_FORWARD_BARS
        fwd_full = np.zeros_like(closes_full)
        fwd_full[:-H] = (closes_full[H:] - closes_full[:-H]) / closes_full[:-H]
        wf = walk_forward_pf_summary(
            features=dir_feat, labels=dir_labels,
            forward_returns=fwd_full, pip=ps, n_folds=5, gap=H,
        )
        log.info("  walk-fwd : median_PF=%.3f  frac_profitable=%.2f  n_folds=%d  (always-long baseline)",
                 wf["median_pf"], wf["frac_profitable"], wf["n_folds"])
        dir_metrics["walk_forward_passive_median_pf"] = wf["median_pf"]
        dir_metrics["walk_forward_passive_frac_prof"] = wf["frac_profitable"]
        dir_metrics["walk_forward_n_folds"]           = wf["n_folds"]
    except Exception as _e:
        log.warning("[%s] walk-forward summary failed: %s", symbol, _e)

    # 6b. Skill gate — refuse to export a model that doesn't beat its own
    #    passive baseline ("always long" / "always short"). On bull-trending
    #    assets (BTC/SPX/Metals) this catches models that are actually just
    #    riding drift; on FX it catches the no-signal-after-cost case. The
    #    gate is bypassable via HYDRA_SKIP_SKILL_GATE=1 for debugging.
    import os as _os
    _bypass = _os.environ.get("HYDRA_SKIP_SKILL_GATE") == "1"
    if excess_pf is not None and excess_pf < 0.30 and not _bypass:
        print(f"  [{symbol}] SKIPPED ONNX export — excess_PF={excess_pf:+.3f} "
              f"< 0.30 (model doesn't beat passive baseline). "
              f"Set HYDRA_SKIP_SKILL_GATE=1 to bypass.")
        return {
            "symbol": symbol, "agent": agent, "val_acc": val_acc,
            "bars": n_bars, "epochs_run": ep_run, "onnx_ok": False,
            "skipped_reason": "skill_gate", "excess_pf": excess_pf,
        }

    # 7. Export ONNX
    print(f"  [{symbol}] Exporting ONNX...")
    ok_dir  = export_direction(dir_model,  agent, symbol, train_metrics=dir_metrics)
    ok_exec = export_execution(exec_model, agent, symbol, train_metrics=exec_metrics)
    ok_mod  = export_modify(mod_model,     agent, symbol, train_metrics=mod_metrics)

    if ok_dir and ok_exec and ok_mod:
        print(f"  [{symbol}] ONNX export OK.")
    else:
        print(f"  [{symbol}] WARNING: some ONNX exports failed — check logs.")

    return {
        "symbol":    symbol,
        "agent":     agent,
        "val_acc":   val_acc,
        "bars":      n_bars,
        "epochs_run": ep_run,
        "onnx_ok":   ok_dir and ok_exec and ok_mod,
    }


# ---------------------------------------------------------------------------
# Multi-symbol runner used by each agent script
# ---------------------------------------------------------------------------

def run_agent(agent: str, symbols: List[str], create_dir_fn,
              epochs: int, skip_extract: bool, seed: int = 42,
              mt5_features: bool = False, mt5_features_root=None):
    """Connect to MT5, train all symbols for one agent, print summary.

    mt5_features=True: load pre-computed feature vectors from MQL5
    exporter (binary files in mt5_features_root, default MT5_COMMON_DIR);
    Python feature_engine is bypassed.
    """
    import os
    from data_pipeline    import connect, disconnect
    from hardware_detector import get as get_hw
    from _seeding         import set_global_seed

    set_global_seed(seed)
    hw = get_hw()
    print(f"\n{'='*60}")
    print(f"  HYDRA mk4 — {agent} agent")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Device  : {hw.device.upper()}  tier={hw.tier}  "
          f"batch={hw.batch_size:,}  max_bars={hw.max_bars:,}")
    print(f"  Epochs  : {epochs}")
    print(f"{'='*60}\n")

    if not skip_extract:
        print("Connecting to MT5...")
        terminal_path = os.environ.get("MT5_TERMINAL_PATH", "")
        if not connect(path=terminal_path):
            sys.exit(1)
        print("MT5 connected.\n")

    results = []
    wall_start = time.time()

    for symbol in symbols:
        t0 = time.time()
        try:
            r = train_symbol(symbol, agent, create_dir_fn, epochs, skip_extract,
                             mt5_features=mt5_features,
                             mt5_features_root=mt5_features_root)
            r["elapsed"] = time.time() - t0
            results.append(r)
        except Exception as e:
            print(f"  [{symbol}] FAILED: {e}")
            log.exception("train_symbol failed for %s", symbol)
            results.append({"symbol": symbol, "agent": agent,
                            "val_acc": 0.0, "onnx_ok": False,
                            "elapsed": time.time() - t0})

    if not skip_extract:
        disconnect()

    # Summary
    wall = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"  {agent} — results")
    print(f"{'='*60}")
    print(f"  {'Symbol':<10} {'ValAcc':>8}  {'Bars':>10}  {'Epochs':>7}  {'ONNX':>6}  Time")
    print(f"  {'-'*60}")
    all_pass = True
    for r in results:
        sym  = r.get("symbol", "?")
        acc  = r.get("val_acc", 0.0)
        bars = r.get("bars", 0)
        ep   = r.get("epochs_run", 0)
        ok   = r.get("onnx_ok", False)
        sec  = r.get("elapsed", 0)
        flag = "OK" if ok else "FAIL"
        mark = "✓" if acc >= 0.60 else "!"
        if acc < 0.60:
            all_pass = False
        print(f"  {sym:<10} {acc:>8.4f}  {bars:>10,}  {ep:>7}  {flag:>6}  "
              f"{int(sec//60)}m{int(sec%60):02d}s  {mark}")

    h = int(wall // 3600); m = int((wall % 3600) // 60); s = int(wall % 60)
    print(f"\n  Total time: {h}h {m}m {s}s")
    if all_pass:
        print("  All models trained. EA will hot-reload within 30 seconds.")
    else:
        print("  Some models have low val_acc (< 60%).")
        print("  Try: more bars (ensure MT5 has enough history), more epochs,")
        print("  or run `python eval_harness.py <symbol>` for walk-forward diagnostics.")
    print(f"{'='*60}\n")

    return results
