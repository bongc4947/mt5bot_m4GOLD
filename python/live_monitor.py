"""
live_monitor.py — Watch HYDRA4_closed_trades.csv for real trade outcomes,
compute per-symbol win rate and PnL, trigger retrains, write monitor JSON.

Run: python live_monitor.py

Previous version read from the signal log (HYDRA4_signals.csv) which has no
'pips' column — outcomes are unknown at signal time.  This version reads from
HYDRA4_closed_trades.csv written by RunLogger.LogClosedTrade() on every
OnTradeTransaction(DEAL_ENTRY_OUT) event.
"""

import json
import logging
import sys
import time
import threading
import datetime as dt
from pathlib import Path
from collections import deque
from typing import Dict, Optional, Deque, Set

from config import (
    ALL_SYMBOLS, AGENT_SYMBOL_MAP,
    closed_trades_log_path, monitor_json_path, retrain_flag_path,
    progress_json_path,
    MONITOR_INTERVAL_SEC, WIN_RATE_WINDOW, WIN_RATE_DROP_THRESH,
    MODEL_AGE_RETRAIN_HRS, meta_path,
)
from data_pipeline import parse_closed_trades_log
try:
    from models.meta_controller import MetaController
except ImportError:
    MetaController = None  # type: ignore  # torch not installed — monitor runs without it

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("live_monitor.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol → agent lookup
# ---------------------------------------------------------------------------

def _symbol_to_agent(symbol: str) -> str:
    for agent, syms in AGENT_SYMBOL_MAP.items():
        if symbol in syms:
            return agent
    return "PRISM"


# ---------------------------------------------------------------------------
# Per-symbol rolling metrics  (computed from REAL closed trade outcomes)
# ---------------------------------------------------------------------------

class SymbolMetrics:
    def __init__(self, symbol: str, window: int = WIN_RATE_WINDOW):
        self.symbol    = symbol
        self.window    = window
        self._pips:        Deque[float] = deque(maxlen=window)
        self._pnl:         Deque[float] = deque(maxlen=window)
        self._confidences: Deque[float] = deque(maxlen=window)
        self.val_acc_at_train: float = 0.0
        self.last_trade_ts: Optional[dt.datetime] = None

    def push(self, pips: float, pnl: float, confidence: float = 0.5):
        self._pips.append(pips)
        self._pnl.append(pnl)
        self._confidences.append(confidence)
        self.last_trade_ts = dt.datetime.now(dt.timezone.utc)

    # ---- computed properties ----

    @property
    def win_rate(self) -> float:
        if not self._pips:
            return 0.0          # 0 = unknown, not 0.5 — caller should check n_trades
        wins = sum(1 for p in self._pips if p > 0)
        return wins / len(self._pips)

    @property
    def avg_pips(self) -> float:
        if not self._pips:
            return 0.0
        return sum(self._pips) / len(self._pips)

    @property
    def total_pnl(self) -> float:
        return sum(self._pnl)

    @property
    def profit_factor(self) -> float:
        gross_win  = sum(p for p in self._pnl if p > 0)
        gross_loss = sum(-p for p in self._pnl if p < 0)
        if gross_loss < 1e-8:
            return float("inf") if gross_win > 0 else 1.0
        return gross_win / gross_loss

    @property
    def avg_confidence(self) -> float:
        if not self._confidences:
            return 0.0
        return sum(self._confidences) / len(self._confidences)

    @property
    def n_trades(self) -> int:
        return len(self._pips)

    def needs_retrain(self) -> Optional[str]:
        """Returns a reason string if a retrain should be triggered, else None."""
        if self.n_trades < self.window:
            return None
        # Only compare if we have a real val_acc baseline
        if self.val_acc_at_train < 0.50:
            return None
        if self.win_rate < self.val_acc_at_train - WIN_RATE_DROP_THRESH:
            return (f"win_rate={self.win_rate:.3f} < "
                    f"val_acc({self.val_acc_at_train:.3f}) - {WIN_RATE_DROP_THRESH}")
        # Profitability trigger: retrain if live PF < 1.0 (losing money) and
        # we have at least a full window of trades.
        pf = self.profit_factor
        if pf < 1.0:
            return f"profit_factor={pf:.3f} < 1.0  (unprofitable over {self.n_trades} trades)"
        return None


# ---------------------------------------------------------------------------
# Model meta helpers
# ---------------------------------------------------------------------------

def _model_age_hours(symbol: str) -> float:
    agent = _symbol_to_agent(symbol)
    p = meta_path(agent, symbol)
    if not p.exists():
        return float("inf")
    try:
        with open(p) as f:
            meta = json.load(f)
        trained_at = dt.datetime.fromisoformat(
            meta.get("trained_at", "2000-01-01T00:00:00Z").rstrip("Z")
        )
        return (dt.datetime.now(dt.timezone.utc) - trained_at.replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600.0
    except Exception:
        return float("inf")


def _load_val_acc(symbol: str) -> float:
    agent = _symbol_to_agent(symbol)
    p = meta_path(agent, symbol)
    if not p.exists():
        return 0.0          # 0 = no model, not 0.5
    try:
        with open(p) as f:
            return float(json.load(f).get("val_acc", 0.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Retrain trigger
# ---------------------------------------------------------------------------

def _symbol_to_agent_flag(symbol: str) -> str:
    """Map symbol to train_all --agents flag value."""
    agent = _symbol_to_agent(symbol)
    return {
        "PRISM": "prism",
        "GNN":   "gnn",
        "APEX":  "apex",
        "CE":    "ce",
    }.get(agent, "prism")


# Track active retrain threads and cooldown per symbol
_retraining: Set[str] = set()
_retrain_lock = threading.Lock()
_last_retrain_at: Dict[str, dt.datetime] = {}
RETRAIN_COOLDOWN_HRS = 2.0   # don't re-trigger same symbol within 2 hours


def _retrain_worker(symbol: str, agent_flag: str):
    """Background thread: invoke train.py without blocking the monitor loop."""
    flag = retrain_flag_path(symbol)
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "train.py", "all",
             "--skip-extract",
             "--agents", agent_flag],
            cwd=str(Path(__file__).parent),
            timeout=3600,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            log.info("Retrain succeeded for %s (agent=%s)", symbol, agent_flag)
        else:
            log.error("Retrain failed for %s:\n%s", symbol, result.stderr[-2000:])
    except FileNotFoundError:
        log.warning("train.py not found — skipping retrain for %s", symbol)
    except Exception as e:
        log.error("Retrain error for %s: %s", symbol, e)
    finally:
        flag.unlink(missing_ok=True)
        with _retrain_lock:
            _retraining.discard(symbol)
            _last_retrain_at[symbol] = dt.datetime.now(dt.timezone.utc)


def trigger_retrain(symbol: str, reason: str):
    """Fire-and-forget retrain in a daemon thread; honours per-symbol cooldown."""
    now = dt.datetime.now(dt.timezone.utc)
    with _retrain_lock:
        # Skip if already retraining this symbol
        if symbol in _retraining:
            log.debug("Retrain %s skipped — already in progress", symbol)
            return
        # Skip if retrained recently
        last = _last_retrain_at.get(symbol)
        if last and (now - last).total_seconds() < RETRAIN_COOLDOWN_HRS * 3600:
            log.debug("Retrain %s skipped — cooldown (last=%.1fh ago)",
                      symbol, (now - last).total_seconds() / 3600)
            return
        _retraining.add(symbol)

    log.info("RETRAIN TRIGGERED  %s: %s", symbol, reason)
    agent_flag = _symbol_to_agent_flag(symbol)
    t = threading.Thread(target=_retrain_worker, args=(symbol, agent_flag),
                         daemon=True, name=f"retrain-{symbol}")
    t.start()


# ---------------------------------------------------------------------------
# Monitor JSON writer
# ---------------------------------------------------------------------------

def _write_monitor(metrics_map: Dict[str, SymbolMetrics],
                   meta_ctrl):
    out = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat() + "Z",
        "symbols": {},
        "meta_weights": meta_ctrl.weights_dict() if meta_ctrl is not None else {},
    }
    for sym, m in metrics_map.items():
        age = _model_age_hours(sym)
        out["symbols"][sym] = {
            "n_trades":       m.n_trades,
            "win_rate":       round(m.win_rate, 4),
            "avg_pips":       round(m.avg_pips, 2),
            "total_pnl":      round(m.total_pnl, 2),
            "profit_factor":  round(m.profit_factor, 3),
            "val_acc":        round(m.val_acc_at_train, 4),
            "avg_confidence": round(m.avg_confidence, 4),
            "model_age_hrs":  round(age, 2),
            "last_trade":     str(m.last_trade_ts) if m.last_trade_ts else "",
        }

    p = monitor_json_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        log.warning("Could not write monitor JSON: %s", e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    log.info("HYDRA mk4 LiveMonitor starting  interval=%ds  window=%d trades",
             MONITOR_INTERVAL_SEC, WIN_RATE_WINDOW)
    log.info("Reading outcomes from: %s", closed_trades_log_path())

    metrics_map: Dict[str, SymbolMetrics] = {s: SymbolMetrics(s) for s in ALL_SYMBOLS}
    meta_ctrl  = MetaController() if MetaController is not None else None
    seen_rows  = 0    # total rows processed so far from the closed trades CSV

    # Load current val_acc from existing models on startup
    for sym, m in metrics_map.items():
        m.val_acc_at_train = _load_val_acc(sym)
        if m.val_acc_at_train > 0:
            log.info("  %s  val_acc=%.4f  (loaded from meta.json)", sym, m.val_acc_at_train)

    while True:
        try:
            # ----------------------------------------------------------------
            # Read new closed trade rows
            # ----------------------------------------------------------------
            df = parse_closed_trades_log()
            new_rows = len(df) - seen_rows if len(df) > seen_rows else 0

            if new_rows > 0:
                new_df = df.iloc[-new_rows:]
                for _, row in new_df.iterrows():
                    sym  = str(row.get("symbol", "")).strip()
                    pips = float(row.get("pips",   0.0) or 0.0)
                    pnl  = float(row.get("pnl_usd", 0.0) or 0.0)
                    conf = float(row.get("confidence", 0.0) or 0.0)

                    if sym not in metrics_map:
                        continue

                    metrics_map[sym].push(pips, pnl, conf)

                    # Feed MetaController (skipped if torch not installed)
                    if meta_ctrl is not None:
                        agent = _symbol_to_agent(sym)
                        try:
                            from models.meta_controller import AGENT_NAMES
                            ag_id = AGENT_NAMES.index(agent) if agent in AGENT_NAMES else 0
                            meta_ctrl.record_pnl(ag_id, pnl)
                        except Exception:
                            pass

                    outcome = "WIN" if pips > 0 else "LOSS"
                    log.info("  Trade closed  %-12s  %s  pips=%+.1f  pnl=%+.2f",
                             sym, outcome, pips, pnl)

                if meta_ctrl is not None:
                    try:
                        meta_ctrl.rebalance()
                    except Exception:
                        pass
                seen_rows = len(df)

                # Log running summary
                for sym, m in metrics_map.items():
                    if m.n_trades > 0:
                        log.info("  %-12s  n=%d  win=%.1f%%  avg_pips=%+.1f  PF=%.2f",
                                 sym, m.n_trades,
                                 m.win_rate * 100, m.avg_pips, m.profit_factor)

            # ----------------------------------------------------------------
            # Manual retrain flags
            # ----------------------------------------------------------------
            for sym in ALL_SYMBOLS:
                if retrain_flag_path(sym).exists():
                    trigger_retrain(sym, "manual_flag")
                    metrics_map[sym].val_acc_at_train = _load_val_acc(sym)

            # ----------------------------------------------------------------
            # Win-rate decay trigger
            # ----------------------------------------------------------------
            for sym, m in metrics_map.items():
                reason = m.needs_retrain()
                if reason:
                    trigger_retrain(sym, reason)
                    m._pips.clear()
                    m._pnl.clear()
                    m.val_acc_at_train = _load_val_acc(sym)
                    continue

                # Model age trigger (only if no recent trades — active symbols
                # retrain via win-rate decay above).  Cooldown prevents storm.
                age = _model_age_hours(sym)
                if age > MODEL_AGE_RETRAIN_HRS and m.n_trades == 0:
                    trigger_retrain(sym, f"model_age={age:.1f}h  no_trades")
                    metrics_map[sym].val_acc_at_train = _load_val_acc(sym)

        except Exception as e:
            log.exception("Monitor loop error: %s", e)

        finally:
            # Always write monitor JSON — even on exception — so the dashboard
            # and EA receive fresh data every interval regardless of trade parse errors.
            try:
                _write_monitor(metrics_map, meta_ctrl)
            except Exception as e:
                log.warning("_write_monitor failed: %s", e)

        time.sleep(MONITOR_INTERVAL_SEC)


if __name__ == "__main__":
    run()
