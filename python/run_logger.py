"""
run_logger.py — Structured run logging for HYDRA mk4 training pipeline.

Writes two files per training run:
  data/logs/runs/HYDRA4_run_<timestamp>.json   — full detail (one per run)
  data/logs/HYDRA4_train_history.csv           — cumulative one-row-per-symbol ledger

The JSON log is the primary artifact for analysis. The CSV is for quick trend
comparisons (val_acc over time, epoch counts, label distributions).

Usage:
    from run_logger import RunLogger
    rl = RunLogger()
    rl.start_run(agent="PRISM", symbols=["EURUSD","GBPUSD"])
    rl.log_symbol(symbol, agent, dir_metrics, exec_metrics, mod_metrics,
                  n_bars, label_counts, h1_bars, h4_bars, feature_dim)
    rl.finish_run(total_elapsed_sec)
"""

import csv
import json
import logging
import os
import socket
import sys
import time
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import LOGS_DIR, HYDRA_VERSION, BASE_DIR

log = logging.getLogger(__name__)

RUNS_DIR     = LOGS_DIR / "runs"
HISTORY_CSV  = LOGS_DIR / "HYDRA4_train_history.csv"

_CSV_FIELDS = [
    "run_id", "started_at", "agent", "symbol",
    "n_bars_m5", "n_bars_h1", "n_bars_h4",
    "feature_dim", "exec_feature_dim", "mod_feature_dim",
    "label_long", "label_short", "label_flat",
    "dir_val_acc", "dir_epochs", "dir_best_epoch",
    "exec_val_loss", "exec_epochs",
    "mod_val_loss", "mod_epochs",
    "onnx_ok", "elapsed_sec",
    "hydra_version", "python_version", "hostname",
]


class RunLogger:
    """Accumulates per-symbol results for one training run and persists them."""

    def __init__(self):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self._run_id    = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._started   = time.time()
        self._agent     = ""
        self._symbols   = []
        self._results   = []   # list of per-symbol dicts
        self._run_path  = RUNS_DIR / f"HYDRA4_run_{self._run_id}.json"
        self._hostname  = socket.gethostname()
        self._pyver     = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # ------------------------------------------------------------------
    def start_run(self, agent: str, symbols: List[str]):
        self._agent   = agent
        self._symbols = symbols
        log.info("RunLogger: run_id=%s  agent=%s  symbols=%s",
                 self._run_id, agent, symbols)

    # ------------------------------------------------------------------
    def log_symbol(self,
                   symbol: str,
                   agent: str,
                   dir_metrics:  Dict[str, Any],
                   exec_metrics: Dict[str, Any],
                   mod_metrics:  Dict[str, Any],
                   n_bars_m5:    int,
                   label_counts: Dict[str, int],   # {"long":N,"short":N,"flat":N}
                   n_bars_h1:    int = 0,
                   n_bars_h4:    int = 0,
                   feature_dim:  int = 0,
                   exec_feature_dim: int = 0,
                   mod_feature_dim:  int = 0,
                   onnx_ok:      bool = False,
                   elapsed_sec:  float = 0.0):

        entry = {
            "run_id":           self._run_id,
            "started_at":       dt.datetime.now(dt.timezone.utc).isoformat() + "Z",
            "agent":            agent,
            "symbol":           symbol,
            "hydra_version":    HYDRA_VERSION,
            "python_version":   self._pyver,
            "hostname":         self._hostname,
            # Data
            "n_bars_m5":        n_bars_m5,
            "n_bars_h1":        n_bars_h1,
            "n_bars_h4":        n_bars_h4,
            "feature_dim":      feature_dim,
            "exec_feature_dim": exec_feature_dim,
            "mod_feature_dim":  mod_feature_dim,
            # Labels
            "label_long":       label_counts.get("long",  0),
            "label_short":      label_counts.get("short", 0),
            "label_flat":       label_counts.get("flat",  0),
            "label_total":      label_counts.get("long", 0) + label_counts.get("short", 0) + label_counts.get("flat", 0),
            # Direction model
            "dir_val_acc":      float(dir_metrics.get("val_acc",       0.0)),
            "dir_val_loss":     float(dir_metrics.get("val_loss",      0.0)),
            "dir_train_loss":   float(dir_metrics.get("train_loss",    0.0)),
            "dir_epochs":       int(dir_metrics.get("epochs_run",      0)),
            "dir_best_epoch":   int(dir_metrics.get("best_epoch",      0)),
            "dir_train_bars":   int(dir_metrics.get("trained_bars",    0)),
            # Epoch-by-epoch history (for plotting)
            "dir_epoch_history": dir_metrics.get("epoch_history", []),
            # Execution model
            "exec_val_loss":    float(exec_metrics.get("val_loss",     0.0)),
            "exec_epochs":      int(exec_metrics.get("epochs_run",     0)),
            "exec_timing_acc":  float(exec_metrics.get("val_timing_acc",  0.0)),
            "exec_sl_mae":      float(exec_metrics.get("val_sl_mae_pips", 0.0)),
            "exec_tp_mae":      float(exec_metrics.get("val_tp_mae_pips", 0.0)),
            # Modification model
            "mod_val_loss":     float(mod_metrics.get("val_loss",      0.0)),
            "mod_epochs":       int(mod_metrics.get("epochs_run",      0)),
            "mod_be_acc":       float(mod_metrics.get("val_be_acc",    0.0)),
            "mod_close_acc":    float(mod_metrics.get("val_close_acc", 0.0)),
            # Result
            "onnx_ok":          onnx_ok,
            "elapsed_sec":      round(elapsed_sec, 1),
        }

        self._results.append(entry)
        self._flush_json()
        self._append_csv(entry)

        log.info("RunLogger: logged %s/%s  val_acc=%.4f  onnx=%s",
                 agent, symbol, entry["dir_val_acc"], onnx_ok)

    # ------------------------------------------------------------------
    def finish_run(self, total_elapsed_sec: float = 0.0):
        summary = {
            "run_id":          self._run_id,
            "agent":           self._agent,
            "symbols":         self._symbols,
            "n_symbols":       len(self._results),
            "total_elapsed":   round(total_elapsed_sec, 1),
            "all_onnx_ok":     all(r["onnx_ok"] for r in self._results),
            "mean_val_acc":    round(
                sum(r["dir_val_acc"] for r in self._results) / max(len(self._results), 1), 4),
            "results":         self._results,
        }

        with open(self._run_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        log.info("RunLogger: run complete → %s", self._run_path.name)
        print(f"\n  Run log saved: {self._run_path}")
        print(f"  History CSV  : {HISTORY_CSV}")

    # ------------------------------------------------------------------
    def _flush_json(self):
        """Write partial results immediately so crashes don't lose data."""
        partial = {
            "run_id":    self._run_id,
            "agent":     self._agent,
            "partial":   True,
            "results":   self._results,
        }
        with open(self._run_path, "w", encoding="utf-8") as f:
            json.dump(partial, f, indent=2)

    # ------------------------------------------------------------------
    def _append_csv(self, entry: dict):
        write_header = not HISTORY_CSV.exists()
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(entry)
