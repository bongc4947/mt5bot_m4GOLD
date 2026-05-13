"""
hydra_dashboard.py — HYDRA mk4 Master Control Dashboard

Single entry point for the full Python pipeline:
  - Hardware detection & status
  - Model status per symbol (val_acc, age, ONNX present)
  - Interactive train menu (all symbols, selected symbols, per-agent)
  - Data extraction
  - Live monitor (win-rate decay / regime drift / retrain triggers)
  - Real-time training progress with epoch-by-epoch display

Usage:
    python hydra_dashboard.py

No arguments needed — all controls are interactive.
"""

import json
import logging
import os
import sys
import time
import threading
import datetime as dt
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Rich imports — must be first so logging doesn't clobber the display
# ---------------------------------------------------------------------------
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt, Confirm
from rich import box

console = Console()

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HYDRA_VERSION, ALL_SYMBOLS, AGENT_SYMBOL_MAP,
    MT5_COMMON_DIR, ONNX_OUTPUT_DIR,
    FEATURE_DIM_DIR, FEATURE_DIM_EXEC, FEATURE_DIM_MOD,
    EPOCHS, RETRAIN_EPOCHS, PATIENCE, LR,
    MONITOR_INTERVAL_SEC, WIN_RATE_DROP_THRESH, MODEL_AGE_RETRAIN_HRS,
    meta_path, onnx_det_path, onnx_mc_path, onnx_exec_path, onnx_modify_path,
    signal_log_path, monitor_json_path, retrain_flag_path, progress_json_path,
)
from hardware_detector import get as get_hw, detect as detect_hw

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("hydra_dashboard.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER = f"[bold cyan]HYDRA mk4[/bold cyan]  [dim]v{HYDRA_VERSION}[/dim]"
AGENT_COLORS = {
    "PRISM": "cyan",
    "GNN":   "yellow",
    "APEX":  "magenta",
    "CE":    "green",
}

# ---------------------------------------------------------------------------
# Symbol → agent lookup
# ---------------------------------------------------------------------------

def _sym_agent(symbol: str) -> str:
    for agent, syms in AGENT_SYMBOL_MAP.items():
        if symbol in syms:
            return agent
    return "PRISM"


# ---------------------------------------------------------------------------
# Model status helpers
# ---------------------------------------------------------------------------

def _model_status(symbol: str) -> Dict:
    """Return dict with val_acc, age_hrs, files_ok for a symbol."""
    agent = _sym_agent(symbol)
    meta_p = meta_path(agent, symbol)

    files_ok = all(p.exists() for p in [
        onnx_det_path(agent, symbol),
        onnx_mc_path(agent, symbol),
        onnx_exec_path(agent, symbol),
        onnx_modify_path(agent, symbol),
    ])

    val_acc       = 0.0
    profit_factor = 0.0
    win_rate      = 0.0
    age_hrs       = float("inf")
    trained_at    = ""

    if meta_p.exists():
        try:
            with open(meta_p) as f:
                meta = json.load(f)
            val_acc       = float(meta.get("val_acc", 0.0))
            profit_factor = float(meta.get("profit_factor", 0.0))
            win_rate      = float(meta.get("win_rate", 0.0))
            trained_at    = meta.get("trained_at", "")
            if trained_at:
                ts = dt.datetime.fromisoformat(trained_at.rstrip("Z"))
                age_hrs = (dt.datetime.now(dt.timezone.utc) - ts.replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600.0
        except Exception:
            pass

    return {
        "agent":         agent,
        "files_ok":      files_ok,
        "val_acc":       val_acc,
        "profit_factor": profit_factor,
        "win_rate":      win_rate,
        "age_hrs":       age_hrs,
        "trained_at":    trained_at,
    }


def _read_progress() -> Optional[Dict]:
    """Read trainer progress.json written by the training loop."""
    p = progress_json_path()
    try:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Display panels
# ---------------------------------------------------------------------------

def _hardware_panel() -> Panel:
    hw = get_hw()
    tier_colors = {
        "enterprise": "bold green",
        "mid":        "green",
        "low":        "yellow",
        "minimum":    "red",
    }
    color = tier_colors.get(hw.tier, "white")

    lines = [
        f"  Device : [bold]{hw.device.upper()}[/bold]",
        f"  Tier   : [{color}]{hw.tier.upper()}[/{color}]",
        f"  RAM    : {hw.ram_gb:.1f} GB",
        f"  VRAM   : {hw.vram_gb:.1f} GB" if hw.vram_gb > 0 else "  VRAM   : N/A (CPU)",
        f"  AMP    : {'[green]ON[/green]' if hw.amp else '[dim]off[/dim]'}",
        f"  Batch  : {hw.batch_size:,}",
        f"  MaxBars: [bold]{hw.max_bars:,}[/bold]",
        f"  Workers: {hw.workers}",
    ]
    return Panel("\n".join(lines), title="[bold]Hardware[/bold]",
                 border_style="cyan", padding=(0, 1))


def _model_status_table() -> Table:
    tbl = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="dim",
        expand=True,
    )
    tbl.add_column("Symbol",   style="bold", width=10)
    tbl.add_column("Agent",    width=8)
    tbl.add_column("Val Acc",  justify="center", width=9)
    tbl.add_column("PF",       justify="center", width=7)
    tbl.add_column("WR",       justify="center", width=7)
    tbl.add_column("Age",      justify="right",  width=9)
    tbl.add_column("ONNX",     justify="center", width=8)
    tbl.add_column("Status",   width=16)

    for sym in ALL_SYMBOLS:
        s = _model_status(sym)
        agent_color = AGENT_COLORS.get(s["agent"], "white")

        # Val acc coloring
        acc = s["val_acc"]
        if acc >= 0.70:
            acc_str = f"[green]{acc:.1%}[/green]"
        elif acc >= 0.58:
            acc_str = f"[yellow]{acc:.1%}[/yellow]"
        elif acc > 0:
            acc_str = f"[red]{acc:.1%}[/red]"
        else:
            acc_str = "[dim]---[/dim]"

        # Profit factor coloring
        pf = s["profit_factor"]
        if pf >= 2.0:
            pf_str = f"[green]{pf:.2f}[/green]"
        elif pf >= 1.0:
            pf_str = f"[yellow]{pf:.2f}[/yellow]"
        elif pf > 0:
            pf_str = f"[red]{pf:.2f}[/red]"
        else:
            pf_str = "[dim]---[/dim]"

        # Win rate coloring
        wr = s["win_rate"]
        wr_str = f"{wr:.1%}" if wr > 0 else "[dim]---[/dim]"

        # Age coloring
        age = s["age_hrs"]
        if age == float("inf"):
            age_str = "[dim]never[/dim]"
        elif age < 1:
            age_str = f"[green]{age*60:.0f}m[/green]"
        elif age < MODEL_AGE_RETRAIN_HRS:
            age_str = f"[yellow]{age:.1f}h[/yellow]"
        else:
            age_str = f"[red]{age:.1f}h[/red]"

        onnx_str = "[green]OK[/green]" if s["files_ok"] else "[red]MISSING[/red]"

        # Status text
        if not s["files_ok"] and acc == 0:
            status_str = "[dim]Not trained[/dim]"
        elif not s["files_ok"]:
            status_str = "[yellow]ONNX missing[/yellow]"
        elif age > MODEL_AGE_RETRAIN_HRS:
            status_str = "[yellow]Stale[/yellow]"
        elif acc < 0.58:
            status_str = "[red]Low accuracy[/red]"
        elif pf > 0 and pf < 1.0:
            status_str = "[red]Unprofitable[/red]"
        else:
            status_str = "[green]Ready[/green]"

        tbl.add_row(
            sym,
            f"[{agent_color}]{s['agent']}[/{agent_color}]",
            acc_str,
            pf_str,
            wr_str,
            age_str,
            onnx_str,
            status_str,
        )

    return tbl


def _monitor_panel() -> Panel:
    """Read monitor.json and render current live stats."""
    p = monitor_json_path()
    if not p.exists():
        return Panel("[dim]monitor.json not found — run Live Monitor first[/dim]",
                     title="[bold]Live Monitor[/bold]", border_style="dim")

    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return Panel("[red]Error reading monitor.json[/red]",
                     title="[bold]Live Monitor[/bold]", border_style="red")

    ts = data.get("timestamp", "")
    syms = data.get("symbols", {})

    tbl = Table(box=box.MINIMAL, show_header=True,
                header_style="bold", expand=True, padding=(0, 1))
    tbl.add_column("Symbol", width=10)
    tbl.add_column("WinRate", justify="center", width=9)
    tbl.add_column("PF",      justify="center", width=7)
    tbl.add_column("ValAcc",  justify="center", width=9)
    tbl.add_column("Trades",  justify="right",  width=8)
    tbl.add_column("Conf",    justify="center", width=8)
    tbl.add_column("Age",     justify="right",  width=8)

    for sym, m in syms.items():
        wr   = m.get("win_rate", 0)
        pf   = m.get("profit_factor", 0)
        va   = m.get("val_acc", 0)
        n    = m.get("n_trades", 0)
        conf = m.get("avg_confidence", 0)
        age  = m.get("model_age_hrs", 0)

        wr_color = "green" if wr >= va - WIN_RATE_DROP_THRESH else "red"
        pf_color = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
        tbl.add_row(
            sym,
            f"[{wr_color}]{wr:.1%}[/{wr_color}]",
            f"[{pf_color}]{pf:.2f}[/{pf_color}]" if n > 0 else "[dim]---[/dim]",
            f"{va:.1%}",
            str(n),
            f"{conf:.2f}",
            f"{age:.1f}h",
        )

    age_str = f" — updated {ts[:19].replace('T', ' ')} UTC" if ts else ""
    return Panel(tbl, title=f"[bold]Live Monitor[/bold][dim]{age_str}[/dim]",
                 border_style="blue")


# ---------------------------------------------------------------------------
# Menu helpers
# ---------------------------------------------------------------------------

def _print_header():
    console.print()
    console.rule(HEADER, style="cyan")
    console.print()


def _print_main_menu():
    menu = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    menu.add_column("Key",   style="bold cyan", width=5)
    menu.add_column("Action", style="white")

    rows = [
        ("1", "Train ALL symbols (full pipeline)"),
        ("2", "Train SELECTED symbols"),
        ("3", "Train by AGENT  (PRISM / GNN / APEX / CE)"),
        ("4", "Extract data from MT5 only"),
        ("5", "Refresh model status"),
        ("6", "Start LIVE MONITOR  (background watchdog)"),
        ("7", "Stop live monitor"),
        ("8", "Hardware info"),
        ("9", "Set MT5 Common Files path"),
        ("Q", "Quit"),
    ]
    for k, a in rows:
        menu.add_row(k, a)

    console.print(Panel(menu, title="[bold]Main Menu[/bold]",
                        border_style="cyan", padding=(0, 1)))


def _pick_symbols() -> List[str]:
    """Interactive symbol picker."""
    console.print("\n[bold]Available symbols:[/bold]")
    indexed = list(ALL_SYMBOLS)
    for i, sym in enumerate(indexed, 1):
        agent = _sym_agent(sym)
        color = AGENT_COLORS.get(agent, "white")
        console.print(f"  [{color}]{i:2}[/{color}] {sym}  [dim]({agent})[/dim]")

    console.print("\nEnter numbers separated by commas (e.g. 1,3,5)  or [bold]A[/bold] for all:")
    raw = Prompt.ask("[cyan]>[/cyan]").strip()
    if raw.upper() == "A":
        return indexed
    selected = []
    for part in raw.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(indexed):
                selected.append(indexed[idx])
        except ValueError:
            pass
    if not selected:
        console.print("[red]No valid selection — returning to menu.[/red]")
    return selected


def _pick_agent() -> Optional[List[str]]:
    """Let user choose an agent; returns list of its symbols."""
    agents = list(AGENT_SYMBOL_MAP.keys())
    console.print("\n[bold]Agents:[/bold]")
    for i, ag in enumerate(agents, 1):
        color = AGENT_COLORS.get(ag, "white")
        syms  = ", ".join(AGENT_SYMBOL_MAP[ag])
        console.print(f"  [{color}]{i}[/{color}] {ag}  [dim]{syms}[/dim]")
    raw = Prompt.ask("[cyan]>[/cyan]").strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(agents):
            return AGENT_SYMBOL_MAP[agents[idx]]
    except ValueError:
        pass
    console.print("[red]Invalid choice.[/red]")
    return None


# ---------------------------------------------------------------------------
# Training runner with live display
# ---------------------------------------------------------------------------

def _run_training(symbols: List[str], retrain: bool = False):
    """
    Train direction + exec + modify models for each symbol in sequence.
    Shows live Rich progress with per-epoch updates.
    """
    if not symbols:
        console.print("[red]No symbols selected.[/red]")
        return

    # Confirm
    sym_list = ", ".join(symbols)
    mode_str = "RETRAIN" if retrain else "TRAIN"
    console.print(f"\n[bold]{mode_str}[/bold] {len(symbols)} symbol(s): [cyan]{sym_list}[/cyan]")
    epochs_to_use = RETRAIN_EPOCHS if retrain else EPOCHS
    console.print(f"Epochs : {epochs_to_use}   Patience : {PATIENCE}   LR : {LR}")
    if not Confirm.ask("Proceed?", default=True):
        return

    hw = get_hw()
    console.print(f"\nDevice : [bold]{hw.device.upper()}[/bold]  "
                  f"Tier : [bold]{hw.tier}[/bold]  "
                  f"Batch : {hw.batch_size:,}  "
                  f"MaxBars : {hw.max_bars:,}\n")

    # Import pipeline components lazily to avoid slow startup
    console.print("[dim]Importing pipeline components...[/dim]")
    try:
        from data_pipeline import connect, disconnect, run_full_pipeline
        from feature_engine import build_feature_dataframe
        from labeler import compute_direction_labels
        from labeler_exec import compute_exec_labels
        from labeler_modify import compute_modify_labels
        from trainer import HardwareAwareTrainer
        from exporter import export_direction, export_execution, export_modify
        from models.prism import create_prism
        from models.gnn_metals import create_gnn
        from models.apex import create_apex
        from models.ce_net import create_ce
        from models.exec_net import create_exec_net
        from models.modify_net import create_modify_net
    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[dim]Make sure all dependencies are installed: pip install -r requirements.txt[/dim]")
        return

    # Connect to MT5
    console.print("[bold]Connecting to MetaTrader 5...[/bold]")
    if not connect():
        console.print("[red]Failed to connect to MT5. Make sure MT5 is running and logged in.[/red]")
        return
    console.print("[green]MT5 connected.[/green]\n")

    trainer = HardwareAwareTrainer(hw)

    overall_results = {}
    wall_start = time.time()

    for sym_idx, symbol in enumerate(symbols, 1):
        agent = _sym_agent(symbol)
        color = AGENT_COLORS.get(agent, "white")

        console.rule(
            f"[{color}]{symbol}[/{color}]  [dim]({agent})[/dim]  "
            f"[dim]{sym_idx}/{len(symbols)}[/dim]",
            style=color,
        )

        # ------------------------------------------------------------------
        # Step 1: Extract / load data
        # ------------------------------------------------------------------
        with console.status(f"[bold]Extracting bars for {symbol}...[/bold]"):
            try:
                bars_dict = run_full_pipeline(symbols=[symbol], max_bars=hw.max_bars)
                bars = bars_dict.get(symbol)
                if bars is None or len(bars) == 0:
                    console.print(f"[red]No bars returned for {symbol}. Skipping.[/red]")
                    continue
            except Exception as e:
                console.print(f"[red]Data extraction failed for {symbol}: {e}[/red]")
                continue

        n_bars = len(bars)
        console.print(f"  Data     : [bold]{n_bars:,}[/bold] bars")

        # ------------------------------------------------------------------
        # Step 2: Fetch H1/H4 bars for MTF features
        # ------------------------------------------------------------------
        _PIP_SIZE = {
            "GOLD": 0.01, "SILVER": 0.01, "PLATINUM": 0.01, "COPPER": 0.001,
            "USDJPY": 0.01, "BTCUSD": 1.0, "ETHUSD": 0.1, "LTCUSD": 0.01,
            "CrudeOIL": 0.01, "BRENT_OIL": 0.01, "NATURAL_GAS": 0.001,
            "US_500": 0.01, "UK_100": 0.01,
        }
        pip_size = _PIP_SIZE.get(symbol, 0.01 if "JPY" in symbol else 0.0001)

        h1_bars = h4_bars = None
        with console.status(f"[bold]Fetching H1/H4 bars for {symbol}...[/bold]"):
            try:
                from data_pipeline import fetch_h1_bars, fetch_h4_bars
                h1_bars = fetch_h1_bars(symbol)
                h4_bars = fetch_h4_bars(symbol)
                if h1_bars is not None:
                    console.print(f"  H1 bars  : [bold]{len(h1_bars):,}[/bold]")
                if h4_bars is not None:
                    console.print(f"  H4 bars  : [bold]{len(h4_bars):,}[/bold]")
            except Exception as e:
                console.print(f"[yellow]H1/H4 fetch warning for {symbol}: {e} — using M5-only features[/yellow]")

        # ------------------------------------------------------------------
        # Step 3: Feature engineering + labeling
        # ------------------------------------------------------------------
        with console.status(f"[bold]Building features for {symbol}...[/bold]"):
            try:
                dir_feat, exec_feat = build_feature_dataframe(
                    bars, symbol, pip_size=pip_size,
                    h1_df=h1_bars, h4_df=h4_bars)
                dir_labels, regime  = compute_direction_labels(bars)
                exec_labels         = compute_exec_labels(bars, dir_labels, pip_size=pip_size,
                                                          symbol=symbol)
                mod_labels          = compute_modify_labels(bars, dir_labels,
                                                            pip_size=pip_size,
                                                            exec_sl_labels=exec_labels[:, 1])

                from config import FEATURE_WARMUP_BARS
                w = FEATURE_WARMUP_BARS
                dir_feat    = dir_feat[w:]
                exec_feat   = exec_feat[w:]
                dir_labels  = dir_labels[w:]
                exec_labels = exec_labels[w:]
                mod_labels  = mod_labels[w:]
                mod_feat    = np.concatenate(
                    [dir_feat, np.zeros((len(dir_feat), 8), dtype=np.float32)],
                    axis=1,
                )
            except Exception as e:
                console.print(f"[red]Feature engineering failed for {symbol}: {e}[/red]")
                continue

        n_dir  = int((dir_labels != 0).sum())
        n_long = int((dir_labels  > 0).sum())
        n_shrt = int((dir_labels  < 0).sum())
        console.print(f"  Labels   : {n_dir:,} non-flat  ({n_long:,} long / {n_shrt:,} short)")

        # ------------------------------------------------------------------
        # Step 4: Train direction model
        # ------------------------------------------------------------------
        console.print(f"\n  [bold]Training DIRECTION model...[/bold]")
        dir_model = (create_prism()  if agent == "PRISM" else
                     create_gnn()   if agent == "GNN"   else
                     create_apex()  if agent == "APEX"  else
                     create_ce())

        dir_metrics = _train_with_progress(
            label=f"{symbol} DIR",
            color=color,
            train_fn=lambda: trainer.train_direction(
                dir_model, dir_feat, dir_labels,
                epochs=epochs_to_use, symbol=symbol,
            ),
        )

        dir_acc = dir_metrics.get("val_acc", 0.0)
        _print_metric_bar("val_acc", dir_acc, threshold=0.58)

        # ------------------------------------------------------------------
        # Step 5: Train execution model
        # ------------------------------------------------------------------
        console.print(f"\n  [bold]Training EXECUTION model...[/bold]")
        exec_model = create_exec_net()
        exec_metrics = _train_with_progress(
            label=f"{symbol} EXEC",
            color=color,
            train_fn=lambda: trainer.train_execution(
                exec_model, exec_feat, exec_labels,
                epochs=epochs_to_use, symbol=symbol,
            ),
        )

        # ------------------------------------------------------------------
        # Step 6: Train modification model
        # ------------------------------------------------------------------
        console.print(f"\n  [bold]Training MODIFICATION model...[/bold]")
        mod_model = create_modify_net()
        mod_metrics = _train_with_progress(
            label=f"{symbol} MOD",
            color=color,
            train_fn=lambda: trainer.train_modify(
                mod_model, mod_feat, mod_labels,
                epochs=epochs_to_use, symbol=symbol,
            ),
        )

        # ------------------------------------------------------------------
        # Step 7: Export ONNX
        # ------------------------------------------------------------------
        console.print(f"\n  [bold]Exporting ONNX models...[/bold]")
        try:
            with console.status("  Exporting direction models..."):
                export_direction(dir_model, agent, symbol, train_metrics=dir_metrics)
            with console.status("  Exporting execution model..."):
                export_execution(exec_model, agent, symbol, train_metrics=exec_metrics)
            with console.status("  Exporting modification model..."):
                export_modify(mod_model, agent, symbol, train_metrics=mod_metrics)
            console.print(f"  [green]ONNX export complete → {ONNX_OUTPUT_DIR}[/green]")
        except Exception as e:
            console.print(f"  [red]ONNX export failed: {e}[/red]")
            log.exception("Export error for %s", symbol)

        overall_results[symbol] = {
            "agent":    agent,
            "val_acc":  dir_acc,
            "bars":     n_bars,
            "epochs":   dir_metrics.get("epochs_run", 0),
        }

    disconnect()

    # Summary table
    _print_training_summary(overall_results, time.time() - wall_start)


def _train_with_progress(label: str, color: str, train_fn) -> Dict:
    """
    Run train_fn in a thread while reading progress.json to update a
    Rich progress bar per epoch.
    """
    result_box = [None]
    error_box  = [None]
    done_event = threading.Event()

    def _worker():
        try:
            result_box[0] = train_fn()
        except Exception as e:
            error_box[0] = e
            log.exception("Training thread error")
        finally:
            done_event.set()

    thread = threading.Thread(target=_worker, daemon=True)

    with Progress(
        SpinnerColumn(style=color),
        TextColumn(f"[{color}]{label}[/{color}]"),
        BarColumn(bar_width=28, style=color, complete_style=f"bold {color}"),
        MofNCompleteColumn(),
        TextColumn("[dim]ep[/dim]"),
        TextColumn("[cyan]{task.fields[val_acc]}[/cyan]"),
        TextColumn("[dim]acc[/dim]"),
        TextColumn("[yellow]{task.fields[tr_loss]}[/yellow]"),
        TextColumn("[dim]loss[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as prog:
        task = prog.add_task(
            label, total=EPOCHS,
            val_acc="---", tr_loss="---",
        )

        thread.start()
        last_epoch = 0

        while not done_event.is_set():
            pdata = _read_progress()
            if pdata:
                ep    = pdata.get("epoch", 0)
                total = pdata.get("total_epochs", EPOCHS)
                vacc  = pdata.get("val_acc", pdata.get("best", 0.0))
                loss  = pdata.get("train_loss", 0.0)

                if ep != last_epoch:
                    prog.update(
                        task,
                        completed=ep,
                        total=total,
                        val_acc=f"{vacc:.4f}",
                        tr_loss=f"{loss:.4f}",
                    )
                    last_epoch = ep

            done_event.wait(timeout=0.5)

        # Final update
        pdata = _read_progress()
        if pdata:
            ep    = pdata.get("epoch", 0)
            total = pdata.get("total_epochs", EPOCHS)
            vacc  = pdata.get("val_acc", pdata.get("best", 0.0))
            loss  = pdata.get("train_loss", 0.0)
            prog.update(task, completed=ep, total=total,
                        val_acc=f"{vacc:.4f}", tr_loss=f"{loss:.4f}")

        thread.join()

    if error_box[0] is not None:
        console.print(f"  [red]Training error: {error_box[0]}[/red]")
        return {}

    return result_box[0] or {}


def _print_metric_bar(name: str, value: float, threshold: float = 0.58):
    """Print a colored accuracy bar."""
    bar_len = 30
    filled  = int(value * bar_len)
    thresh_pos = int(threshold * bar_len)

    bar = ""
    for i in range(bar_len):
        if i < filled:
            bar += "█" if i < thresh_pos else "▓"
        else:
            bar += "░"

    color = "green" if value >= threshold else ("yellow" if value >= 0.53 else "red")
    console.print(
        f"  [{color}]{name} = {value:.4f}[/{color}]  "
        f"[{color}]{bar}[/{color}]  "
        f"[dim]target ≥ {threshold:.0%}[/dim]"
    )


def _print_training_summary(results: Dict, elapsed: float):
    if not results:
        return

    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    secs    = int(elapsed % 60)

    console.rule("[bold]Training Summary[/bold]", style="cyan")

    tbl = Table(box=box.SIMPLE_HEAD, header_style="bold white on dark_blue", expand=True)
    tbl.add_column("Symbol",  style="bold", width=10)
    tbl.add_column("Agent",   width=8)
    tbl.add_column("Val Acc", justify="center", width=10)
    tbl.add_column("PF",      justify="center", width=7)
    tbl.add_column("WR",      justify="center", width=7)
    tbl.add_column("Bars",    justify="right",  width=12)
    tbl.add_column("Epochs",  justify="right",  width=8)
    tbl.add_column("Status",  width=14)

    all_pass = True
    for sym, r in results.items():
        agent_color = AGENT_COLORS.get(r["agent"], "white")
        acc = r["val_acc"]
        pf  = r.get("profit_factor", 0.0)
        wr  = r.get("win_rate", 0.0)
        acc_color = "green" if acc >= 0.70 else ("yellow" if acc >= 0.58 else "red")
        pf_color  = "green" if pf >= 2.0 else ("yellow" if pf >= 1.0 else "red")
        if acc < 0.58 or pf < 1.0:
            all_pass = False
        if acc >= 0.70 and pf >= 2.0:
            status = "[green]Excellent[/green]"
        elif acc >= 0.58 and pf >= 1.0:
            status = "[green]Good[/green]"
        elif acc >= 0.58:
            status = "[yellow]Low PF[/yellow]"
        else:
            status = "[red]Low Acc[/red]"
        tbl.add_row(
            sym,
            f"[{agent_color}]{r['agent']}[/{agent_color}]",
            f"[{acc_color}]{acc:.4f}[/{acc_color}]",
            f"[{pf_color}]{pf:.3f}[/{pf_color}]" if pf > 0 else "[dim]---[/dim]",
            f"{wr:.3f}" if wr > 0 else "[dim]---[/dim]",
            f"{r['bars']:,}",
            str(r["epochs"]),
            status,
        )

    console.print(tbl)
    console.print(f"\n  Wall time : {hours}h {minutes}m {secs}s")
    if all_pass:
        console.print("  [green bold]All models trained successfully.[/green bold]")
        console.print("  [dim]EA will hot-reload within 30 seconds.[/dim]")
    else:
        console.print("  [yellow]Some models have low val_acc or PF < 1.0. More bars or epochs may help.[/yellow]")
        console.print("  [dim]Tip: Run 'Extract data' first to ensure maximum bars are cached.[/dim]")


# ---------------------------------------------------------------------------
# Data extraction only
# ---------------------------------------------------------------------------

def _run_extract():
    console.print("\n[bold]Data Extraction[/bold]")
    hw = get_hw()
    console.print(f"Max bars per symbol : [bold]{hw.max_bars:,}[/bold]")

    if not Confirm.ask("Connect to MT5 and extract bars for ALL symbols?", default=True):
        return

    try:
        from data_pipeline import connect, disconnect, run_full_pipeline
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print("[bold]Connecting to MT5...[/bold]")
    if not connect():
        console.print("[red]MT5 connection failed.[/red]")
        return

    with console.status("[bold]Extracting bars for all symbols...[/bold]"):
        try:
            result = run_full_pipeline(symbols=ALL_SYMBOLS, max_bars=hw.max_bars)
        except Exception as e:
            console.print(f"[red]Extraction failed: {e}[/red]")
            disconnect()
            return

    disconnect()

    for sym, bars in result.items():
        n = len(bars) if bars is not None else 0
        status = f"[green]{n:,} bars[/green]" if n > 0 else "[red]FAILED[/red]"
        console.print(f"  {sym:10} {status}")

    console.print("\n[green]Extraction complete. Data cached to parquet.[/green]")


# ---------------------------------------------------------------------------
# Live monitor
# ---------------------------------------------------------------------------

_monitor_proc: Optional[subprocess.Popen] = None
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = threading.Event()


def _start_live_monitor():
    global _monitor_proc, _monitor_thread

    if _monitor_proc is not None and _monitor_proc.poll() is None:
        console.print("[yellow]Live monitor is already running (PID %d).[/yellow]" % _monitor_proc.pid)
        return

    log_path = Path(__file__).parent / "live_monitor.log"
    monitor_script = Path(__file__).parent / "live_monitor.py"

    if not monitor_script.exists():
        console.print("[red]live_monitor.py not found.[/red]")
        return

    _monitor_stop.clear()
    log_file = open(log_path, "a", encoding="utf-8")

    _monitor_proc = subprocess.Popen(
        [sys.executable, str(monitor_script)],
        cwd=str(Path(__file__).parent),
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    console.print(f"[green]Live monitor started (PID {_monitor_proc.pid}).[/green]")
    console.print(f"[dim]Logs → {log_path}[/dim]")
    console.print(f"[dim]Polls every {MONITOR_INTERVAL_SEC}s — retrain triggers written to flag files.[/dim]")


def _stop_live_monitor():
    global _monitor_proc
    if _monitor_proc is None or _monitor_proc.poll() is not None:
        console.print("[dim]No live monitor process running.[/dim]")
        return
    _monitor_proc.terminate()
    _monitor_proc = None
    console.print("[yellow]Live monitor stopped.[/yellow]")


def _monitor_status_str() -> str:
    if _monitor_proc is None:
        return "[dim]Monitor: not running[/dim]"
    if _monitor_proc.poll() is None:
        return f"[green]Monitor: running (PID {_monitor_proc.pid})[/green]"
    return "[red]Monitor: stopped[/red]"


# ---------------------------------------------------------------------------
# MT5 path setting
# ---------------------------------------------------------------------------

def _set_mt5_path():
    console.print("\n[bold]MT5 Common Files Path[/bold]")
    console.print(f"Current : [cyan]{MT5_COMMON_DIR}[/cyan]")
    console.print(
        "[dim]This path must point to your MT5 Common/Files folder, e.g.:\n"
        "  C:/Users/<Name>/AppData/Roaming/MetaQuotes/Terminal/Common/Files[/dim]"
    )
    new_path = Prompt.ask("New path (leave blank to keep current)").strip()
    if not new_path:
        return

    p = Path(new_path)
    if not p.exists():
        if Confirm.ask(f"[yellow]Path does not exist: {p}[/yellow]\nCreate it?", default=False):
            p.mkdir(parents=True, exist_ok=True)
        else:
            console.print("[dim]Cancelled.[/dim]")
            return

    env_path = Path(__file__).parent.parent / ".env"
    try:
        lines = []
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()

        updated = False
        normalized = str(p)
        for i, line in enumerate(lines):
            if line.strip().startswith("MT5_COMMON_DIR="):
                lines[i] = f"MT5_COMMON_DIR={normalized}"
                updated = True
                break

        if not updated:
            lines.append(f"MT5_COMMON_DIR={normalized}")

        env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        os.environ["MT5_COMMON_DIR"] = normalized
        console.print(f"[green].env updated with MT5_COMMON_DIR={normalized}[/green]")
        console.print("[dim]Restart the dashboard to reload config.py with the new path.[/dim]")
    except Exception as e:
        console.print(f"[red]Could not update .env: {e}[/red]")
        console.print("[dim]Manually add MT5_COMMON_DIR=... to the project .env file.[/dim]")


# ---------------------------------------------------------------------------
# Hardware display
# ---------------------------------------------------------------------------

def _show_hardware():
    console.print()
    hw = detect_hw()
    console.print(_hardware_panel())
    console.print(f"\n  [dim]To improve training speed:[/dim]")
    if hw.device == "cpu":
        console.print("  [dim]Install CUDA PyTorch for GPU training: https://pytorch.org[/dim]")
    else:
        console.print(f"  [green]GPU detected. Training will use {hw.device.upper()}.[/green]")
    console.print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    console.clear()
    _print_header()

    # Show hardware on startup
    hw = get_hw()
    tier_color = {"enterprise": "green", "mid": "yellow", "low": "yellow", "minimum": "red"}.get(hw.tier, "white")
    console.print(
        f"  Hardware : [bold]{hw.device.upper()}[/bold] | "
        f"Tier : [{tier_color}]{hw.tier.upper()}[/{tier_color}] | "
        f"MaxBars : [bold]{hw.max_bars:,}[/bold] | "
        f"Batch : {hw.batch_size:,}"
    )
    console.print(f"  MT5 Dir  : [dim]{MT5_COMMON_DIR}[/dim]")
    console.print()

    while True:
        # Show status table + monitor status
        console.print(_model_status_table())
        console.print(f"  {_monitor_status_str()}")
        console.print()

        _print_main_menu()

        choice = Prompt.ask("[cyan]Choose[/cyan]").strip().upper()
        console.print()

        if choice == "1":
            _run_training(ALL_SYMBOLS)

        elif choice == "2":
            syms = _pick_symbols()
            if syms:
                _run_training(syms)

        elif choice == "3":
            syms = _pick_agent()
            if syms:
                _run_training(list(syms))

        elif choice == "4":
            _run_extract()

        elif choice == "5":
            pass  # table refreshes at top of loop

        elif choice == "6":
            _start_live_monitor()
            console.print()
            console.print(_monitor_panel())

        elif choice == "7":
            _stop_live_monitor()

        elif choice == "8":
            _show_hardware()

        elif choice == "9":
            _set_mt5_path()

        elif choice == "Q":
            if _monitor_proc is not None and _monitor_proc.poll() is None:
                if Confirm.ask("Stop live monitor before quitting?", default=True):
                    _stop_live_monitor()
            console.print("[dim]Goodbye.[/dim]")
            break

        else:
            console.print("[red]Unknown option.[/red]")

        console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye.[/dim]")
        if _monitor_proc is not None and _monitor_proc.poll() is None:
            _monitor_proc.terminate()
