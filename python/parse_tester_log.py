"""
parse_tester_log.py — extract PF / win-rate / drawdown / equity curve from
an MT5 tester agent log. MT5 didn't write the HTML report (absolute-path
Report= flag is finicky) so we reconstruct stats from the deal stream.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict


# Lines look like:
#   CS\t0\t15:23:12.760\tTrades\t2025.11.20 08:10:00   deal #2 sell 0.01 GOLD at 4049.96 done (based on order #2)
# Capture both wall-clock and simulated time so we can isolate a single
# tester run by the wall-clock window in which it was running.
DEAL_RE = re.compile(
    r"^\w\w\t\d\t(?P<wc>\d{2}:\d{2}:\d{2})\.\d+\tTrades\t"
    r"(?P<dt>\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\s+deal #(?P<id>\d+)\s+"
    r"(?P<side>buy|sell)\s+(?P<vol>[\d.]+)\s+GOLD\s+at\s+(?P<px>[\d.]+)",
    flags=re.MULTILINE,
)


def main(log_path: str, wc_from: str = "15:23:00",
         wc_to: str = "15:27:00") -> int:
    """Filter by wall-clock launch window so we isolate one tester run."""
    p = Path(log_path)
    if not p.exists():
        print(f"NOT FOUND: {log_path}"); return 1
    text = p.read_text(encoding="utf-16", errors="replace")
    deals = []
    for m in DEAL_RE.finditer(text):
        wc = m.group("wc")
        if not (wc_from <= wc <= wc_to):
            continue
        deals.append({
            "wc": wc,
            "dt": m.group("dt"),
            "id": int(m.group("id")),
            "side": m.group("side"),
            "vol": float(m.group("vol")),
            "px":  float(m.group("px")),
        })
    print(f"parsed {len(deals)} deals  (wallclock {wc_from} <= t <= {wc_to})")
    if not deals:
        return 1

    # MT5 deals come in pairs: opening deal then closing deal.
    # Match them by FIFO on (side opposite + same vol).
    # But the simplest approach: build trades by walking pairs.
    # Approach: each deal flips position. If pos==0, this deal OPENS;
    # if pos!=0 and deal closes/reverses it, that's a CLOSE.
    pos = 0.0
    avg_entry = 0.0
    entry_side = None
    open_dt = None
    trades = []           # list of (entry_dt, exit_dt, side, entry_px, exit_px, pnl_pts)
    for d in deals:
        signed = d["vol"] if d["side"] == "buy" else -d["vol"]
        new_pos = pos + signed
        if abs(pos) < 1e-9 and abs(new_pos) > 1e-9:
            # opening
            avg_entry = d["px"]
            entry_side = "long" if signed > 0 else "short"
            open_dt = d["dt"]
        elif abs(pos) > 1e-9 and abs(new_pos) < 1e-9:
            # closing
            exit_px = d["px"]
            if entry_side == "long":
                pts = exit_px - avg_entry
            else:
                pts = avg_entry - exit_px
            trades.append({
                "open_dt": open_dt, "close_dt": d["dt"],
                "side": entry_side,
                "entry": avg_entry, "exit": exit_px,
                "points": pts,
            })
            entry_side = None
            avg_entry = 0.0
        # (we ignore stacked/reversed cases for now — MetaTrend with stack=1
        #  in our run doesn't do those)
        pos = new_pos

    n_tr = len(trades)
    if n_tr == 0:
        print("no completed round-trips found")
        return 1

    wins = [t for t in trades if t["points"] > 0]
    losses = [t for t in trades if t["points"] <= 0]
    sum_w = sum(t["points"] for t in wins)
    sum_l = -sum(t["points"] for t in losses) or 1e-12
    pf = sum_w / sum_l
    win_rate = len(wins) / n_tr

    # Approximate $ PnL: GOLD CFD on AvaTrade is typically $1 per point per
    # 0.01 lot. We assume that, then report against the $1000 starting deposit
    # implied from the EA InpBaseLot=0.01 and the agent log final balance.
    # Sum the points to a P&L proxy.
    pts_per_lot01 = 1.0     # MT5 contract sizing — close enough for relative comparison
    pnl_total = sum(t["points"] for t in trades) * pts_per_lot01

    # max drawdown from running equity (points)
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for t in trades:
        eq += t["points"]
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)

    avg_win  = sum_w / max(len(wins), 1)
    avg_loss = sum_l / max(len(losses), 1)

    print()
    print("=" * 64)
    print(f"  MetaTrend 6-month Strategy Tester result (GOLD,M5)")
    print("=" * 64)
    print(f"  closed trades       : {n_tr}")
    print(f"  wins / losses       : {len(wins)} / {len(losses)}")
    print(f"  win rate            : {win_rate*100:.1f}%")
    print(f"  profit factor       : {pf:.3f}")
    print(f"  total points (sum)  : {pnl_total:+.2f}")
    print(f"  avg win             : {avg_win:+.2f}")
    print(f"  avg loss            : -{avg_loss:.2f}")
    print(f"  reward/risk (W/L)   : {avg_win/avg_loss:.2f}")
    print(f"  max drawdown (pts)  : {mdd:.2f}")
    print(f"  best trade          : {max(t['points'] for t in trades):+.2f}")
    print(f"  worst trade         : {min(t['points'] for t in trades):+.2f}")
    first = trades[0]["open_dt"]; last = trades[-1]["close_dt"]
    print(f"  period              : {first}  ->  {last}")
    print("=" * 64)
    print()
    print("First 10 round-trips:")
    for t in trades[:10]:
        print(f"  {t['open_dt']} {t['side']:5s} {t['entry']:.2f} -> "
              f"{t['close_dt']} {t['exit']:.2f}  pts={t['points']:+.2f}")
    print("Last 5 round-trips:")
    for t in trades[-5:]:
        print(f"  {t['open_dt']} {t['side']:5s} {t['entry']:.2f} -> "
              f"{t['close_dt']} {t['exit']:.2f}  pts={t['points']:+.2f}")
    return 0


if __name__ == "__main__":
    default_log = r"C:\Users\Angela Ramos\AppData\Roaming\MetaQuotes\Tester\D0E8209F77C8CF37AD8BF550E51FF075\Agent-127.0.0.1-3000\logs\20260520.log"
    log_arg = sys.argv[1] if len(sys.argv) > 1 else default_log
    wc_from = sys.argv[2] if len(sys.argv) > 2 else "15:23:00"
    wc_to   = sys.argv[3] if len(sys.argv) > 3 else "15:24:00"
    sys.exit(main(log_arg, wc_from, wc_to))
