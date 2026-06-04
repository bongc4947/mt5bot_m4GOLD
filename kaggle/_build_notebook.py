"""Build the consolidated single-cell Kaggle backtest notebook.
Run: python kaggle/_build_notebook.py
"""
import json
from pathlib import Path

CELL1_SOURCE = r'''# =========================================================================
# === PARAMETERS - edit these to test your hypothesis ===
# =========================================================================
DATA_PATH            = None        # None = auto-detect M5 file from /kaggle/input
FROM_DATE            = None        # e.g. '2010-01-01', or None for full history
TO_DATE              = None        # e.g. '2020-12-31', or None for latest
DEPOSIT              = 10000.0
LOT                  = 0.01
MAX_STACK            = 1           # 1 = no pyramiding (matches Tester baseline)

# Cost model
USE_VARIABLE_SPREAD  = False       # False = flat SPREAD_USD; True = per-bar <SPREAD>
SPREAD_USD           = 0.29        # flat round-trip USD (0.29 ~ AvaTrade live median)

# Cost-hygiene filters (mirror EA v1.40 inputs; default OFF for clean baseline)
MAX_SPREAD_POINTS    = 0           # 0 = OFF. Skip entry if bar spread > N points
SKIP_ASIA_SESSION    = False       # skip 22:00-06:00 UTC entries (thin liquidity)

# A/B mode: run all 4 filter combinations in one go (15-25 min each, ~1-1.5h total)
RUN_AB_SUITE         = False       # True = ignore the singular params above, run a 2x2 suite
# =========================================================================

import os, sys, subprocess, re, logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# --- 1/4: install deps + force-FRESH-clone the repo ---
# Always nuke and re-clone. Previously tried fetch+reset --hard origin/master
# but the shallow clone's tracking refs can vary (origin/main vs origin/master
# depending on when/how it was cloned), causing exit 128 on the reset. Fresh
# clone is small (~5 MB) and trivially fast - the most robust option.
subprocess.run(['pip', 'install', '-q', 'onnxruntime'], check=True)
import shutil
REPO    = '/kaggle/working/mt5bot_m4GOLD'
GIT_URL = 'https://github.com/bongc4947/mt5bot_m4GOLD.git'
if os.path.isdir(REPO):
    shutil.rmtree(REPO)
subprocess.run(['git', 'clone', '--depth', '1', GIT_URL, REPO],
               check=True, capture_output=True)

head_hash    = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
head_ts_unix = int(subprocess.run(['git', '-C', REPO, 'log', '-1', '--format=%ct', 'HEAD'], capture_output=True, text=True).stdout.strip())
head_subject = subprocess.run(['git', '-C', REPO, 'log', '-1', '--format=%s', 'HEAD'], capture_output=True, text=True).stdout.strip()
head_ts      = datetime.fromtimestamp(head_ts_unix, tz=timezone.utc)
age_min      = (datetime.now(timezone.utc) - head_ts).total_seconds() / 60.0
if REPO + '/python' in sys.path: sys.path.remove(REPO + '/python')
sys.path.insert(0, f'{REPO}/python')
for mod in list(sys.modules):
    if mod.startswith('kaggle_backtest_ea') or mod.startswith('aurum'):
        del sys.modules[mod]
print(f'[1/4] repo synced.  HEAD={head_hash}  ({head_ts.isoformat()} - {age_min:.0f} min ago)')
print(f'      subject: {head_subject}')

# --- 2/4: locate dataset ---
if DATA_PATH is None:
    all_files = []
    for dp, _, files in os.walk('/kaggle/input'):
        for f in files:
            all_files.append(os.path.join(dp, f))
    def _score(p):
        n = os.path.basename(p).lower()
        bonus = 5 if 'xau' in n else 0
        if re.search(r'(_m5_|_5m_|m5\.|5min)', n): return 100 + bonus
        if re.search(r'(_m1_|_1m_|m1\.|1min)', n): return 80  + bonus
        if re.search(r'(_m15_|15min)', n):         return 50  + bonus
        return 0
    ranked = sorted(all_files, key=lambda p: -_score(p))
    if not ranked or _score(ranked[0]) == 0:
        raise RuntimeError('No M5/M1 XAU file found in /kaggle/input. Attach the dataset via Add Data.')
    DATA_PATH = ranked[0]
print(f'[2/4] DATA_PATH: {DATA_PATH}')

# --- 3/4: load data + meta-gate ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)
from kaggle_backtest_ea import load_m5_data, load_meta_gate, DEFAULT_INPUTS, run_backtest

m5 = load_m5_data(Path(DATA_PATH))
if FROM_DATE: m5 = m5[m5['time'] >= pd.Timestamp(FROM_DATE, tz='UTC')]
if TO_DATE:   m5 = m5[m5['time'] <= pd.Timestamp(TO_DATE,   tz='UTC')]
m5 = m5.reset_index(drop=True)
if 'spread' in m5.columns:
    spread_note = f'YES (mean {float(m5["spread"].mean()):.1f} pts)'
else:
    spread_note = f'NO (fallback to flat ${SPREAD_USD})'
print(f'[3/4] M5 bars: {len(m5):,}  span: {m5["time"].iloc[0]} -> {m5["time"].iloc[-1]}')
print(f'      gold range: ${m5["close"].min():.2f} -> ${m5["close"].max():.2f}')
print(f'      spread col: {spread_note}')

sess, in_name, spec = load_meta_gate(
    Path(REPO + '/onnx_out/M4GOLD_METATREND_GOLD.onnx'),
    Path(REPO + '/onnx_out/M4GOLD_METATREND_GOLD_spec.json'))
print(f'      meta-gate: {spec.get("version")}  n_features={spec["n_features"]}  thr={spec["act_threshold"]}  CV_PF={spec["cv"]["meta_mean_pf"]}')

# --- 4/4: run the backtest(s) ---
def _do_run(label, overrides):
    inputs = dict(DEFAULT_INPUTS)
    inputs.update(dict(base_lot=LOT, max_stack=MAX_STACK,
                        spread_usd=SPREAD_USD, use_variable_spread=USE_VARIABLE_SPREAD,
                        max_spread_points=MAX_SPREAD_POINTS,
                        skip_asia_session=SKIP_ASIA_SESSION))
    inputs.update(overrides)
    print(f'\n  -- run [{label}] -- overrides: {overrides}')
    t0 = datetime.now()
    r = run_backtest(m5, sess, in_name, spec, inputs, deposit=DEPOSIT, verbose=False)
    elapsed = (datetime.now()-t0).total_seconds()
    s = r['summary']
    print(f'    elapsed {elapsed:.0f}s  -  '
          f'final ${s["final_equity"]:,.0f}  ({s["return_pct"]:+.2f}%)  PF {s["profit_factor"]}  '
          f'{s["n_trades"]} trades  WR {s["win_rate_pct"]:.1f}%  DD {s["max_dd_pct"]:.1f}%  '
          f'skipped spread={s["spread_filter_skips"]} session={s["session_filter_skips"]}')
    return label, r

est_min = (len(m5) // 60000) + 1
if RUN_AB_SUITE:
    print(f'[4/4] A/B suite: 4 runs * ~{est_min} min each (~{est_min*4} min total)')
    ab_results = [
        _do_run('A: baseline',         dict(max_spread_points=0,  skip_asia_session=False)),
        _do_run('B: spread<=50',       dict(max_spread_points=50, skip_asia_session=False)),
        _do_run('C: skip Asia',        dict(max_spread_points=0,  skip_asia_session=True)),
        _do_run('D: spread + Asia',    dict(max_spread_points=50, skip_asia_session=True)),
    ]
    results = ab_results[0][1]   # keep first run results for Cell 2 plotting
    print()
    print('=' * 100)
    print(f'  A/B SUITE  -  v1.30 on {len(m5):,} bars  ({m5["time"].iloc[0].date()} -> {m5["time"].iloc[-1].date()})')
    print('=' * 100)
    print(f'  {"run":24} {"final $":>12} {"return%":>9} {"PF":>6} {"trades":>8} {"WR%":>6} {"DD%":>6}')
    print('  ' + '-' * 78)
    for lbl, r in ab_results:
        s = r['summary']
        print(f'  {lbl:24} {s["final_equity"]:>12,.0f} {s["return_pct"]:>+9.2f} '
              f'{s["profit_factor"]:>6.3f} {s["n_trades"]:>8} {s["win_rate_pct"]:>6.1f} {s["max_dd_pct"]:>6.1f}')
    print('=' * 100)
    print(f'  Reminder: Python ~5pp more optimistic than MT5 Tester (no per-bar slippage).')
else:
    print(f'[4/4] single run (~{est_min} min)')
    _, results = _do_run('single', {})
    s = results['summary']
    print()
    print('=' * 72)
    print(f'  RESULT  -  v1.30 on {len(m5):,} bars  ({m5["time"].iloc[0].date()} -> {m5["time"].iloc[-1].date()})')
    print('=' * 72)
    for k, v in s.items():
        print(f'  {k:22}: {v}')
    print('=' * 72)
    print(f'  Reminder: Python ~5pp more optimistic than MT5 Tester (no per-bar slippage).')
'''

CELL2_SOURCE = r'''# Run AFTER Cell 1 completes - equity curve, per-year, exit reasons, stress folds
import numpy as np
import matplotlib.pyplot as plt

trades_df = pd.DataFrame(results['trades'])
eq = np.array(results['equity_curve'], dtype=np.float64)
eq[eq == 0] = DEPOSIT
peak = np.maximum.accumulate(eq)
dd_pct = (peak - eq) / np.maximum(peak, 1) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(m5['time'], eq, lw=1.0, color='#0066cc', label='Equity')
ax1.plot(m5['time'], peak, lw=0.7, color='#999', alpha=0.6, label='Peak')
ax1.axhline(DEPOSIT, ls=':', color='gray', label='Deposit')
ax1.set_ylabel('Equity (USD)')
ax1.set_title(f'v1.30 - {s["return_pct"]:+.2f}%  PF {s["profit_factor"]}  {s["n_trades"]} trades  DD {s["max_dd_pct"]:.1f}%')
ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
ax2.fill_between(m5['time'], 0, -dd_pct, color='#cc0000', alpha=0.4)
ax2.set_ylabel('Drawdown (%)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

if len(trades_df):
    trades_df['open_time'] = pd.to_datetime(trades_df['open_time'])
    trades_df['year']      = trades_df['open_time'].dt.year
    yearly = trades_df.groupby('year').agg(
        trades=('pnl','count'),
        wins=('pnl', lambda s: (s>0).sum()),
        pnl_sum=('pnl','sum'),
        pf=('pnl', lambda s: round(s[s>0].sum() / max(-s[s<=0].sum(), 1e-9), 3)),
    ).round(2)
    yearly['win_rate'] = (yearly['wins'] / yearly['trades'] * 100).round(1)
    print('=== PER-YEAR ===')
    print(yearly.to_string())
    print()
    by_reason = trades_df.groupby('exit_reason').agg(
        n=('pnl','count'),
        sum_pnl=('pnl','sum'),
        avg_pnl=('pnl','mean'),
    ).round(2)
    print('=== EXIT REASONS ===')
    print(by_reason.to_string())
    print()
    trades_df['period'] = trades_df['open_time'].dt.to_period('6M').astype(str)
    fold = trades_df.groupby('period').agg(
        trades=('pnl','count'),
        pf=('pnl', lambda s: round(s[s>0].sum() / max(-s[s<=0].sum(), 1e-9), 3)),
    )
    bad  = (fold['pf'] < 0.92).sum()
    print('=== STRESS GATE (6mo folds) ===')
    print(f'total folds: {len(fold)}  -  passing (PF >= 0.92): {len(fold)-bad}  -  failing: {bad}')
    worst = fold['pf'].min()
    verdict = 'FAIL' if worst < 0.92 else 'PASS'
    print(f'worst fold PF: {worst}     -- deploy gate {verdict}')
'''

INTRO = r'''# MT5bot_m4Gold v1.30 - Out-of-sample Kaggle backtest

Runs the same v1.30 EA (24-feature meta-gate + S/R+Fib) the live MT5 EA uses, against multi-decade XAUUSD data.

**How to use**:
1. Attach the `feriandanaputra/comprehensive-xauusd-historical-price-data` dataset (right sidebar -> Add Data)
2. Edit the PARAMETERS block at the top of Cell 1
3. Run Cell 1 (~5 min for 1-year window, ~25 min for full 21-year)
4. (Optional) Run Cell 2 for equity-curve + per-year + exit-reason analysis

**Note**: Python backtester runs ~5pp more optimistic than the real MT5 Tester (no per-bar slippage). Subtract 3-5pp from the headline return for honest live-broker expectation.

The HEAD commit hash + timestamp printed at the start of Cell 1 lets you verify the clone is up to date with the latest pushed code.
'''

def lines(s):
    return [ln + "\n" for ln in s.splitlines()]

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": lines(INTRO)},
        {"cell_type": "code",     "execution_count": None, "metadata": {},
         "outputs": [], "source": lines(CELL1_SOURCE)},
        {"cell_type": "markdown", "metadata": {},
         "source": lines("## Cell 2 (optional) - equity curve, per-year, exit reasons, stress folds\n\nRun this AFTER Cell 1 completes.")},
        {"cell_type": "code",     "execution_count": None, "metadata": {},
         "outputs": [], "source": lines(CELL2_SOURCE)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# verify Python syntax in code cells
import ast
for idx in (1, 3):
    src = "".join(nb["cells"][idx]["source"])
    ast.parse(src)
    print(f"  cell {idx} parses OK ({len(src)} chars)")

out_path = Path(__file__).parent / "backtest_v130.ipynb"
out_path.write_text(json.dumps(nb, indent=1))
print(f"wrote {out_path}")
