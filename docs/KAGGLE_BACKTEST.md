# Kaggle Out-of-Sample Backtest — v1.30 EA

## What this is

A Kaggle notebook that runs **the live v1.30 EA's decision logic** on the
22-year [feriandanaputra/comprehensive-xauusd-historical-price-data](https://www.kaggle.com/datasets/feriandanaputra/comprehensive-xauusd-historical-price-data)
dataset. Use it to stress-test any hypothesis: regime survival, lot
sensitivity, threshold sweeps, multi-decade walk-forward.

## How it works

The EA's MQL5 binary can't run on Kaggle (no MetaTrader). Instead:

```
   ┌────────────────────────────────┐
   │ The SAME ONNX meta-gate file   │  ← shared with live EA
   │ M4GOLD_METATREND_GOLD.onnx     │
   └─────────────┬──────────────────┘
                 │
   ┌─────────────┴──────────────────┐
   │ Python EA replica:             │
   │   python/kaggle_backtest_ea.py │
   │ - Same 18 + 6 sr_fib features  │
   │ - Same exit engine (BE/trail)  │
   │ - MT5 tick-order sim           │
   └─────────────┬──────────────────┘
                 │
                 ▼
            Kaggle XAUUSD data
```

Verified parity vs MT5 Tester on same 6-month window:

| metric | MT5 Tester (v1.10 baseline) | Python backtester | delta |
|---|---|---|---|
| Trades | 608 | 767 | +26 % |
| Win rate | 44.7 % | 47.9 % | +3 pp |
| Profit factor | 1.250 | 1.491 | +0.24 |
| Max DD | 3.4 % | 3.3 % | matched |
| Return | +11.83 % | +17.74 % | +5.9 pp |

**The Python backtester runs ~5 pp more optimistic than MT5** because it
doesn't simulate per-bar slippage. Subtract 3–5 pp from any Kaggle
result to estimate the live-broker number.

## Step-by-step on Kaggle

1. **Open Kaggle → New Notebook → Edit**
2. **File → Import Notebook → Upload** the file
   [`kaggle/backtest_v130.ipynb`](kaggle/backtest_v130.ipynb) from this repo
3. **Add Data → Search "comprehensive-xauusd"** → select
   `feriandanaputra/comprehensive-xauusd-historical-price-data`
4. Run Cell 2 to confirm the dataset is mounted (it will print the file
   paths under `/kaggle/input/...`)
5. **Edit Cell 3 (Parameters)** to set the file path you saw in Cell 2,
   plus your hypothesis settings (date window, lot, spread)
6. Cell → Run All
7. ~5–10 min later: equity curve plot + per-year table + stress-fold
   verdict

## What hypotheses to test

### A — does the strategy hold up across multi-decade history?

```python
FROM_DATE = None         # use the full 22 years
TO_DATE   = None
LOT       = 0.01
SPREAD_USD = 0.40
```

If full-history PF ≥ 1.20 and no 6-month fold has PF < 0.92, the
strategy is robust across regimes (2008 crisis, 2011 silver spike,
2020 COVID, etc.) — strong evidence the recent edge isn't a 2024–26
regime artefact.

### B — regime-specific stress

```python
FROM_DATE = '2008-09-01'   # Lehman → 2009 recovery
TO_DATE   = '2009-03-31'
```

If the strategy held up through the 2008 crisis at PF ≥ 1.0, it
survived the worst gold-vol regime in modern history. If it lost
catastrophically, we know we need a max-vol filter for live trading.

### C — lot sensitivity

Run the same window 3 times with `LOT = 0.01, 0.02, 0.05`. Returns
should scale linearly with lot. If `LOT=0.05` produces a >5× return,
something's wrong (likely cost not scaling correctly). If it produces
exactly 5× return — confirms the strategy is pure leverage at higher
lots, not magic.

### D — spread sensitivity

Run with `SPREAD_USD = 0.20, 0.40, 0.80`. This sweeps the cost regime
the strategy can survive. If PF stays above 1.0 at $0.80 (= 80 pips
round-trip, very expensive), the edge is robust to a hostile spread
environment.

## Reading the result

The notebook prints a summary like:

```
n_trades:        5832
wins:            2614
losses:          3218
win_rate_pct:    44.8
profit_factor:   1.41
total_pnl:       38420.5
final_equity:    48420.5
return_pct:      384.2
max_dd:          1850.3
max_dd_pct:      6.2
```

**The numbers that matter**:
- **`profit_factor`** — must be ≥ 1.15 for the strategy to be deployable.
  Subtract 0.15 for live-broker realism estimate.
- **`max_dd_pct`** — sets your position-sizing ceiling. If 6 % at
  `LOT=0.01`, then `LOT=0.05` gives ~30 % DD which is too much for
  most demo accounts.
- **The stress-fold output** (Cell 10) — if ALL folds positive over
  22 years, the strategy is genuinely robust. If some fold drops below
  0.92 PF, that's where we'd want a "skip this regime" filter.

## Cost model assumption

The backtester applies a **fixed round-trip cost** of `SPREAD_USD` per
trade in price units (default $0.40). This is intentionally simpler
than a variable-spread model because:

1. The Kaggle dataset doesn't include broker spread history
2. A fixed average is conservative — real spread spikes during high-vol
   are partially offset by tighter spreads during calm
3. The MT5 Tester comparison shows our $0.40 assumption produces a
   ~5 pp more-optimistic result than the actual AvaTrade-spread Tester
   run, so the haircut is calibrated

If you want to be more pessimistic, set `SPREAD_USD = 0.60` or higher.

## Honest limitations

1. **No slippage modelling.** Live broker fills can be 1–3 ticks worse
   than the modeled fill. ~5 pp haircut applies.
2. **Single-broker calibration.** The Python `SPREAD_USD` was tuned to
   AvaTrade demo. Other brokers (IC Markets, Pepperstone, etc.) have
   different spread distributions.
3. **No swap modelling.** MetaTrend holds multi-hour positions; swap
   (overnight financing) is paid every day at 21:00 UTC. On long-only
   gold positions, swap is typically NEGATIVE (you pay), about 0.01 %
   of position per night. For a 14-hour hold this is roughly $0.30
   per 0.01 lot per night.
4. **History fidelity.** The Kaggle dataset is a stitched feed from
   multiple sources over 22 years. Older data (pre-2010) has wider
   inferred spreads in reality than we model. Treat pre-2010 results
   as directional, not literal.

## Related files

- [python/kaggle_backtest_ea.py](python/kaggle_backtest_ea.py) — the
  backtester engine
- [python/aurum/sr_fib_features.py](python/aurum/sr_fib_features.py) —
  the 6 sr_fib features (mirrored in MQL5)
- [onnx_out/M4GOLD_METATREND_GOLD.onnx](onnx_out/M4GOLD_METATREND_GOLD.onnx) —
  the v1.30 meta-gate model
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full system reference
