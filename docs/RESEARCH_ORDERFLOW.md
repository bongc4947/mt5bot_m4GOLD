# Iteration 7 — Order-Flow Features as Meta-Gate Inputs

> Honest ablation: do microstructure features extracted from the tick-bar
> parquet lift the leak-free purged-CV edge of MetaTrend?
>
> Short answer: **no**, on this data feed. They regress mean PF and break
> a fold. Documented so we don't try again without different inputs.

## What I tried

Per-M5 aggregations of `HYDRA4_TBARS_GOLD_100tpb.parquet`, bucketed by
floor-to-5-min on the tbar close timestamp. Five new causal features
appended to the 18-feature MetaTrend stack:

| feature | what it is |
|---|---|
| `of_micro_drift_z48` | rolling z-score of per-M5 sum of microprice_drift over 48 bars |
| `of_micro_drift_cum24` | cumulative micro drift over 24 M5 bars, normalised by activity |
| `of_spread_z96` | rolling z-score of mean spread over 96 bars |
| `of_sreg_now` | current bucket's max spread_regime, centred at 0.20 |
| `of_activity_z96` | rolling z-score of tbar count per M5 (volatility/activity proxy) |

Implementation: [python/aurum/orderflow_features.py](python/aurum/orderflow_features.py).
Disk-cached, ~200 ms to build for 185 k bars.

## Critical data finding — 5 of 8 microstructure columns are dead

Quantile audit of the tick-bar parquet:

| column | live? | evidence |
|---|---|---|
| `ofi` | **NO** | constant at -1.0, std 0 |
| `cvd` | **NO** | saturated near -1 (derived from ofi) |
| `taker_ratio` | **NO** | constant at 1.0 across all quantiles |
| `tick_volume` | **NO** | 0 at 99th percentile |
| `real_volume` | **NO** | 0 at 99th percentile |
| `microprice_drift` | yes | ~ +/- 0.65 range, real variance |
| `spread` | yes | 0.27..1.67 quantile range |
| `spread_regime` | yes | 0.13..0.31, narrow but live |

This is a data-availability fact: MT5 retail ticks generally do not carry
the buy/sell aggressor flag, so the upstream tick-bar builder cannot
classify trades. **Any future order-flow work has to start by sourcing a
feed that ships these fields with content** (e.g. Polygon, Tickdata,
broker-direct FIX feed) — not by reprocessing the existing tick parquet.

## Result — full 174 k M5 bars

| variant | meanPF | excess | min fold | folds all-pos | deploy |
|---|---|---|---|---|---|
| baseline (18 feat) | 1.406 | +0.287 | 0.997 | 5/6 | YES |
| +tspulse (22 feat) | **1.440** | **+0.326** | **1.062** | **6/6** | YES |
| **+orderflow (23 feat)** | **1.373** | +0.252 | **0.860** | 5/6 | **NO** |
| +tspulse+orderflow (27 feat) | 1.405 | +0.292 | 0.930 | 5/6 | YES (no gain) |

The order-flow pack:
- regresses headline PF by 0.033
- breaks fold 2 worse (0.860 vs 0.997 baseline)
- offers no marginal lift on top of tspulse

## What this means

- The 3 alive microstructure columns are either **already encoded** in
  the existing 18 price features (spread spikes correlate with vol, vol
  is already in ATR/RV features) or are **noise** for a slow-trend
  strategy with a 240-bar horizon.
- It is NOT a refutation of order-flow signal in general. It is a
  refutation of this specific source. Trade-classified OFI on a feed
  that actually has aggressor flags is still a worthwhile target if a
  proper feed becomes available.

## Reproduce

```
python python/train_h7_metatrend.py --with-orderflow              # 23 feat
python python/train_h7_metatrend.py --with-tspulse --with-orderflow # 27 feat
```

Research-variant artifacts are written with a suffix
(`M4GOLD_METATREND_GOLD_orderflow.onnx` etc.) so they never overwrite
the production 18-feature bundle the live EA reads.

## Status

- Implementation: kept (lives behind `--with-orderflow`, off by default)
- Production bundle: **unchanged** — still the 18-feature MetaTrend
- Roadmap: order-flow re-attempt deferred until a feed with live
  aggressor classification is available
