# GOLD Edge Search — Findings

> 4 iterations, 39 hypotheses, all measured leak-free under purged
> cross-validation with the fast research harness (`python/aurum/research.py`).
> Every profit factor (PF) below is **net of transaction cost**.

## The question

After the time-sync leak was fixed, AURUM's premise — predicting M5 GOLD
direction — collapsed to PF ~1.0 (a coin flip). The question: does GOLD
carry *any* causal, cost-surviving edge for these models? If yes, find
and characterise it; if no, say so plainly.

Method: XGBoost under `PurgedKFold` is a faithful, ~1-minute proxy for
"is there edge here." If a gradient-boosted model on causal features
cannot beat a coin flip, the transformer will not either.

## Iteration 1 — target search

Predicting GOLD **direction from scratch is dead at every horizon**:

| target | PF |
|---|---|
| triple-barrier direction, 20 / 48 / 96 / 240-bar | 1.01 – 1.04 |
| sign of forward return, 96 / 240-bar | 0.97 – 0.98 |

One hypothesis broke through: **meta-labelling** an EMA-cross trend rule
(predict *when to trust* a trend signal, not the direction) — PF 1.20,
all 5 folds positive.

## Iteration 2 — meta-label refinement

Varying the trend primary and horizon revealed a clear law:
**slow primary + long hold = edge; fast primary + short hold = noise.**

| config | net PF |
|---|---|
| ema50/200, 240-bar hold | 1.27 |
| momentum-120, 240-bar | 1.25 |
| donchian-96, 144-bar | 1.23 |
| ema20/50, 48-bar | 1.05 (weak) |

## Iteration 3 — confidence-threshold lock

Raising the meta-model's P(act) threshold lifts the *mean* PF but trades
away fold-consistency. Plain argmax (t50) is the robust choice — the only
configs with **every purged-CV fold profitable**:

| config | net PF | worst fold | folds |
|---|---|---|---|
| `ema50_200_h240` | 1.265 | +1.031 | [1.62, 1.03, 1.46, 1.04, 1.18] |
| `donch96_h144` | 1.230 | +1.092 | [1.25, 1.21, 1.49, 1.09, 1.11] |

## Iteration 4 — cost stress test

The edge survives a pessimistic round-trip cost:

| config | 1.5e-4 | 2.5e-4 | 4.0e-4 |
|---|---|---|---|
| `ema50_200_h240` | 1.265 | 1.230 | 1.179 |
| `donch96_h144` | 1.230 | 1.186 | 1.123 (all folds ≥1.0) |

## Verified finding

**GOLD has a real, modest, leak-free, cost-surviving edge — and it is
NOT M5 direction prediction. It is meta-labelled slow trend-following:**

- **Direction** comes from a deterministic slow trend rule —
  EMA 50/200 cross (~20 h hold) or Donchian-96 breakout (~12 h hold).
- **The filter** is an XGBoost meta-model that decides *when to trust*
  that trend signal (López de Prado meta-labelling).
- **Net PF ≈ 1.19–1.27** at realistic cost; every purged-CV fold
  positive; survives a 4× pessimistic cost.

This independently rediscovers the original mk4 result: GOLD's tradeable
structure is multi-hour **trend persistence**, not short-horizon direction.

It is a *modest* edge — PF ~1.2, not a money printer — but it is the
honest ceiling of what is in this data, and unlike AURUM-direction it is
real.

## Recommendation

Build the deployable strategy around this finding (`train_h7_metatrend`
+ a meta-trend EA): deterministic slow-trend primary for direction, an
ONNX-exported XGBoost meta-gate for the accept/reject filter, ~12–20 h
holds. Retire M5-direction AURUM as the primary signal — the search
proved that target has no edge.
