# MetaTrend — Deployment Guide

The **validated** GOLD strategy. Unlike AURUM-direction (retired — no
edge), this one cleared a leak-free purged-CV gate and is the
recommended thing to field-test.

## What it is

- **Direction** — a slow EMA 50/200 cross on M5. Deterministic.
- **Filter** — an XGBoost meta-gate that scores P(trust this trend); the
  EA enters only when P(act) ≥ the spec threshold.
- **Exit** — trend flip, a loose ATR trailing stop, or a timeout.

Validation (leak-free purged 6-fold CV, net of cost):

| metric | value |
|---|---|
| mean profit factor | **1.41** |
| raw trend rule (control) | 1.12 — the meta-gate adds **+0.29** |
| per-fold PF | 1.57 / 0.997 / 1.33 / 1.45 / 1.59 / 1.49 |
| ONNX parity | 1.2e-7 (clean) |

It is a *modest, honest* edge — not a money printer — but it is real:
cost-aware, leak-free, 5 of 6 folds strongly positive.

## Train (local — ~1 minute, no GPU/Kaggle needed)

```
.\train.bat metatrend
```

Produces in `onnx_out\`:
- `M4GOLD_METATREND_GOLD.onnx` — the meta-gate
- `M4GOLD_METATREND_GOLD_spec.json` — threshold + `deploy` flag + CV report

`deploy` is `true` only if the run clears the gate (mean PF ≥ 1.15,
beats the raw rule by ≥ 0.05, no fold below 0.92).

## Deploy to MT5

1. Copy both artifacts into MT5 Common Files:
   `...\Terminal\Common\Files\`
2. Copy `ea\` into `...\MQL5\Experts\MT5bot_m4Gold\` (keep `includes\`).
3. MetaEditor → open `MT5bot_m4Gold_MetaTrend.mq5` → **F7** (compile).
4. Attach `MT5bot_m4Gold_MetaTrend` to a **GOLD chart** (any timeframe —
   it reads M5 internally).

On attach, expect:
```
[MetaGate] ready  version=metatrend-1.0.0  deploy=true  act_thr=0.55
[MetaTrend] LIVE — EMA 50/200 trend + meta-gate.
```

## Inputs

| input | default | meaning |
|---|---|---|
| `InpBaseLot` | 0.01 | lot per trade |
| `InpSlAtr` | 3.0 | initial stop (ATR multiples) — wide, trend-following |
| `InpUseTrailing` / `InpTrailStartAtr` / `InpTrailAtr` | on / 2.0 / 3.0 | loose trailing stop |
| `InpMaxHoldBars` | 288 | force-exit after ~24 h |
| `InpExitOnFlip` | true | exit when the EMA trend flips |
| `InpRespectDeploy` | true | sit idle if `deploy=false` |

## Honest expectations

- A PF of ~1.4 in purged CV is a genuine edge, but it is *frictionless-
  label* CV; live PF will be lower once real spread/swap/execution apply.
  Expect something in the ~1.1–1.3 range live if it holds up.
- Field-test on **demo for 30+ days** before any real capital. Compare
  the demo's realised PF to the 1.41 spec figure.
- This is a slow strategy — multi-hour holds, relatively few trades.
  Long flat stretches between trends are normal, not a malfunction.
