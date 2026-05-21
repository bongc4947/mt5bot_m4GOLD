# MT5bot_m4Gold — Architecture & Governance

> The single authoritative reference. If two docs disagree, this one
> wins. Last updated 2026-05-21 reflecting v1.20.

## 1. What this system is, in one paragraph

A live trading bot for **GOLD only**, running as an MT5 Expert Advisor
on a 5-minute chart. It enters trades using a **deterministic slow
trend rule** (EMA 50/200 cross) filtered by a **machine-learning
meta-gate** (XGBoost classifier exported to ONNX). It manages exits
with a four-mechanism engine (initial SL, breakeven move, ATR
trailing stop, trend-flip / timeout). The model and EA are co-trained
in Python; the trained model is exported to ONNX so the live EA can
run it inside MT5 with zero Python dependency. All decisions and
parameters that govern live behaviour come from **four artifacts** in
MT5 Common Files (Section 5) — the EA is just the runtime that
executes that contract.

Validation: **purged 6-fold CV PF 1.41** (Python, frictionless cost
assumption), **Tester PF 1.25 / +11.83 % over 6 months** (MT5 with
real AvaTrade spread). Net of the +0.02 PF lift from the breakeven
move shipped in v1.20, expected live PF range is **1.15-1.25**.

## 2. The three-layer stack

```
   ┌──────────────────────────────────────────────────────────────┐
   │  LAYER 3 — Live runtime (MQL5 EA on MT5)                      │
   │     ea/MT5bot_m4Gold_MetaTrend.mq5      decision loop         │
   │     ea/includes/MetaGate.mqh            feature + ONNX runner │
   │     reads:  M4GOLD_METATREND_GOLD.onnx + spec.json            │
   │             (optional) M4GOLD_QUANTILE_q*_GOLD.onnx           │
   │             (optional) M4GOLD_TSPULSE_GOLD.onnx               │
   └──────────────────────────────────────────────────────────────┘
                                  ▲ ONNX + spec.json
   ┌──────────────────────────────┴───────────────────────────────┐
   │  LAYER 2 — Training (Python, XGBoost)                         │
   │     python/aurum/metatrend.py            features + label     │
   │     python/train_h7_metatrend.py         baseline meta-gate   │
   │     python/train_h8_quantile.py          7 quantile heads     │
   │     python/cv/purged_kfold.py            leak-free CV         │
   │     deploy gate: meanPF >= 1.15 AND excess >= 0.05 AND        │
   │                  min_fold_pf >= 0.92                          │
   └──────────────────────────────────────────────────────────────┘
                                  ▲ feature matrix + targets
   ┌──────────────────────────────┴───────────────────────────────┐
   │  LAYER 1 — Data (sibling mk4 repo, shared)                    │
   │     data/parquet/HYDRA4_5MFROMTICKS_GOLD.parquet              │
   │     data/parquet/HYDRA4_TBARS_GOLD_100tpb.parquet (research)  │
   │     data/ticks/HYDRA4_TICKS_GOLD.parquet (research)           │
   └──────────────────────────────────────────────────────────────┘
```

**Train/serve parity** is enforced by three rules:
1. The 18-feature definition lives in `aurum/metatrend.py` and the
   MQL5 reimplementation in `MetaGate.mqh` is kept in lock-step. The
   `python/test_metagate_parity.py` script verifies a sample.
2. The XGBoost model is exported to ONNX and the EA runs **that same
   ONNX** via `OnnxRun`. No re-training, re-quantising, or framework
   conversion at deploy time.
3. The model's contract (n_features, act_threshold, deploy flag) is
   carried in a sidecar `_spec.json` that the EA parses on init. The
   EA never hard-codes thresholds the trainer doesn't agree on.

## 3. What governs each trade decision — the four laws

These are the rules the live EA actually follows, in evaluation order
on each new **closed** M5 bar:

### 3.1 Law of direction — the trend rule

> The primary signal is `+1` (long) when **EMA(50) > EMA(200)** on the
> closed M5 series, else `-1` (short). No exceptions.

- Deterministic. Same MQL5 code, same Python pandas code.
- This is the only signal that decides *direction*. The meta-gate
  only decides *whether to act* on it.
- Defined in `aurum/metatrend.py::primary_signal` and mirrored in
  `MetaGate.mqh::MG_Primary`.

### 3.2 Law of consent — the meta-gate

> Enter only when the XGBoost meta-gate returns `P(act) ≥
> act_threshold` (from spec, currently `0.55`).

- The meta-gate predicts: "would following the primary signal over
  the next 240 M5 bars (~20 h) beat the round-trip cost?"
- Labels are net-of-cost: a trade is a positive example only if
  `primary * fwd_return - 1.8e-4 > 0`. Cost-aware by construction.
- 18 features (see Section 6) — all causal, all from closed bars.
- Trained with XGBoost classifier, exported to ONNX with parity
  ≤ 1.2 × 10⁻⁷ (verified per build).

### 3.3 Law of restraint — the deploy gate

> The model is allowed to trade live only if **all three** of these
> are met under leak-free purged 6-fold CV:
> - `meta_mean_pf ≥ 1.15`
> - `excess_vs_raw ≥ 0.05` (meta-gate must beat the raw EMA rule)
> - `min_fold_pf ≥ 0.92` (no single sub-period catastrophically loses)

Encoded in `train_h7_metatrend.py::_GATE_MIN_*` and surfaced in the
spec as `deploy: true|false`. If `false`, the EA stays idle unless
`InpRespectDeploy=false` is set manually for diagnostic runs.

The current production model: meanPF **1.406**, excess **+0.287**,
min fold **0.997** → `deploy: true`.

### 3.4 Law of exit — the four-mechanism engine

Order of evaluation per tick:

| # | Trigger | When it fires | What it does | Default |
|---|---|---|---|---|
| 1 | **Breakeven move** | profit ≥ `InpBreakevenAtr` × ATR(14) | SL ratchets to entry + small buffer | ON, 1.0 ATR |
| 2 | **Partial close** | profit ≥ `InpPartialAtr` × ATR | close `InpPartialPct` of the unit | OFF |
| 3 | **ATR trailing** | profit ≥ `InpTrailStartAtr` × ATR | SL trails at `InpTrailAtr` behind | ON, 2.0/3.0 ATR |
| 4 | **Trend flip / timeout** | EMA cross reverses OR ≥ `InpMaxHoldBars` bars old | close entire stack | ON, ~24 h |

Initial SL is set at `InpSlAtr × ATR` (default 3.0).
Optional fixed TP at `InpTpAtr × ATR` (default 0 = disabled).

This is **the answer** to the "no profit goal" critique. Breakeven
guarantees every trade that earns +1 ATR has a downside floor of
breakeven. The trail is the dynamic upside engine. No fixed TP
because trend-followers live on the long tail of winners.

## 4. The configuration surface — every input the EA exposes

Inputs are set per chart in the EA Properties dialog. Bold = touch with care.

### Identity / safety

| input | default | meaning |
|---|---|---|
| `InpMagic` | 49200 | unique magic number for this strategy's positions |
| `InpBaseLot` | 0.01 | lot size per unit (multiplied by Kelly if quantiles on) |
| **`InpRespectDeploy`** | true | sit idle if spec `deploy=false` |
| `InpVerboseLog` | true | print per-bar decisions to MQL5 journal |

### Risk and exits

| input | default | meaning |
|---|---|---|
| **`InpSlAtr`** | 3.0 | initial SL in ATR(14) multiples |
| `InpTpAtr` | 0.0 | hard TP in ATR (0 = disabled) |
| `InpUseBreakeven` | **true** | breakeven move at +`InpBreakevenAtr` ATR |
| `InpBreakevenAtr` | 1.0 | trigger threshold |
| `InpBreakevenBuffer` | 0.05 | tiny buffer above entry on BE move |
| `InpUsePartialClose` | false | partial close at +`InpPartialAtr` ATR |
| `InpPartialAtr` | 1.5 | trigger |
| `InpPartialPct` | 0.5 | fraction of unit to close |
| `InpUseTrailing` | true | enable the ATR trail |
| `InpTrailStartAtr` | 2.0 | profit threshold to activate the trail |
| `InpTrailAtr` | 3.0 | distance the SL trails behind price |
| `InpMaxHoldBars` | 288 | force-exit after N M5 bars (~24 h) |
| `InpExitOnFlip` | true | close on EMA trend reversal |

### Pyramiding

| input | default | meaning |
|---|---|---|
| `InpMaxStack` | 3 | max stacked units in a winning trend (1 = no stacking) |
| `InpStackStepAtr` | 1.0 | add a unit only after the stack is ahead by N×ATR per existing unit. Pyramid, never martingale |

### Quantile gate (v1.20 — dormant by default)

| input | default | meaning |
|---|---|---|
| **`InpUseQuantiles`** | **false** | OFF by default. The 7 ONNX heads load but are not consulted unless this is true |
| `InpQ10VetoAtr` | 2.5 | veto trade if q10 forecast worse than -N × ATR |
| `InpUseQ50Filter` | false | require q50 sign agrees with primary |
| `InpKellyFraction` | 0.25 | quarter-Kelly position sizing |
| `InpKellyMin/Max` | 0.5 / 2.0 | clamp for the lot multiplier |
| `InpSlBufferAtr` | 0.5 | extra buffer beyond q05 when dynamic SL is on |

## 5. File manifest — what lives where in MT5 Common Files

`%APPDATA%\MetaQuotes\Terminal\Common\Files\`:

| file | role | size | live? |
|---|---|---|---|
| `M4GOLD_METATREND_GOLD.onnx` | the 18-feature meta-gate classifier | ~290 KB | **YES — required** |
| `M4GOLD_METATREND_GOLD_spec.json` | act_threshold, deploy flag, n_features, CV report | ~1 KB | **YES — required** |
| `M4GOLD_QUANTILE_q{05,10,25,50,75,90,95}_GOLD.onnx` | 7 quantile regressor heads | 7×290 KB | research artifact (off by default) |
| `M4GOLD_QUANTILE_GOLD_spec.json` | quantile-head manifest + CV stats | ~2 KB | research artifact |
| `M4GOLD_TSPULSE_GOLD.onnx` | IBM Granite TS encoder, 22-feat variant | ~2 MB | shipped but dormant — see `RESEARCH_TSPULSE.md` |

`MT5\MQL5\Experts\MT5bot_m4Gold\ea\`:

| file | role |
|---|---|
| `MT5bot_m4Gold_MetaTrend.mq5` | EA source (v1.20) |
| `MT5bot_m4Gold_MetaTrend.ex5` | compiled binary |
| `includes\MetaGate.mqh` | feature builder + ONNX runner + quantile loader |

Other sibling EAs (`MT5bot_m4Gold_AURUM.mq5`, `_Dispatcher.mq5`, etc.)
exist as **deprecated** — they belong to earlier strategies that
failed the deploy gate. Don't attach them to a chart.

## 6. The 18 meta-gate features (canonical)

In the order the model expects them (in `META_FEATURES` in
`aurum/metatrend.py`):

| # | name | definition |
|---|---|---|
| 0 | `ret12` | log return over the last 12 M5 bars (1 h) |
| 1 | `ret48` | log return over 48 bars (4 h) |
| 2 | `ret96` | log return over 96 bars (8 h) |
| 3 | `atr14_norm` | ATR(14) / current close |
| 4 | `atr48_norm` | ATR(48) / current close |
| 5 | `rv24` | realised vol of 1-bar returns over 24 bars |
| 6 | `rv96` | realised vol over 96 bars |
| 7 | `pos96` | (close - 96-bar low) / (96-bar high - 96-bar low) |
| 8 | `pos288` | same, 288 bars (~1 day) |
| 9 | `ema_fast_dist` | (close - EMA50) / ATR14 |
| 10 | `ema_slow_dist` | (close - EMA200) / ATR14 |
| 11 | `ema_gap` | (EMA50 - EMA200) / ATR14 |
| 12 | `bars_since_hi96` | argmax position of the 96-bar high, normalised |
| 13 | `bars_since_lo96` | argmin position of the 96-bar low, normalised |
| 14 | `trend_age` | bars since the last EMA(50)/EMA(200) cross / 200, capped |
| 15 | `up_streak` | consecutive same-direction closes, clipped to ±1 |
| 16 | `hod_sin` | sin(2π × hour_of_day / 24) |
| 17 | `hod_cos` | cos(2π × hour_of_day / 24) |

Every column uses only past/current closed bars. Verified leak-free.

## 7. Training and validation — how new models are produced

```
# baseline 18-feature meta-gate (the production model)
python python/train_h7_metatrend.py
# experimental: append 7-quantile heads
python python/train_h8_quantile.py
# parity check between MQL5 and Python feature builders
python python/test_metagate_parity.py
```

The trainer:
1. Loads 174 k M5 GOLD bars (~2.5 years)
2. Computes 18 features + the 240-bar net-of-cost label
3. Splits into **purged 6-fold CV** with 1 % embargo around test folds
4. Scores **meta-gated PF** vs **raw primary PF** (control), per fold
5. Trains the final model on ALL data, exports to ONNX
6. Writes `_spec.json` with the deploy gate result

The MT5 Tester is the **second-stage** validation — runs the actual
EA `.ex5` against real broker spread history. Yesterday's run on the
v1.20 binary produced PF 1.25 over 6 months (Section 9).

## 8. Operations runbook

### Compile the EA

```powershell
$me  = "C:\Program Files\MetaTrader 5\metaeditor64.exe"
$src = "$env:APPDATA\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\MT5bot_m4Gold\ea\MT5bot_m4Gold_MetaTrend.mq5"
$log = "$env:TEMP\compile.log"
Start-Process $me -ArgumentList "/compile:`"$src`"","/log:`"$log`"" -Wait
```

Confirm `Result: 0 errors, 0 warnings`.

### Stage a new bundle

Copy the latest `.onnx` + `_spec.json` from `onnx_out/` to MT5
Common Files. The EA detects the spec on init.

### Run the Tester from CLI

```powershell
Start-Process "C:\Program Files\MetaTrader 5\terminal64.exe" `
  -ArgumentList "/config:`"path\to\tester.ini`""
```

See `tester_reports/tester.ini` for the canonical config. Output:
final balance line in the agent log; parse with
`python python/parse_tester_log.py`.

### Monitor the live demo

```powershell
.\tools\watch_mt5.ps1            # tail mode (Ctrl+C to stop)
.\tools\watch_mt5.ps1 -Summary   # one-shot snapshot
```

Filters terminal log + MQL5 journal for: EA load/remove, broker auth,
[MetaTrend] / [MetaGate] prints, deal events, severity > 0 errors.

## 9. Validation log — what's actually been measured

| date | run | result | notes |
|---|---|---|---|
| 2026-05-20 | Python purged CV (v1.10) | PF **1.406** | 5/6 folds positive, 1 break-even |
| 2026-05-20 | Tester 6mo (v1.10) | PF **1.250** / +$1,183 | first realistic-cost data point |
| 2026-05-21 | Tester 6mo (v1.20 all features ON) | **−$167** | hard negative; quantile + Kelly + dyn SL break it |
| 2026-05-21 | Tester A_baseline (v1.20 features OFF) | +$1,183 | confirms v1.20 ≡ v1.10 when off |
| 2026-05-21 | **Tester B_breakeven only** | **+$1,307** | **the shipped change** — +$124 over baseline |
| 2026-05-21 | Tester C_partial_close | +$1,183 | no effect on hedging-mode MT5 |
| 2026-05-21 | Tester D_quantile only | −$247 | dynamic SL widens stops, kills the edge |

**Current shipped configuration** = the A/B winner = v1.20 with
breakeven ON, everything else off.

## 10. Honest history — what we've already ruled out

If you're tempted to revisit any of these, read the linked doc first.
We have already paid the time to find out.

| direction | verdict | reference |
|---|---|---|
| Short-horizon scalping on GOLD | **No edge.** 39-hypothesis search, all fail after cost | `RESEARCH_FINDINGS.md` |
| AURUM v2 transformer / direction-net | **No edge.** Time-sync leak inflated early numbers; honest baseline is PF 1.0 | `m4gold_wayforward` memo |
| IBM Granite TSPulse-r1 as features | **No edge** when the model is made deterministic. The earlier +0.034 was random-mask noise the meta-gate was averaging over | `RESEARCH_TSPULSE.md` |
| Order-flow from tick-bar parquet | **No edge.** 5 of 8 microstructure columns are dead in this broker's tick feed | `RESEARCH_ORDERFLOW.md` |
| Gold/silver cointegration stat-arb | **No cointegration** on 2023-2026. Half-life 8 weeks to infinity. Beta unstable across quarters | `RESEARCH_STATARB.md` |
| 7-quantile distributional gate (research) | **Net negative** in Tester. Dynamic SL from q05 widens stops too much on a 44.7 %-win-rate strategy | `RESEARCH_QUANTILES.md` + Section 9 |
| Partial close on hedging-mode MT5 | **Zero effect.** Counter-deals net out | Section 9 (run C) |
| Fixed TP at fixed ATR multiple | **Caps upside on a trend follower.** Would reduce PF | rationale in user dialog, v1.20 left as opt-in |

## 11. The "what's the goal" answer — for the record

The strategy does **not** have a per-trade profit target. By design.

What it does have:
1. **A per-trade loss limit** — initial SL at 3 × ATR
2. **A per-trade breakeven floor** — once a trade reaches +1 × ATR
   profit, SL ratchets to entry. The trade can no longer become a
   loser
3. **An asymmetric upside** — once profit reaches +2 × ATR, the trail
   takes over and rides the trend
4. **A regime exit** — when the EMA cross reverses, the position
   closes regardless of P&L. This is "the regime that justified this
   trade is over"
5. **A statistical edge** — population-level PF 1.25 in realistic
   backtest, +13.07 % over 6 months. Most trades give back a little.
   A minority of trades carry the trend body and pay for the rest

This is trend-following 101 done with leak-free validation and
honest costs. It is not "guess the target price" — that mechanism
has no edge on this data, on this broker, at this timeframe (verified
across the negative results in Section 10).

## 12. What governs the program, restated in 5 lines

1. **Direction**: EMA(50/200) cross — deterministic
2. **Consent**: XGBoost meta-gate `P(act) ≥ 0.55` — learned from data, cost-aware label
3. **Restraint**: deploy gate (meanPF ≥ 1.15, beats raw rule, no fold worse than 0.92)
4. **Exit**: SL → breakeven move → ATR trail → trend flip / timeout, in that order
5. **Everything else** is parameters in [Section 4](#4-the-configuration-surface--every-input-the-ea-exposes), or research that didn't make the cut [Section 10](#10-honest-history--what-weve-already-ruled-out)
