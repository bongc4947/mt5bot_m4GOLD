# MT5bot_m4Gold — Strategy & Way-Forward

> Reference document. Captures what was learned through the AURUM build +
> the leak-free edge search, and sets the forward strategy around three
> ideas the operator named: **point the machine properly**, **give it
> sense organs**, **feed it the proper information**.

---

## 1. What was learned (the journey)

| Stage | Finding |
|---|---|
| AURUM v2 built | Patch-transformer + SSL + conformal + meta-gate — modern, complete AI stack |
| First metrics | PF 2.4–3.98 — looked excellent |
| **Time-sync leak found** | Multi-timeframe features ended on the *current* HTF bar, which aggregates *future* M5 bars. Every metric was inflated. |
| Leak fixed | causal HTF anchors + closed-bar EA reads + clock-aligned resample |
| Leak-free re-test | Baselines collapsed PF 2.4 → **1.0**. The apparent edge **was the leak.** |
| Edge search | 39 hypotheses, purged-CV, cost-aware (`python/aurum/research.py`) |
| **Verdict** | M5 GOLD *direction* prediction has **no edge** (PF ~1.0) at any horizon. **Meta-labelled slow trend** has a **real edge** (PF 1.19–1.27, every fold positive, survives 4× cost). |

Multi-period MT5 backtests (2024–2026) independently confirmed the
M5-direction model trades poorly.

## 2. The core principle (the lesson)

> **Edge = Information × the right Target × honest Validation.
> NOT model sophistication.**

A transformer fine-tuned to perfection cannot extract signal that is not
in its inputs, or predict a target the market does not make predictable.
The sophisticated AURUM stack on M5-direction scored a coin flip; a plain
meta-labelled trend rule beat it. The machinery is good — it was aimed at
the wrong target and fed the wrong information.

## 3. The diagnosis — "blindness" = missing sense organs

The model is not lacking a brain. It is fed only *"what just happened"* —
returns, ranges, volatility. A discretionary trader's intuition reads
*"where are we"* — market **structure**: swing highs/lows, support /
resistance, break vs. reject, trend sequence, regime. The model is blind
to all of that. Not a missing brain — **missing sense organs.**

Intuition is not mystical. It decomposes into computable, causal features.

## 4. The strategy — four pillars

### Pillar 1 — Sense organs (new information layers)
Feed the model what a trader actually reads:
- **Market structure** — swing pivots, HH/HL/LH/LL sequence, support /
  resistance levels + proximity (ATR units), break-vs-reject.
- **Regime** — trend / range / high-vol, as a first-class input AND gate.
- **Order flow** — `ofi`, `cvd`, `taker_ratio`, `microprice_drift` already
  sit unused in the tick-bar parquet. Wire them in.
- **Volatility state** — realised-vol term structure, vol-of-vol,
  regime transitions (volatility is genuinely predictable, ~66% acc).
- **Session / calendar** — time-of-day, session overlaps.

### Pillar 2 — Point the machine at targets that HAVE signal
Stop predicting M5 direction. Predict:
- **Meta-label of a trend signal** — validated, PF 1.25.
- **Meta-label of a mean-reversion / cointegration signal** — gold-silver
  stat-arb (to be tested).
- **Volatility state** — as a gate / position sizer.
- **Regime** — as the master gate over which strategy may act.

### Pillar 3 — Honest validation (already built)
Purged + embargoed CV, leak-free, cost-aware, every-fold-positive bar.
`python/aurum/research.py` is the fast search engine; the deploy gate is
the ship/no-ship rule.

### Pillar 4 — Build, gate, combine
Validated edges → trainers → ONNX → EA, each gated on `deploy: true`.
Combine *uncorrelated* edges (trend + stat-arb + vol-gating) into a
portfolio — that is how returns compound, not by a smarter single model.

## 5. Phased roadmap

| Phase | Work | Gate to pass |
|---|---|---|
| **A** | Market-structure feature search — pivots, S/R, break/reject, regime gating; re-run the leak-free harness | structure features lift purged-CV PF |
| **B** | Order-flow feature search — wire `ofi`/`cvd`/`taker_ratio` from tick-bars | order-flow lifts purged-CV PF |
| **C** | Gold-silver cointegration stat-arb research (iteration 6) | spread strategy clears cost-surviving PF |
| **D** | Build the validated edges — meta-trend first (`train_h7_metatrend` + EA), then whatever cleared A/B/C | all-folds-positive, beats baseline |
| **E** | Portfolio — combine uncorrelated edges; regime decides which is active | combined Sharpe ≥ best single edge |

Each phase is independently gated. A phase that fails its gate is a
*finding*, not a failure — it tells us that information/target carries
no edge, and we stop spending on it.

## 6. Decision log — what is settled

- M5-direction AURUM is **retired** as a primary signal. Confirmed dead.
- The transformer / SSL / conformal machinery is **kept** — re-pointed at
  the meta-gate of strategies whose target has signal.
- **Phase A DONE** — market-structure features (pivots, S/R, trend-age,
  streak, ema-gap) lifted the meta-trend edge from PF 1.27 → **1.41**.
- **Phase D DONE (meta-trend)** — `python/train_h7_metatrend.py` +
  `ea/MT5bot_m4Gold_MetaTrend.mq5` + `ea/includes/MetaGate.mqh` shipped.
  Net PF 1.41 purged-CV, `deploy=true`, ONNX parity clean. This is the
  first field-testable strategy with a measured, leak-free edge.
  See `docs/DEPLOY_METATREND.md`.
- Remaining: Phase B (order-flow features), Phase C (gold/silver
  stat-arb), Phase E (portfolio combine).
