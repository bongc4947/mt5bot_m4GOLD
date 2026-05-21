# Research — 7-Quantile Distributional Forecast for MetaTrend

> Question: can we replace MetaTrend's binary `P(act)` with a richer
> 7-quantile forecast of the 240-bar forward return, and use those
> quantiles to drive sizing, SL placement, and entry filtering?
>
> Short answer: yes, the engineering is well-defined, the math is
> established enterprise practice, and the most likely lift is
> +0.02-0.10 PF. The build is ~2 days. Whether it's worth it depends on
> whether the live demo PF (when it comes in) is comfortably above the
> 1.15 ship gate or sitting at the edge.

## What "7 quantiles" buys you that a binary gate doesn't

The current meta-gate emits one scalar: `P(act) ∈ [0, 1]`. That's the
probability the EMA trend rule beats cost over the next 240 bars.
What it does NOT tell you:

- **Asymmetry.** A trade scoring `P(act) = 0.60` could be a +30 pt /
  −5 pt expected outcome OR a +5 pt / −30 pt mess that just happens to
  win 60% of the time. The binary score collapses both into the same
  number. Quantiles do not.
- **Tail risk.** Two trades with the same median forecast can have
  very different left-tail risk. A q05 of -50 pts vs -8 pts is a 6x
  difference in capital-at-risk that the meta-gate is blind to.
- **Position sizing.** Currently `InpBaseLot = 0.01` is a constant.
  The Kelly-optimal lot size needs **expected return AND variance** —
  both of which fall out of quantile estimates trivially.
- **SL placement.** Currently `InpSlAtr = 3.0`, fixed. The right SL is
  "just past where the price is plausibly going against me" — i.e.,
  a tail quantile of the forward return, not a constant ATR multiple.

## The 7 quantile slots and what each does

I propose the standard financial-risk septile, mapped to specific
decisions inside the EA:

| quantile | name | what it tells us | EA decision it informs |
|---|---|---|---|
| **q05** | extreme downside | bottom 5% of forward returns conditional on features | initial **SL placement**: place at `price + q05_pts` (long) or `price - q05_pts` (short), instead of fixed 3 ATR |
| **q10** | bad tail | left-tail of plausible outcomes | **entry filter**: refuse trade if `q10 < -K * ATR` (fat left tail veto) |
| **q25** | lower IQR | Q1 of forward return distribution | **asymmetry check**: combine with q75 to measure spread / uncertainty |
| **q50** | median | point forecast of forward return | **trend confirmation**: sign of q50 must agree with primary signal; magnitude scales position size |
| **q75** | upper IQR | Q3 of forward return distribution | **realistic target**: trail SL when price reaches q75 to lock in baseline gain |
| **q90** | bull tail | right-tail of plausible outcomes | **trail behaviour**: loosen trail (let winner run further) when q90 is far above current |
| **q95** | extreme upside | top 5% conditional forecast | **size-up trigger**: if q95 is unusually high AND q05 is bounded → conviction trade, scale up via fractional Kelly |

These seven quantiles cover everything risk management actually cares
about: VaR95 (= -q05 for losses), VaR99 ≈ -q01 extrapolation, the
inter-quartile range, the median forecast, and the upside tail.

## Enterprise-grade calculation engines

Four production-quality options, ordered by fit-with-current-pipeline:

### 1. XGBoost 2.0+ `reg:quantileerror` with `quantile_alpha` array — **recommended**

- Train **one** model that simultaneously predicts all 7 quantiles
- Same library, same ONNX export path, same MetaGate.mqh wiring as the
  current binary gate
- `quantile_alpha=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]`,
  `tree_method="hist"` (the exact tree method is broken for quantile),
  `multi_strategy="multi_output_tree"` so all 7 quantiles share splits
- Known limitation: quantile crossing — XGBoost does not enforce
  `q05 ≤ q10 ≤ ... ≤ q95`. Documented issue (see sources below). Fix:
  post-hoc isotonic regression across the 7 outputs, or sort the
  vector — cheap, runs in <1 ms in MQL5

See [XGBoost 2 quantile regression docs](https://xgboost.readthedocs.io/en/release_2.0.0/python/examples/quantile_regression.html).

### 2. Conformalized Quantile Regression (CQR) — Romano, Patterson, Candès (NeurIPS 2019)

Wraps any quantile estimator (XGBoost qreg, LightGBM, NN) and gives
**finite-sample coverage guarantees**. If you ask for 90% coverage on
the [q05, q95] interval, CQR mathematically guarantees that
**out-of-sample, 90% of realized forward returns will fall inside that
interval** — regardless of how well the underlying quantile model
fits.

This is the single biggest argument for going quantile over staying
binary: a calibrated coverage probability is **honest**, where a
classifier score is just a number. You can size positions confidently
because the "1-in-20 chance the price moves against me beyond q05" is
a real probability, not a model artefact.

Implementation: holdout 20% of training data as a calibration set, run
CQR's nonconformity score adjustment, output adjusted quantile bands.
Reference: [Romano et al., Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222),
official code at [yromano/cqr](https://github.com/yromano/cqr).

### 3. LightGBM `objective='quantile'` — alternative, more mature for QR

- Microsoft's gradient booster, often slightly faster than XGBoost
- Worse fit for our stack: needs **7 separate models** for 7 quantiles
  (no native multi-output), 7 ONNX files in production
- Quantile crossing problem is even worse than XGBoost — see
  [LightGBM issue #5727](https://github.com/microsoft/LightGBM/issues/5727)

Not recommended given we're already on XGBoost.

### 4. statsmodels `QuantReg` — linear baseline, sanity check only

- 1978 Koenker-Bassett linear quantile regression
- Fast, interpretable, but linear — under-fits the meta-gate features
- Use as a **sanity baseline** during ablation: if non-linear XGBoost
  qreg can't beat the linear baseline by a clear margin on pinball
  loss, something is wrong with the setup

## Concrete integration in MetaTrend

### Training side (`python/train_h7_metatrend_quantile.py` — new file)

```python
from xgboost import XGBRegressor

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

qmodel = XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=QUANTILES,
    tree_method="hist",
    multi_strategy="multi_output_tree",
    n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
# X is the same 18-feature matrix; y is the forward 240-bar log return
qmodel.fit(X_train, y_fwd_logret)
# At inference: qmodel.predict(X) -> [N, 7]
```

### EA side (`MetaGate.mqh` extension)

```mql5
// new in MetaGate.mqh
float g_q[7];   // q05, q10, q25, q50, q75, q90, q95 in log-return units

bool MG_QuantileForecast(float &q_out[]) {
   // run the 7-quantile ONNX, fill q_out
   // sort/clip to enforce non-crossing
}
```

### Decision logic (`MT5bot_m4Gold_MetaTrend.mq5` v1.20)

```mql5
// On a new M5 bar:
if (MG_ActProb() < InpActThr) return;       // existing P(act) gate
if (!MG_QuantileForecast(q)) return;
// === new quantile gates ===
double q10_pts = ATRpts * (exp(q[1]) - 1.0); // q10 forward return → points
if (q10_pts < -InpQ10VetoAtr * g_atr14) return;   // fat-left-tail veto
double q50_dir = (q[3] > 0) ? +1 : -1;
if (q50_dir != prim) return;                  // median must agree with primary
// === quantile-aware sizing ===
double mu = q[3];
double sigma = (q[5] - q[1]) / 2.56;          // IQR-implied sigma (q90-q10)
double kelly_f = mu / (sigma * sigma + 1e-9);
double lot = InpBaseLot * clip(kelly_f * InpKellyFraction, 0.5, 3.0);
// === quantile-aware SL ===
double sl_pts = -ATRpts * (exp(q[0]) - 1.0);  // place SL just past q05
sl = price - dir * (sl_pts + InpSlBufferATR * g_atr14);
```

## Pros

1. **Richer signal** — meta-gate becomes "P(act) ≥ thr AND tail is
   tolerable AND median agrees with trend". Three filters where there
   was one.
2. **Asymmetric trade detection** — the gate can now distinguish "60%
   win-rate on +30/-5 setup" from "60% win-rate on +5/-30 setup", and
   reject the latter. Currently invisible to the EA.
3. **Cost-aware sizing without fitting a separate model** — Kelly
   fraction falls out of `mu = q50` and `sigma ≈ (q90 - q10) / 2.56`
   for free.
4. **Dynamic SL placement** — fixed 3 ATR is replaced by a
   forecast-conditional SL. Quiet bar → tight SL. Volatile bar → wide
   SL. This is the right behaviour, the current EA has it as a
   constant.
5. **Drawdown control via tail filter** — q10 veto skips the worst
   setups. The Tester run yesterday had a -$217.21 worst trade; a q10
   filter would likely have skipped that one.
6. **Conformal coverage = honest probabilities** — CQR gives you "I
   am 90% confident this trade's outcome falls in [q05, q95]" with a
   real coverage guarantee. Position sizers and risk reports can rely
   on that.
7. **Compatible with existing infrastructure** — same XGBoost, same
   ONNX pipeline, same purged-CV gate. The current 18-feature meta-gate
   stays; quantile model lives alongside it.

## Cons

1. **Quantile crossing is a real engineering issue.** XGBoost,
   LightGBM, and most quantile-NN models do **not** enforce
   monotonicity across the 7 outputs. Out-of-sample you will see
   `q75 < q25` on some bars unless you isotonize. The fix is cheap
   (sort the vector), but the fact that it's needed is a code smell
   — your model is admitting it doesn't fully understand the
   distribution. Composite-loss arctan-pinball training
   (arXiv:2406.02293) is the proper fix; non-trivial implementation.
2. **More ONNX in production.** Either one big 7-output ONNX (XGBoost
   `multi_strategy="multi_output_tree"`) or 7 separate ONNX (LightGBM
   path). Either way, double the inference cost per M5 decision.
   Probably still <5 ms on CPU; not a real blocker.
3. **Training surface area triples.** 7 quantile heads instead of 1
   classifier head, all needing to be trained, validated, parity-
   tested, and CV-scored. The pinball-loss metric for quantile
   accuracy is not directly comparable to PF. We will need both:
   pinball-loss for the quantile head, plus end-to-end PF backtest of
   the full integrated EA.
4. **Tail estimation on 174 k bars is noisy.** The q05 and q95 are the
   most useful AND the least reliable — by definition you only have
   5% of the data informing each tail estimate. Bootstrap confidence
   intervals on q05 are typically wide. CQR partially fixes this with
   coverage guarantees, but the underlying point estimates are still
   noisy.
5. **Adds complexity to a strategy that already passes the gate.** The
   18-feature MetaTrend cleared PF 1.41 in CV / 1.25 in Tester. Live
   demo data is **not in yet.** Building elaborate refinements before
   you have a live PF number to compare against is exactly the
   "optimise pre-evidence" failure mode I keep flagging. The right
   sequence is: live demo for 14 days → see real PF → decide if the
   quantile refinement is worth the complexity.
6. **Most likely lift is modest.** Realistic ablation prior:
   - +0.02 to +0.05 PF if tail-veto and sizing-by-quantile both work
   - +0.05 to +0.10 PF if the dynamic SL placement also lifts (because
     the current 3 ATR SL is likely a sub-optimal blanket)
   - Net new-PF estimate: **1.30 → 1.40 Tester PF range**. That is
     similar to or slightly better than the binary gate. Not a game
     changer.
7. **MQL5 doesn't ship `clip`, `Kelly`, or quantile inversion
   functions.** All of that math has to be re-implemented in MQL5,
   parity-tested against the Python side, and kept in lock-step with
   training. The MetaGate.mqh file will roughly double in size.
8. **Risk of overfitting to backtest.** Adding 7 new degrees of freedom
   to a strategy whose validation is a single 174k-bar CV is a
   significant overfit risk. The PF lift you see in CV is upper-bound;
   live will likely be lower.

## Recommendation

**Defer for now. Build after the live demo data is in.** Specifically:

1. **Step 1 (today's reality)**: Live demo MetaTrend (18-feature
   binary gate, current build) is attached and running. Let it
   accumulate 14 days. Capture the realised PF.
2. **Step 2 (in 14 days)**: Compare live PF to the Tester baseline of
   1.25. If live PF is comfortably above 1.15 (the deploy gate) — say
   1.20+ — you do not need quantile refinement to ship. The strategy
   works. Skip to Phase D (multi-asset / longer horizons).
3. **Step 3 (only if live PF is borderline)**: If live PF sits at
   1.10-1.18 — i.e., real but marginal — the quantile-augmented EA
   becomes the right next iteration. Expected lift puts you back to
   the 1.20-1.30 zone with honest coverage guarantees.
4. **If you choose to build it anyway** (research value, not deploy
   need): **~2 days of work**, structured as
   - **Day 1**: train_h7_metatrend_quantile.py + XGBoost 2.0
     multi-quantile + CQR calibration + pinball-loss CV report + ONNX
     export with scalar parity test (the lesson from the tspulse
     non-determinism bug). End-to-end PF backtest comparing v1 binary
     gate vs v2 quantile-augmented.
   - **Day 2**: MetaGate.mqh quantile loader, MetaTrend.mq5 v1.20 with
     the new decision logic, MQL5 vs Python parity test using the
     existing test_metagate_parity.py harness, Tester run for direct
     PF comparison vs yesterday's 1.250 baseline.

I'd rather see the live demo number first. The quantile build is real
value if needed, but it's value gated on evidence we don't have yet.

## Sources

- [XGBoost 2 Quantile Regression docs](https://xgboost.readthedocs.io/en/release_2.0.0/python/examples/quantile_regression.html)
- [XGBoost issue #9848 — multi-quantile crossing](https://github.com/dmlc/xgboost/issues/9848)
- [Romano, Patterson, Candès — Conformalized Quantile Regression (NeurIPS 2019)](https://arxiv.org/abs/1905.03222)
- [yromano/cqr — official CQR implementation](https://github.com/yromano/cqr)
- [LightGBM issue #5727 — preserving monotonicity across multiple quantiles](https://github.com/microsoft/LightGBM/issues/5727)
- [LightGBM issue #3371 — monotone constraint broken with quantile distribution](https://github.com/microsoft/LightGBM/issues/3371)
- [arXiv:2406.02293 — Composite Quantile Regression with XGBoost using Arctan Pinball Loss](https://arxiv.org/pdf/2406.02293)
- [arXiv:2111.04805 — Solution to the Non-Monotonicity and Crossing Problems in Quantile Regression](https://arxiv.org/pdf/2111.04805)
- [QuantPedia — Beware of Excessive Leverage: Kelly and Optimal F](https://quantpedia.com/beware-of-excessive-leverage-introduction-to-kelly-and-optimal-f/)
