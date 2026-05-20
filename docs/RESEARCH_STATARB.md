# Iteration 8 — Gold/Silver Cointegration Stat-Arb

> Hypothesis: gold and silver share a long-run equilibrium (the
> historical gold/silver ratio); a rolling-beta cointegration spread
> should mean-revert and admit a tradeable edge.
>
> **Verdict: it doesn't.** Not on 2023-09 to 2026-04 H1 data. Honest
> negative result. Don't ship.

## The cointegration evidence

Engle-Granger on log(GOLD) ~ alpha + beta*log(SILVER):

| sample | beta | R² | ADF stat | ADF p | cointegrated at 5%? |
|---|---|---|---|---|---|
| Full 2023-09 to 2026-04 (14 057 H1) | 0.72 | 0.89 | -1.83 | 0.37 | **NO** |
| Full M5 (168 k bars) | 0.72 | 0.89 | -1.87 | 0.34 | **NO** |
| 2025 Q1 | 1.06 | — | -3.02 | 0.033 | yes (barely) |
| 2025 Q4 | 0.24 | — | -2.98 | 0.037 | yes (barely) |
| every other quarter | 0.16-0.75 | — | > -3.0 | > 0.07 | NO |

Two observations:
1. **The R² 0.89 is a correlation trap.** It just says gold and silver
   move together — not that their spread is stationary. The ADF on the
   residual is the actual test, and it fails.
2. **Beta is unstable.** It swings from 0.16 to 1.06 across quarters.
   Any strategy that assumes a stable beta loses; any strategy that
   adapts beta fast enough has to use a window short enough to be noisy.

## Parameter sweep — does ANY config work?

36 configurations on H1: beta_window ∈ {240, 480, 720, 1440}, z_window ∈
{120, 240, 480}, (entry, stop) ∈ {(1.5, 3.0), (2.0, 3.5), (2.5, 4.0)}.
Spread strategy nets per-leg cost.

```
best:      beta=240  z=480  entry=2.5  stop=4.0  PF=1.074  trades=43  HL=2647 bars (4 months)
median:    PF ≈ 1.005  (coin flip net of cost)
worst:     PF=0.962
```

The "best" is overfit — 43 trades over 2.5 years on a single backtest,
PF that won't survive purged CV. And the **half-life of mean reversion
is 4 months in the best case, ranging up to infinity** — meaning the
spread is essentially a random walk on tradeable horizons. Even if
mean reversion *exists* on a multi-year scale, no realistic capital
allocator will hold a leveraged pair trade through a 4-month drawdown
hoping the spread comes back.

## Why this likely fails

- **Post-2011 decoupling.** Gold and silver had textbook cointegration
  from 1985-2007 driven by industrial+monetary demand sharing.
  Silver's industrial use (~50% of demand) since the solar/EV boom
  has decoupled it from gold's monetary-haven role. The "gold/silver
  ratio mean-reverts" thesis worked in the 90s; doesn't now.
- **Different supply shocks.** Silver supply is industrial-mining
  dominated (lead/zinc/copper byproducts); gold supply is
  monetary-driven (central bank buys). They respond to different
  inputs.

## Reproduce

```
python python/research_gold_silver_statarb.py
```

Confirms: meanPF 0.994 across 6 purged-CV folds, half-life 24 863 bars,
deploy=False.

## Status

- **Implementation**: kept as
  [python/research_gold_silver_statarb.py](python/research_gold_silver_statarb.py)
  for future re-test (re-running on a longer history or after a regime
  shift would not be unreasonable).
- **Production**: unchanged. MetaTrend (18-feature, PF 1.41) remains
  the single deployable strategy.
- **Stat-arb retried later?** Only worth it if (a) a credible regime
  change re-couples gold and silver (e.g. a deflationary shock), or
  (b) we widen the pair search to industrial-metals baskets — copper /
  platinum / palladium / silver — where the broader basket might
  cointegrate more stably than the gold/silver pair alone.
