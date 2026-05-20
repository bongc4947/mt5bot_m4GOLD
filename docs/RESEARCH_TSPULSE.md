# Iteration 6 — IBM Granite TSPulse-r1 as Meta-Gate Features

> **Updated: 2026-05-20.** The original v1 ablation found a +0.034 PF
> lift. While building the live EA wiring I discovered that
> tspulse-r1's reconstruction head runs *random* time and FFT masking
> at every forward pass, **even in eval mode** — the model is
> non-deterministic by design. The v1 "lift" was the meta-gate learning
> to average over that randomness, not a real signal. With the maskers
> patched to deterministic pass-through (mandatory for live serving),
> the lift disappears and tspulse becomes a mild **regression**. This
> document keeps both results so the lesson sticks.

## Setup

- **Model**: `ibm-granite/granite-timeseries-tspulse-r1` (main branch — anomaly
  variant, `TSPulseForReconstruction`). 1.08 M params, CPU-friendly, ctx 512.
- **Features extracted** (`python/aurum/tspulse_features.py`, strictly causal):
  1. `tspulse_fwd_logret` — net log-return forecast over the next 16 bars
  2. `tspulse_fwd_mag` — forecast magnitude in ATR-proxy units
  3. `tspulse_recon_err` — masked reconstruction error (anomaly score)
  4. `tspulse_fft_div` — KL of forecast FFT softmax vs observed (regime divergence)
- **Pipeline**: append to MetaTrend's 18 features → 22-feature meta-gate
  → leak-free 6-fold purged CV, net of cost.

## Result — full 174 k M5 bars

| metric | baseline (18 feat) | tspulse v1 (random mask) | **tspulse v2 (deterministic)** |
|---|---|---|---|
| meta meanPF | 1.406 | 1.433 (noise-driven) | **1.355** |
| raw primary PF (control) | 1.119 | 1.114 | 1.114 |
| excess vs raw | +0.287 | +0.320 (noise-driven) | **+0.241** |
| min fold PF | 0.997 | 1.021 | 0.979 |
| folds all-positive | 5 / 6 | 6 / 6 | 5 / 6 |
| reproducible same-input output | n/a | NO | YES |
| ONNX parity (scalar level) | n/a | 1e-2 (loose) | **4e-7 (tight)** |

**TSPulse is not deployable for this strategy.** Once we make the
model deterministic — a hard requirement for live serving, since the
EA cannot evaluate features that change between calls on the same M5
bar — the lift disappears. Final call: **production stays on the
18-feature baseline.**

## What we learned

- **The model is stochastic by design.** `TSPulseForReconstruction`
  runs random time-masking (`mask_ratio=0.7`) and FFT magnitude
  masking on *every* forward call, even in `eval()` mode. That isn't a
  bug — it's how the self-supervised masked-reconstruction objective
  works at inference. But it means raw outputs vary by ~$1 on a $2000
  GOLD price between consecutive calls on identical input.
- **The noisy "lift" was the meta-gate learning to average over
  randomness.** Training on a stochastic feature distribution lets the
  meta-gate find robust decision regions; live serving picks one
  sample of that distribution and gets a worse signal. Classical
  train/serve gap, dressed up as a foundation-model win.
- **Always check determinism before trusting a feature.** Run the same
  input twice; if the output differs by more than float-rounding, you
  have a problem.

## Deployment status

- Production artifacts in `onnx_out/M4GOLD_METATREND_GOLD.*` remain
  the **18-feature baseline** (PF 1.406).
- The EA wiring is in place (auto-detects `n_features` from the spec
  and loads `M4GOLD_TSPULSE_GOLD.onnx` only for the 22-feature
  variant) — see [ea/includes/MetaGate.mqh](ea/includes/MetaGate.mqh).
  It is dormant code: useful capability for any future foundation-TS
  experiment, but not active in the shipping bundle.
- `python/export_tspulse_onnx.py` patches the time and FFT maskers to
  pass-through before tracing, then verifies bit-stable scalar parity.
  This is the right starting point if a future TS foundation model
  *is* deterministic in eval.

## Reproduce

```
python python/train_h7_metatrend.py                  # 18-feature baseline
python python/train_h7_metatrend.py --with-tspulse   # 22-feature ablation
```

First tspulse run pulls the model (~5 MB) and computes features for all
174 k anchors (~7 min on CPU). Subsequent runs hit the on-disk cache
(`onnx_out/tspulse_features_GOLD_*.npy`) and are instant.
