# Iteration 6 — IBM Granite TSPulse-r1 as Meta-Gate Features

> Honest ablation: does appending TSPulse-derived features to MetaTrend's
> 18-feature set lift the leak-free purged-CV edge?

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

| metric | baseline (18 feat) | **with tspulse (22 feat)** |
|---|---|---|
| meta meanPF | 1.406 | **1.440** |
| raw primary PF (control) | 1.119 | 1.114 |
| excess vs raw | +0.287 | **+0.326** |
| min fold PF | 0.997 (one break-even) | **1.062** |
| folds all-positive | 5 / 6 | **6 / 6** |
| ONNX parity | 1.19 e-7 ✓ | 1.19 e-7 ✓ |

**TSPulse modestly helps.** Headline PF gain is small (+0.034), but the
structural improvement is real — every purged-CV fold now strongly
positive, no break-even sub-period.

## What this does and does not mean

- **Does mean**: the four causal scalars derived from tspulse carry
  information the 18-feature set didn't fully capture, in a way the
  meta-gate can exploit. This is a real (if small) lift.
- **Does NOT mean** "foundation TS model = silver bullet". The 30 k-bar
  smoke run showed tspulse hurt slightly on a hostile sub-window. The
  edge is *small and depends on having a large training set*.

## Deployment status

The training-side experiment is **operational** (`train_h7_metatrend.py
--with-tspulse`) and the result is preserved here. The **production
artifacts in `onnx_out/M4GOLD_METATREND_GOLD.*` remain the 18-feature
version** (PF 1.406) — same as the bundle already staged in MT5 Common
Files. Deploying the tspulse variant requires additional MT5-side work:

- Export tspulse-r1 to ONNX (the model already has a sister ONNX repo
  `onnx-community/granite-timeseries-patchtst` — tspulse needs separate
  conversion).
- Load it inside MT5 alongside the meta-gate ONNX.
- Implement the 4 tspulse features in `MetaGate.mqh` (calls the tspulse
  ONNX on the past-512-close window per M5 decision bar).
- Train/serve parity test.

That is a real ~2-3 hour build with operational complexity (a second
ONNX model in the live EA, 512-bar context per inference). Whether it's
worth +0.034 PF for the extra moving part is a deployment judgement.

## Reproduce

```
python python/train_h7_metatrend.py                  # 18-feature baseline
python python/train_h7_metatrend.py --with-tspulse   # 22-feature ablation
```

First tspulse run pulls the model (~5 MB) and computes features for all
174 k anchors (~7 min on CPU). Subsequent runs hit the on-disk cache
(`onnx_out/tspulse_features_GOLD_*.npy`) and are instant.
