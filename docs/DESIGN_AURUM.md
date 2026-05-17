# AURUM — MT5bot_m4Gold v2 AI Architecture

> Status: **accepted design, phased implementation**. This document is the
> single source of truth for the v2 AI layer. The legacy mk4 heads
> (`python/models/`) stay until AURUM clears its deploy gate.

AURUM is the GOLD-only, ONNX-deployable instantiation of the six-layer
awareness design. It is constrained by three hard realities:

1. **Inference runs inside MT5 via ONNX.** Every deployed component must
   export to an opset MetaTrader's ONNX runtime supports.
2. **Single instrument (GOLD).** Multivariate-across-symbols methods do
   not apply; multivariate-across-*timeframes* and across-*microstructure
   channels* do.
3. **Honest priors.** Plain transformers frequently lose to linear models
   on noisy financial series (Zeng et al., AAAI 2023). AURUM keeps strong
   simple baselines as the control and only promotes a deep model if it
   beats them on **purged** cross-validation.

---

## 1. Research grounding

| Layer | Idea | Primary source |
|---|---|---|
| L0 | Self-supervised masked-patch + contrastive pretraining | PatchTST self-supervised (Nie et al., ICLR 2023); TS2Vec (Yue et al., AAAI 2022) |
| L1 | Patch-transformer backbone, channel-independent | PatchTST (Nie et al., ICLR 2023) |
| L2 | Cross-timeframe attention | Crossformer (Zhang & Yan, ICLR 2023) |
| L3 | Multi-task heads incl. quantile regression | Temporal Fusion Transformer (Lim et al., 2021) |
| L4 | Meta-labeling — P(primary signal correct) | López de Prado, *Advances in Financial ML*, 2018, Ch.3 |
| L5 | Conformal calibration of confidence | Conformalized Quantile Regression (Romano et al., NeurIPS 2019) |
| L6 | Quantile-driven fractional-Kelly sizing | Kelly (1956); vol targeting (Moskowitz et al., 2012) |
| CV | Purged + embargoed K-fold, combinatorial purged CV | López de Prado, *AFML*, 2018, Ch.7 |
| Baseline | DLinear honest control | Zeng et al., AAAI 2023 |

---

## 2. Architecture

```
unlabeled GOLD tick-bars
        │
   L0  self-supervised patch encoder  (masked reconstruction + contrastive)
        │  frozen weights
   L1  patch-transformer backbone  ×3   (M5 / M15 / H1)
        │
   L2  cross-timeframe fusion (cross-attention)
        │
   L3  heads: direction(3) · quantile(3) · execution(3) · regime(4)
        │
   L4  meta-label gate (XGBoost)  →  P(act)
        │
   L5  conformal calibration  →  trade only on singleton confidence set
        │
   L6  quantile-Kelly sizing  →  lot multiplier
        │
   ONNX bundle  →  MT5 EA (AurumAgent.mqh)
```

### Input contract (deployed model)

A single flat tensor keeps MT5 ONNX I/O trivial:

```
input  : float[1, 2048]   layout = [ M5(128 bars × 8 ch),
                                     M15(64 bars × 8 ch),
                                     H1(64 bars × 8 ch) ]   row-major [bar, ch]
output : float[1, 13]     layout = [ dir_logits(3), quantiles(3),
                                     exec(3), regime(4) ]
```

8 channels per bar: `ret, hl_range, body, upper_wick, lower_wick,
signed_vol, vol_ratio, atr_norm`. Channel normalisation stats are baked
into the spec JSON, so the EA feeds raw channels and the model is
self-normalising (BatchNorm running stats + stored mean/std).

### Deployed artifacts

| File | What |
|---|---|
| `M4GOLD_AURUM_GOLD.onnx` | main net — float[1,2048] → float[1,13] |
| `M4GOLD_AURUM_META_GOLD.onnx` | meta-label XGBoost — float[1,13] → float[1,2] |
| `M4GOLD_AURUM_GOLD_spec.json` | conformal thresholds, sizing params, channel norm stats, contract dims, `deploy` flag |

---

## 3. Module map

```
python/
├── cv/purged_kfold.py        Phase 1 — PurgedKFold + CombinatorialPurgedCV
├── baselines/
│   ├── dlinear.py            Phase 1 — DLinear honest baseline
│   └── xgb_direction.py      Phase 1 — XGBoost direction baseline
├── aurum/
│   ├── aurum_config.py       all v2 hyperparameters
│   ├── datamodule.py         Phase 1 — multi-timeframe dataset builder
│   ├── backbone.py           Phase 2 — patch-transformer encoder
│   ├── pretrain.py           Phase 2 — self-supervised L0 training
│   ├── fusion.py             Phase 3 — cross-timeframe attention
│   ├── heads.py              Phase 3 — multi-task heads
│   ├── model.py              Phase 3 — AurumNet assembly + export wrapper
│   ├── meta_label.py         Phase 4 — XGBoost meta-gate
│   ├── conformal.py          Phase 4 — split-conformal calibration
│   ├── sizing.py             Phase 5 — quantile-Kelly sizing
│   └── export.py             ONNX bundler + ORT parity check
├── train_aurum.py            unified CLI (subcommands per phase)
└── ea/includes/AurumAgent.mqh  MT5 multi-head ONNX loader
```

---

## 4. Training pipeline

| Stage | Where | Notes |
|---|---|---|
| L0 pretrain | Cloud GPU | SSL on years of tick-bars; heavy |
| L1–L3 fine-tune | Local GPU or cloud | small once encoder frozen |
| L4 meta-label | Local CPU | XGBoost, seconds |
| L5 conformal | Local CPU | calibration arithmetic |
| Export + parity | Local | every head checked vs PyTorch |

`hardware_detector.py` picks device/batch automatically — AURUM reuses it.

CLI:

```
python python/train_aurum.py baseline      # Phase 1 control numbers
python python/train_aurum.py pretrain      # Phase 2 L0 SSL encoder
python python/train_aurum.py finetune      # Phase 2-3 backbone + heads
python python/train_aurum.py meta          # Phase 4 meta-label gate
python python/train_aurum.py conformal     # Phase 4 calibration
python python/train_aurum.py export        # ONNX bundle
python python/train_aurum.py all           # full pipeline
```

---

## 5. Validation methodology (non-negotiable)

- **Purged + embargoed K-fold** — remove training samples whose label
  horizon overlaps the test window, plus an embargo gap.
- **Combinatorial purged CV** — a distribution of backtest paths.
- **Deploy gate** — AURUM ships only if it beats both DLinear and XGBoost
  on purged-CV profit factor *and* walk-forward consistency. Same strict
  philosophy as the H4/H5/H6 rule gates.

---

## 6. Phased rollout

| Phase | Deliverable | Gate |
|---|---|---|
| 1 | Purged-CV harness + DLinear/XGBoost baselines | baseline numbers recorded |
| 2 | L0 SSL encoder + L1 backbone | beats DLinear on purged CV |
| 3 | L2 fusion + L3 multi-task heads | multi-task ≥ single-task direction |
| 4 | L4 meta-gate + L5 conformal; EA wiring | precision lift on held-out set |
| 5 | L6 quantile-Kelly sizing | risk-adjusted return ≥ fixed-lot |

Each phase is independently shippable and independently gated.

---

## 7. Explicitly rejected

| Candidate | Reason |
|---|---|
| Mamba / selective SSM backbone | selective-scan op has no clean ONNX export |
| TimesNet (FFT period blocks) | FFT reshaping fiddly to export; edge is long-horizon |
| Diffusion forecasters (CSDI/TimeGrad) | iterative sampling too slow on-tick |
| iTransformer | strength is many-symbol tokens — irrelevant single-symbol |
| TLOB / DeepLOB full LOB models | needs true order-book depth; MT5 retail feed is top-of-book only — deferred to mk6 |

RL position sizing (FinRL/PPO) is kept as a parallel research branch, not
on the deploy path: PPO policies are hard to validate and auditable
quantile-Kelly sizing is preferred.
