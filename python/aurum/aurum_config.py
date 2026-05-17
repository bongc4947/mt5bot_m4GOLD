"""
aurum_config.py — all hyperparameters for the AURUM v2 AI stack.

Kept separate from the legacy config.py so the v2 redesign can evolve
without touching the mk4 rule-strategy config. See docs/DESIGN_AURUM.md.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Symbol / artifacts
# ---------------------------------------------------------------------------
SYMBOL = "GOLD"
ARTIFACT_PREFIX = "M4GOLD_AURUM"          # M4GOLD_AURUM_GOLD.onnx, ..._spec.json

# ---------------------------------------------------------------------------
# Input contract — MUST stay in sync with ea/includes/AurumAgent.mqh
# ---------------------------------------------------------------------------
# Per-timeframe lookback window (in bars of that timeframe).
TIMEFRAMES: dict[str, int] = {"M5": 128, "M15": 64, "H1": 64}

# Microstructure channels computed per bar, per timeframe.
CHANNELS: list[str] = [
    "ret",          # log return close-to-close
    "hl_range",     # (high - low) / close
    "body",         # (close - open) / close
    "upper_wick",   # (high - max(open,close)) / close
    "lower_wick",   # (min(open,close) - low) / close
    "signed_vol",   # tick-rule signed volume, z-scored
    "vol_ratio",    # volume / rolling-mean volume
    "atr_norm",     # ATR(14) / close
]
N_CHANNELS = len(CHANNELS)

# Flattened deployed-model input dimension.
FLAT_INPUT_DIM = sum(L * N_CHANNELS for L in TIMEFRAMES.values())   # 2048

# Output layout (concatenated) — MUST match heads.py + AurumAgent.mqh.
N_DIRECTION_CLASSES = 3        # short / flat / long
QUANTILES = [0.1, 0.5, 0.9]
N_QUANTILES = len(QUANTILES)
N_EXEC = 3                     # sl_atr, tp_atr, timing
N_REGIME = 4                   # trend-up / trend-down / range / high-vol
OUTPUT_DIM = N_DIRECTION_CLASSES + N_QUANTILES + N_EXEC + N_REGIME   # 13

# Output slice offsets into the [1, 13] vector.
SLICE_DIR    = (0, 3)
SLICE_QUANT  = (3, 6)
SLICE_EXEC   = (6, 9)
SLICE_REGIME = (9, 13)

# ---------------------------------------------------------------------------
# Patch transformer backbone (L1)
# ---------------------------------------------------------------------------
PATCH_LEN = 16
PATCH_STRIDE = 8
D_MODEL = 96
DEPTH = 3
N_HEADS = 4
FFN_DIM = 192
DROPOUT = 0.25

# ---------------------------------------------------------------------------
# Self-supervised pretraining (L0)
# ---------------------------------------------------------------------------
SSL_MASK_RATIO = 0.40          # fraction of patches masked for reconstruction
SSL_EPOCHS = 50
SSL_LR = 3e-4
SSL_CONTRASTIVE_WEIGHT = 0.20  # weight of NT-Xent term added to recon MSE
SSL_TEMPERATURE = 0.20

# ---------------------------------------------------------------------------
# Fine-tuning (L1-L3)
# ---------------------------------------------------------------------------
FINETUNE_EPOCHS = 60
FINETUNE_LR = 2e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 20
FREEZE_ENCODER_EPOCHS = 8      # linear-probe warmup before unfreezing L0

# Multi-task loss weights.
LOSS_W_DIRECTION = 1.0
LOSS_W_QUANTILE = 0.5
LOSS_W_EXEC = 0.3
LOSS_W_REGIME = 0.3

# ---------------------------------------------------------------------------
# Labeling — triple barrier (reused from mk4 discipline)
# ---------------------------------------------------------------------------
LABEL_HORIZON_BARS = 20        # M5 bars forward
LABEL_TB_SL_ATR = 1.0
LABEL_TB_TP_ATR = 2.0
ATR_PERIOD = 14

# ---------------------------------------------------------------------------
# Purged cross-validation (cv/purged_kfold.py)
# ---------------------------------------------------------------------------
CV_N_SPLITS = 6
CV_EMBARGO_PCT = 0.01          # embargo gap as fraction of dataset length

# ---------------------------------------------------------------------------
# Meta-label gate (L4)
# ---------------------------------------------------------------------------
META_N_ESTIMATORS = 300
META_MAX_DEPTH = 4
META_LR = 0.05
META_ACT_THRESHOLD = 0.55      # EA fires only when P(act) >= this

# ---------------------------------------------------------------------------
# Conformal calibration (L5)
# ---------------------------------------------------------------------------
CONFORMAL_ALPHA = 0.10         # target miscoverage — 90% confidence sets
CONFORMAL_CAL_FRAC = 0.15      # fraction of train reserved for calibration

# ---------------------------------------------------------------------------
# Sizing (L6) — quantile-Kelly + vol targeting
# ---------------------------------------------------------------------------
SIZING_KELLY_FRACTION = 0.25   # fractional Kelly — never full Kelly
SIZING_VOL_TARGET = 0.10       # annualised vol target
SIZING_MAX_LOT_MULT = 2.0      # cap on the sizing multiplier
SIZING_MIN_LOT_MULT = 0.25

# ---------------------------------------------------------------------------
# Deploy gate — AURUM ships only if it clears these vs the baselines.
# ---------------------------------------------------------------------------
GATE_MIN_PF = 1.20
GATE_MIN_EXCESS_VS_BASELINE = 0.10   # purged-CV PF excess over best baseline
GATE_MIN_WF_CONSISTENCY = 0.50

VAL_SPLIT = 0.20
SEED = 42
