"""
config.py — MT5bot_m4Gold single source of truth.

Fork of HYDRA mk4 config, stripped to GOLD-only. The EA Defines.mqh must
stay in sync with the constants here.

GOLD-only AI stack:
  - PRISM, APEX, CE_NET, GNN_METALS pruned: not relevant to single-symbol GOLD.
  - Active models: exec_net, modify_net, scalp_net, hedge_net (gold-only MR),
    xgb_head (gradient-boosted direction head, ONNX-exported).
  - Inference runs entirely inside MT5 via ONNX — no Python sidecar.
"""

import os
from pathlib import Path


def _load_dotenv():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
HYDRA_VERSION = "m4Gold-1.0.0"
HYDRA_MAGIC = 20260513

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
PYTHON_DIR = BASE_DIR / "python"
MODELS_DIR = PYTHON_DIR / "models"
DATA_DIR = Path(os.environ.get("M4GOLD_DATA_DIR", BASE_DIR / "data"))
PARQUET_DIR = DATA_DIR / "parquet"
TICKS_DIR = DATA_DIR / "ticks"
LOGS_DIR = DATA_DIR / "logs"
ONNX_LOCAL_DIR = BASE_DIR / "onnx_out"

# Allow pointing back at the parent mk4 data folder when running locally so
# we don't duplicate 30 GB of parquets. Set M4GOLD_DATA_DIR to override.
_MK4_FALLBACK = BASE_DIR.parent / "MT5_bot_mk4" / "data"
if not (PARQUET_DIR.exists() and any(PARQUET_DIR.glob("*.parquet"))) and _MK4_FALLBACK.exists():
    DATA_DIR = _MK4_FALLBACK
    PARQUET_DIR = DATA_DIR / "parquet"
    TICKS_DIR = DATA_DIR / "ticks"
    LOGS_DIR = DATA_DIR / "logs"


# ---------------------------------------------------------------------------
# MT5 Common Files — Windows auto-detect, env override, local fallback
# ---------------------------------------------------------------------------
def _detect_mt5_common_dir() -> Path | None:
    env = os.environ.get("MT5_COMMON_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            p = Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files"
            if p.exists():
                return p
    return None


MT5_COMMON_DIR = _detect_mt5_common_dir() or ONNX_LOCAL_DIR
ONNX_OUTPUT_DIR = Path(os.environ.get("ONNX_OUTPUT_DIR", MT5_COMMON_DIR))

# ---------------------------------------------------------------------------
# Single-symbol roster — the whole point of this fork.
# ---------------------------------------------------------------------------
SYMBOL = "GOLD"
ALL_SYMBOLS = [SYMBOL]
METALS_SYMBOLS = [SYMBOL]
FOREX_SYMBOLS: list[str] = []
INDICES_SYMBOLS: list[str] = []
CE_SYMBOLS: list[str] = []

CROSS_ASSET_GOLD = SYMBOL
CROSS_ASSET_BTC = SYMBOL  # legacy compat — no BTC features used

AGENT_FOREX = 0
AGENT_METALS = 0   # single agent slot since we only have one asset class
AGENT_INDICES = 0
AGENT_CE = 0
AGENT_SYMBOL_MAP = {"GNN": [SYMBOL]}

# ---------------------------------------------------------------------------
# Feature dimensions — MUST match EA Defines.mqh
# ---------------------------------------------------------------------------
RAW_FEATURES = 50
M5_DIM = 200  # 50 raw × 4 transforms (raw, mean20, std20, delta20)
FEATURE_DIM_DIR = M5_DIM

EXEC_CTX_DIM = 40
FEATURE_DIM_EXEC = FEATURE_DIM_DIR + EXEC_CTX_DIM

MOD_POS_CTX_DIM = 8
FEATURE_DIM_MOD = FEATURE_DIM_DIR + MOD_POS_CTX_DIM

FEAT_BLOCK_M5_START = 0
M5_WINDOWS = [1, 20]
M5_TRANSFORMS = ["raw", "mean20", "std20", "delta20"]

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
PRISM_H0 = 256
PRISM_H1 = 128
PRISM_H2 = 64
PRISM_H3 = 32
APEX_H0 = 384
APEX_H1 = 192
APEX_H2 = 96
APEX_H3 = 48
GNN_H0 = 128
GNN_HIDDEN = 32
GNN_NODES = 1  # GOLD only
CE_H1 = 96
CE_H2 = 48
EXEC_H1 = 192
EXEC_H2 = 96
MOD_H1 = 96
MOD_H2 = 48
DROPOUT = 0.25

# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
LABEL_FORWARD_BARS = 20
LABEL_SHARPE_MIN = 0.15
LABEL_ATR_THRESH = 0.25
LABEL_DD_PENALIZE = 2.0
LABEL_ATR_PERIOD = 20
LABEL_EARLY_BARS = 3
LABEL_EARLY_ATR_MIN = 0.15

EXEC_TIMING_BARS = 3
EXEC_TIMING_THRESH = 0.3
EXEC_SL_SAFETY = 1.2
EXEC_SL_MIN_ATR = 0.5
EXEC_SL_MAX_ATR = 4.0
EXEC_TP_CONSERVATIVE = 0.8
EXEC_TP_MIN_ATR = 1.0
EXEC_TP_MAX_ATR = 6.0
EXEC_TP_RR_FLOOR = 1.5
EXEC_VOL_CLAMP_LO = 0.5
EXEC_VOL_CLAMP_HI = 2.0
EXEC_SL_MAX_ATR_BY_CLASS = {"metals": 5.0}
EXEC_SESSION_SPREAD_MAX = 1.5
EXEC_ROLLOVER_MIN = 30

MOD_BE_MFE_RATIO = 1.0
MOD_CLOSE_CONF = 0.6
MOD_CLOSE_MAE_FRAC = 0.7

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = 100
RETRAIN_EPOCHS = 25
PATIENCE = 35
LR = 3e-4
MIN_BAR_DATE = "2010-01-01"

LABEL_TB_SL_ATR = 1.0
LABEL_TB_TP_ATR = 2.0
LABEL_TB_USE = True
CLASS_BALANCE_RATIO = 1.5
FEATURE_WARMUP_BARS = 200
LR_WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-5
FOCAL_ALPHA = 0.50
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.2
MIN_SAMPLES_MIXUP = 10_000
VAL_SPLIT = 0.20
ONNX_OPSET = 12
MC_T = 20

BATCH_SIZE = 1024
MAX_BARS = 1_000_000_000
WORKERS = 2

# ---------------------------------------------------------------------------
# MC Dropout / confidence
# ---------------------------------------------------------------------------
MC_UNCERTAINTY_CAP = 0.15
CONF_THRESHOLD = 0.55
SESSION_THRESHOLD = 0.00
TIMING_THRESHOLD = 0.60
TIMING_WAIT_MIN = 0.30
LIMIT_EXPIRY_BARS = 3
LIMIT_OFFSET_ATR = 0.3

# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
DAILY_DD_PAUSE = 0.05
DAILY_DD_SHUTDOWN = 0.10
MAX_RISK_PER_TRADE = 0.01


# ---------------------------------------------------------------------------
# ONNX naming helpers — agent dimension kept for compatibility with mk4
# tooling, but always resolves to GOLD.
# ---------------------------------------------------------------------------
def onnx_det_path(agent: str = "GOLD", symbol: str = SYMBOL) -> Path:
    return ONNX_OUTPUT_DIR / f"M4GOLD_{agent}_{symbol}_dir_det.onnx"


def onnx_mc_path(agent: str = "GOLD", symbol: str = SYMBOL) -> Path:
    return ONNX_OUTPUT_DIR / f"M4GOLD_{agent}_{symbol}_dir_mc.onnx"


def onnx_exec_path(agent: str = "GOLD", symbol: str = SYMBOL) -> Path:
    return ONNX_OUTPUT_DIR / f"M4GOLD_{agent}_{symbol}_exec_det.onnx"


def onnx_modify_path(agent: str = "GOLD", symbol: str = SYMBOL) -> Path:
    return ONNX_OUTPUT_DIR / f"M4GOLD_{agent}_{symbol}_modify_det.onnx"


def meta_path(agent: str = "GOLD", symbol: str = SYMBOL) -> Path:
    return ONNX_OUTPUT_DIR / f"M4GOLD_{agent}_{symbol}_meta.json"


def parquet_path(symbol: str = SYMBOL, n_bars: int = 0) -> Path:
    if n_bars:
        return PARQUET_DIR / f"HYDRA4_FEAT_{symbol}_{n_bars}bars.parquet"
    return PARQUET_DIR / f"HYDRA4_FEAT_{symbol}.parquet"


def ticks_parquet_path(symbol: str = SYMBOL) -> Path:
    return TICKS_DIR / f"HYDRA4_TICKS_{symbol}.parquet"


def tickbars_parquet_path(symbol: str = SYMBOL, ticks_per_bar: int = 100) -> Path:
    return PARQUET_DIR / f"HYDRA4_TBARS_{symbol}_{ticks_per_bar}tpb.parquet"


def signal_log_path() -> Path:
    return MT5_COMMON_DIR / "M4GOLD_signals.csv"


def monitor_json_path() -> Path:
    return MT5_COMMON_DIR / "M4GOLD_monitor.json"


def retrain_flag_path(symbol: str = SYMBOL) -> Path:
    return MT5_COMMON_DIR / f"M4GOLD_retrain_{symbol}.flag"


def progress_json_path() -> Path:
    return BASE_DIR / "progress.json"


def closed_trades_log_path() -> Path:
    return MT5_COMMON_DIR / "M4GOLD_closed_trades.csv"


def mod_events_log_path() -> Path:
    return LOGS_DIR / "modify_events.csv"


MONITOR_INTERVAL_SEC = 60
WIN_RATE_WINDOW = 50
WIN_RATE_DROP_THRESH = 0.08
MODEL_AGE_RETRAIN_HRS = 4.0

for _d in [DATA_DIR, PARQUET_DIR, TICKS_DIR, LOGS_DIR, ONNX_LOCAL_DIR]:
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        pass
