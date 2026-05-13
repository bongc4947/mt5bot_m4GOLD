# Cloud notebook cell template

`cloud/notebook_run.py` (and its byte-identical Kaggle mirror
`cloud/kaggle/run.py`) auto-detects platform (Kaggle / Colab / RunPod /
local) and routes to `cloud/runner.sh`, which then dispatches to the
right Python trainer based on **TRAIN_MODE**.

**mk4.8 default: `TRAIN_MODE="strategies"` (H1+H2+H4 hypothesis combo).**
Other modes remain available but are not the production path (see
[STRATEGIES.md](../STRATEGIES.md) and [PRODUCTION_GAPS.md](../PRODUCTION_GAPS.md)).

## All editable knobs at a glance

| Constant | What | Default | When read |
|---|---|---|---|
| **REPO_URL** | git url (set token via `GH_PAT` secret if private) | `bongc4947/mtbotmk1` | always |
| **REPO_BRANCH** | branch to train | `master` | always |
| **TRAIN_MODE** | `strategies` / `directional` / `scalp` / `hedge` / `scalp_and_hedge` / `rule_meta` | `strategies` | always |
| **KAGGLE_DATASET_SLUG** | dataset slug to attach on Kaggle | `bongcruz/hydra4-tick-data-bundle` | Kaggle |
| **BUNDLE_DRIVE_PATH** | parquet bundle inside Drive | `/content/drive/MyDrive/HYDRA4/HYDRA4_data_bundle.zip` | Colab |
| **BUNDLE_LOCAL_PATH** | local file path | `""` | RunPod / vast.ai / local |
| **BUNDLE_HTTPS_URL** | HTTPS URL to a bundle zip | `""` | any platform |
| **SEED** | reproducibility | `42` | always |
| **HYDRA_BATCH_SIZE** | force a batch size; `""` = auto | `""` | always |
| **PARALLEL_TRAINING** | spawn sub-trainings concurrently inside one cell | `True` | always |
| **MAX_PARALLEL_WORKERS** | concurrent training cap (lower if you OOM) | `4` | always |
| TRAIN_GROUP | `all` / `forex` / `metals` / `indices` / `crypto` | `all` | directional |
| EPOCHS | direction-model epochs | `60` | directional |
| SYMBOLS | filter inside TRAIN_GROUP | `""` | directional |
| TRAIN_SAMPLER | `chronological` / `random-window` | `chronological` | directional |
| SAMPLES_PER_EPOCH | random-window only | `100_000` | directional |
| SCALP_SYMBOLS | space-separated; `""` = all defaults | `EURUSD GBPUSD USDJPY GOLD` | scalp |
| SCALP_EPOCHS | scalp model epochs | `30` | scalp |
| SCALP_WINDOW | tick-bars per scalp sample | `64` | scalp |
| SCALP_BATCH_SIZE | scalp loader batch | `1024` | scalp |
| SCALP_SHOULD_TRADE_THRESHOLD | gate sigmoid(should_trade) > this | `0.55` | scalp |
| HEDGE_PAIRS | `auto` (run screen) or `"A/B C/D"` | `auto` | hedge |
| HEDGE_EPOCHS | hedge model epochs | `30` | hedge |
| HEDGE_BATCH_SIZE | hedge loader batch | `512` | hedge |
| COINT_P_THRESHOLD | Engle-Granger ADF p-value | `0.05` | hedge / `auto` |
| COINT_MIN_WINDOWS | consecutive windows that must pass | `3` | hedge / `auto` |
| COINT_WINDOW_BARS | bars per cointegration window | `10_000` | hedge / `auto` |
| STRATEGIES_SELECTED | subset of `H1,H2,H4` | `H1,H2,H4` | strategies |
| STRATEGIES_SYMBOLS | space-separated; empty = every tick parquet | `""` | strategies |
| STRATEGIES_H1_TICKS_PER_BAR | tick-bar info-uniform size | `100` | strategies |
| STRATEGIES_H1_HORIZON | label horizon in tick-bars | `10` | strategies |
| STRATEGIES_H2_DONCHIAN | session breakout lookback (M5 bars) | `20` | strategies |
| STRATEGIES_H2_TIMEOUT | M5 bars before forced exit | `12` | strategies |
| STRATEGIES_H4_TIMEFRAME | `1h` / `4h` / `1d` | `1h` | strategies |
| STRATEGIES_H4_FAST / SLOW | MA-cross fast / slow periods | `50` / `200` | strategies |
| STRATEGIES_H4_MOM | momentum lookback | `240` | strategies |
| STRATEGIES_H4_NO_SHORT | long-only mode | `False` | strategies |
| STRATEGIES_ESTIMATORS | XGBoost trees (H1 / H2 meta) | `300` | strategies |
| STRATEGIES_MAX_DEPTH | XGBoost depth | `4` | strategies |
| STRATEGIES_AUDIT_FIRST | run `audit_strategies.py` before training | `True` | strategies |
| STRATEGIES_WORKERS | parallel symbols per strategy. **1 on Kaggle** (>1 nukes the kernel) | `1` | strategies |
| STRATEGIES_USE_GPU | XGBoost on `device="cuda"` — modest RAM relief | `False` | strategies |

## Quick recipes

### S) **mk4.8 default — H1+H2+H4 strategies combo** ⭐
```python
TRAIN_MODE              = "strategies"
STRATEGIES_SELECTED     = "H1,H2,H4"
STRATEGIES_AUDIT_FIRST  = True
# leave STRATEGIES_SYMBOLS="" to train on every tick parquet on disk
```
Runs the three hypothesis trainers in sequence, each clearing its own
skill gate. Writes per-(strategy, symbol) meta JSONs + (if deployable)
ONNX or spec JSONs to `onnx_out/`. Combined manifest at
`HYDRA4_STRATEGIES_summary.json`.

### S1) Strategies on a subset of symbols (fast smoke)
```python
TRAIN_MODE          = "strategies"
STRATEGIES_SYMBOLS  = "EURUSD SILVER UK_100"
STRATEGIES_SELECTED = "H4"   # cheapest — < 5 min total
```

### A) Full directional run on all asset classes (legacy default)
```python
TRAIN_MODE = "directional"
TRAIN_GROUP = "all"
EPOCHS = 60
```

### B) Scalp model on FX + Gold with random-feed sampling
```python
TRAIN_MODE = "scalp"
SCALP_SYMBOLS = "EURUSD GBPUSD USDJPY GOLD"
SCALP_EPOCHS = 30
```

### C) Hedge models — auto-screen and train every cointegrated pair
```python
TRAIN_MODE = "hedge"
HEDGE_PAIRS = "auto"
COINT_P_THRESHOLD = 0.05
```

### D) Hedge models — explicit pair list (skip the screen)
```python
TRAIN_MODE = "hedge"
HEDGE_PAIRS = "GOLD/SILVER BTCUSD/ETHUSD CrudeOIL/BRENT_OIL"
```

### E) Both scalp + hedge in one run (longest, most thorough)
```python
TRAIN_MODE = "scalp_and_hedge"
SCALP_SYMBOLS = "EURUSD GBPUSD GOLD BTCUSD"
HEDGE_PAIRS   = "auto"
```

### F) Per-group parallel directional (4 cells, classic mk4.7 use case)
- Cell 1: `TRAIN_MODE="directional"`  `TRAIN_GROUP="forex"`
- Cell 2: `TRAIN_MODE="directional"`  `TRAIN_GROUP="metals"`
- Cell 3: `TRAIN_MODE="directional"`  `TRAIN_GROUP="indices"`
- Cell 4: `TRAIN_MODE="directional"`  `TRAIN_GROUP="crypto"`

### G) **All-in-one in-cell parallel** (recommended for free-tier Kaggle)
The fastest single-cell run that actually uses Kaggle's quota efficiently:
```python
TRAIN_MODE        = "directional"
TRAIN_GROUP       = "all"
PARALLEL_TRAINING = True              # default; just confirming
MAX_PARALLEL_WORKERS = 4
```
This fires `prism` + `gnn` + `apex` + `ce` *simultaneously* in one cell.
On a T4, wall-clock drops from ~4× sequential to ~1.5-2× sequential
because the GPU is now actually saturated.

For scalp instead:
```python
TRAIN_MODE        = "scalp"
SCALP_SYMBOLS     = "EURUSD GBPUSD USDJPY GOLD BTCUSD ETHUSD"
PARALLEL_TRAINING = True
MAX_PARALLEL_WORKERS = 4              # 6 symbols, batched 4 at a time
```

### Why splitting into 4 cells *doesn't* save time

Jupyter / Kaggle / Colab run cells sequentially within one kernel —
cell 2 doesn't start until cell 1 finishes. Putting `TRAIN_GROUP="forex"`
in cell 1, `"metals"` in cell 2, etc. takes the same wall-clock as
running them serially in one cell. **`PARALLEL_TRAINING=True` is the
only way to actually run them at the same time** and consume your free
GPU-hour quota in 1× wall-clock instead of 4×.

The 4-cell layout is still useful for **4 separate notebook sessions**
(Kaggle allows multiple kernels concurrently with their own GPUs), but
within one kernel it's just longhand for sequential.

## The cell wrapper

This is the **paste-once template**. Override only the constants you
need; everything else uses the defaults from `cloud/notebook_run.py`.

### Private repo (default — uses `GH_PAT` secret)

```python
import urllib.request

# Get the PAT from the platform's secret store
try:
    from kaggle_secrets import UserSecretsClient
    _pat = UserSecretsClient().get_secret("GH_PAT")
except ImportError:
    from google.colab import userdata
    _pat = userdata.get("GH_PAT")

_url = "https://raw.githubusercontent.com/bongc4947/mtbotmk1/master/cloud/notebook_run.py"
_req = urllib.request.Request(_url, headers={"Authorization": f"token {_pat}"})
_src = urllib.request.urlopen(_req).read().decode()

# === Override any constants here ===========================================
# Example: switch to scalp mode on FX + Gold for 40 epochs
_src = _src.replace('TRAIN_MODE        = "directional"',
                    'TRAIN_MODE        = "scalp"', 1)
_src = _src.replace('SCALP_SYMBOLS     = "EURUSD GBPUSD USDJPY GOLD"',
                    'SCALP_SYMBOLS     = "EURUSD GBPUSD GOLD"', 1)
_src = _src.replace('SCALP_EPOCHS      = 30',
                    'SCALP_EPOCHS      = 40', 1)
# Point at your dataset (only used on Kaggle; Colab/local ignore this):
_src = _src.replace('KAGGLE_DATASET_SLUG = "bongcruz/hydra4-tick-data-bundle"',
                    'KAGGLE_DATASET_SLUG = "<your-user>/<your-dataset>"', 1)
# ===========================================================================

exec(compile(_src, "notebook_run.py", "exec"), {"__name__": "__main__"})
```

### Public repo (no auth)

Same body, replace the urllib block with a plain `urlopen(_url).read().decode()`.

## What the runner does per TRAIN_MODE

| TRAIN_MODE | Calls |
|---|---|
| `directional` | `python/train.py $TRAIN_AGENT --skip-extract --epochs $EPOCHS --sampler $TRAIN_SAMPLER --samples-per-epoch $SAMPLES_PER_EPOCH` |
| `scalp` | `python/train_scalp.py $SCALP_SYMBOLS --epochs $SCALP_EPOCHS --window $SCALP_WINDOW --batch-size $SCALP_BATCH_SIZE` |
| `hedge` (auto) | `python/train_hedge.py screen --p-threshold $COINT_P_THRESHOLD --min-windows $COINT_MIN_WINDOWS` then `train --epochs $HEDGE_EPOCHS` |
| `hedge` (explicit) | one `train_hedge.py train --pair A B` per pair in `HEDGE_PAIRS` |
| `scalp_and_hedge` | scalp then hedge |

ONNX outputs land in `onnx_out/` regardless. Compliance only trains
when `TRAIN_MODE=directional` and `TRAIN_AGENT=all`.
