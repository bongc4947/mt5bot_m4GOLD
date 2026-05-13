# MT5bot_m4Gold

A GOLD-only fork of `MT5_bot_mk4`, refocused on **AI-powered trading via ONNX**.
Single instrument, four hypotheses, one MQL5 dispatcher EA.

## What's in scope

| Layer | What |
|---|---|
| **AI head** | Direction model trained in Python (PyTorch + XGBoost), exported to ONNX, loaded by the EA via `OnnxCreate` — inference runs *inside* MT5, no Python sidecar. |
| **Strategies** | H1 (tick order-flow XGBoost), H4 (trend), H5 (M5 pullback scalp), H6 (GOLD-only mean reversion). The AI head acts as a per-entry **gate** for H4 / H5 / H6 — strategies fire only when the model agrees with the rule. |
| **Training** | Local (Windows / Linux / WSL, CPU or NVIDIA GPU) **or** cloud (Kaggle / Colab / RunPod / Modal). Hardware is auto-detected — `hardware_detector.py` picks batch size, AMP, and worker count from the device it finds. |
| **EA** | `ea/MT5bot_m4Gold_Dispatcher.mq5` — one EA on a GOLD chart, no second leg, no silver dependency. |

## What's out of scope

- Multi-symbol trading (forex, indices, crypto). All multi-symbol scaffolding (PRISM / APEX / CE_NET / GNN_METALS direction heads, cross-asset features) was stripped.
- GOLD/SILVER cointegration hedge (mk4 H6). Replaced with `train_h6_mr_gold.py` — single-symbol z-score mean reversion on GOLD H1.
- Python-side live inference. All runtime inference happens inside MetaTrader 5 via ONNX.

## Quickstart — local training

```powershell
# 1. Install deps (one-time)
python -m pip install -r python\requirements-train.txt

# 2. Train everything. Auto-detects GPU/CPU.
.\train.bat

# Or train one layer at a time:
.\train.bat strategies     # H1 + H4 + H5 + H6 rules + XGBoost head
.\train.bat ai             # PyTorch direction head -> ONNX
```

The runner prints the detected hardware tier on startup:

```
[m4Gold] device=cuda batch=32768 tier=enterprise  # 12 GB NVIDIA GPU
[m4Gold] device=cpu  batch=1024  tier=low          # 16 GB RAM, no GPU
```

Override with env vars:
```powershell
$env:FORCE_CPU=1            # ignore GPU, train on CPU
$env:EPOCHS=120             # longer AI head training
$env:STRATEGIES="H4,H6"     # subset
```

## Quickstart — cloud training

```bash
# Any Linux box with a recent Python + (optional) NVIDIA GPU:
git clone <your-fork-url> MT5bot_m4Gold
cd MT5bot_m4Gold
bash cloud/runner.sh
```

Or directly from a notebook (Kaggle/Colab/RunPod): see `cloud/README.md` and `cloud/notebook_run.py`.

The same `runner.sh` works on every cloud target — Kaggle's free-tier T4, RunPod 4090s, vast.ai A100s, Lambda H100s. Hardware detection picks the appropriate batch size every time.

## Data

By default, training reads parquet files from `data/`. If the directory is empty (e.g. a fresh clone), `python/config.py` falls back to the sibling `MT5_bot_mk4/data/` folder so a single shared 30 GB dataset serves both projects.

Override with the env var `M4GOLD_DATA_DIR`:

```powershell
$env:M4GOLD_DATA_DIR = "D:\market_data\gold_only"
```

Tick parquet expected at `<DATA_DIR>/ticks/HYDRA4_TICKS_GOLD.parquet`. Re-extract via `python python/extract_data.py GOLD` if missing.

## Deploying to MT5

```powershell
# 1. Train (writes specs + ONNX to onnx_out/ or MT5 Common Files)
.\train.bat

# 2. Copy artifacts to MT5 Common Files (skipped if auto-detected to that dir)
$dst = "$env:APPDATA\MetaQuotes\Terminal\Common\Files"
Copy-Item onnx_out\M4GOLD_*.json $dst
Copy-Item onnx_out\M4GOLD_*.onnx $dst

# 3. Compile and attach the EA
#    MetaEditor -> open ea\MT5bot_m4Gold_Dispatcher.mq5 -> F7
#    MT5 -> drag onto a GOLD chart
```

On attach the EA logs which strategies are live:

```
[m4Gold] AI gate ON  model=M4GOLD_GOLD_GOLD_dir_det.onnx  conf_min=0.55
[m4Gold] H4 ON  kind=ma_cross  tf=PERIOD_H1  fast=50  slow=200
[m4Gold] H5 spec M4GOLD_H5SCALP_GOLD_spec.json missing or deploy=false — H5 OFF
[m4Gold] H6 ON  z_in=2.00  z_out=0.50  z_stop=3.50  win=200
```

## Repository layout

```
MT5bot_m4Gold/
├── README.md                       # this file
├── train.bat                       # Windows one-click local trainer
├── ea/
│   ├── MT5bot_m4Gold_Dispatcher.mq5  # main EA — H4 + H5 + H6 + AI gate
│   ├── MT5bot_m4Gold_AI.mq5          # AI-only ONNX runner (HYDRA-style)
│   ├── MT5bot_m4Gold_FeatureExport.mq5  # exports features for parity testing
│   └── includes/                     # OnnxAgent, FeatureEncoder, TrendRule, etc.
├── python/
│   ├── config.py                   # single-symbol GOLD config
│   ├── train.py                    # AI direction head -> ONNX
│   ├── train_strategies.py         # orchestrator (H1 + H4 + H5 + H6)
│   ├── train_h4_trend.py           # H4 trend (rule)
│   ├── train_h5_scalp_gold.py      # H5 scalp (rule)
│   ├── train_h6_mr_gold.py         # H6 GOLD-only mean reversion (rule) — NEW
│   ├── train_h1_orderflow.py       # H1 XGBoost order-flow -> ONNX
│   ├── hardware_detector.py        # CUDA / MPS / CPU auto-probe
│   ├── exporter.py                 # PyTorch -> ONNX with onnxsim + ORT validation
│   ├── models/                     # exec_net, modify_net, scalp_net, hedge_net, xgb_head
│   └── ...
├── cloud/
│   ├── runner.sh                   # universal cloud entrypoint
│   ├── modal_app.py                # Modal job definition
│   ├── notebook_run.py             # Kaggle / Colab cell entrypoint
│   └── ...
└── onnx_out/                       # generated artifacts (gitignored)
```

## Honest expectations

This is a single-instrument bot trading a single, well-studied asset. Don't expect Renaissance Medallion. Expected behavior:

- **H4 + AI gate** on $1k demo: 20–80 trades/year, **+3 % to +12 %** annual return at 0.01 lots, multi-week flat periods between trend regimes.
- **All four strategies live**: ~3–5× the trade count. Net Sharpe should improve only if H5 and H6 are genuinely additive — the deploy gates are strict precisely so noise-stack additions can't pass.
- **Live drawdown > backtest MDD**. Overnight swap on long gold positions, broker spread widening during news, and regime change make live numbers worse than walk-forward.

Run demo for **30 days minimum** before committing real capital. Start at 25 % of the suggested live lot size for another 30 days after that.
