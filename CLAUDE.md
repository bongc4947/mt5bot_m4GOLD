# Claude project instructions — MT5bot_m4Gold

This project is a GOLD-only fork of the larger HYDRA mk4 system. Anything that does NOT directly support GOLD trading is out of scope. When in doubt: **ask before adding multi-symbol scaffolding.**

## Hard rules

- **Single symbol.** `config.SYMBOL` is `"GOLD"`. Do not introduce `EURUSD`, `SILVER`, `BTCUSD`, etc. unless the user explicitly asks for cross-asset features. The mk4 multi-symbol AGENT_SYMBOL_MAP is intentionally collapsed.
- **No second leg.** The H6 strategy is the GOLD-only z-score mean reversion in `train_h6_mr_gold.py`. The mk4 GOLD/SILVER cointegration hedge was deliberately removed.
- **AI is the entry gate, not a sidecar.** All ML runs inside MT5 as ONNX. There is no Python live-inference loop. If you find yourself adding `live_monitor.py`-style sidecar inference, stop and ask.
- **Hardware detection lives in one place.** `python/hardware_detector.py` is the only thing that decides device / batch / AMP. Don't put `torch.cuda.is_available()` checks anywhere else — they will drift.
- **ONNX naming.** All artifacts use the `M4GOLD_` prefix (not `HYDRA4_`). Helper functions in `config.py` (`onnx_det_path` et al.) produce the canonical names; use them rather than f-string literals.
- **Data is shared with mk4.** If `data/` is empty, `config.py` auto-falls-back to `../MT5_bot_mk4/data/`. Don't duplicate the 30 GB parquet set.

## Cadence preferences

- Local training is the dev loop. Cloud (`cloud/runner.sh`) is for full-history retrains.
- When proposing changes, distinguish strategy changes (require a fresh walk-forward run) from harness changes (don't).
- New strategies follow the existing pattern: trainer in `python/train_h?_*.py`, spec JSON dropped in `onnx_out/`, EA branch in `MT5bot_m4Gold_Dispatcher.mq5` gated on `deploy: true`.

## Things to avoid

- Adding `--symbol` flags that accept multiple values.
- Re-introducing the PRISM / APEX / CE_NET / GNN_METALS direction heads — they were pruned because multi-node graph nets degenerate to plain MLPs at `GNN_NODES=1` and the simpler `exec_net` / `xgb_head` heads dominate at single-symbol scale.
- Adding macro / fundamental / calendar features. This bot is **price + microstructure only** (matches the user's `feedback_no_macro_fundamentals` memory).
- Writing Python sidecars that run during live trading. The whole point of the ONNX path is to keep all inference inside MT5.
