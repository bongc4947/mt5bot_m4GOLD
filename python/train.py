"""
train.py — MT5bot_m4Gold unified training CLI (GOLD-only).

Trains the directional AI head for GOLD, exports to ONNX, and stages the
artifacts in the MT5 Common Files directory for the EA to pick up.

The script auto-detects available hardware (CUDA / MPS / CPU) via
hardware_detector.py and picks a sensible batch size — no manual config
needed for the GPU-vs-CPU split.

USAGE
-----
    # Train the GOLD directional head (cached parquet, recompute features).
    python train.py gold --skip-extract

    # Train with a fresh MT5 extract.
    python train.py gold

    # Use MT5-exported features (no train/live drift).
    python train.py gold --skip-extract --mt5-features

    # Custom epochs and seed.
    python train.py gold --epochs 60 --seed 1337

The architecture used here is the GNN single-node variant (GNN_NODES=1)
since it is the original mk4 head for the METALS_SYMBOLS class; the model
collapses to a plain MLP when the symbol set is one. See models/__init__.py
for the active model surface.

For per-strategy trainers (H4 trend, H5 scalp, H6 mean-reversion) use the
dedicated scripts: train_h4_trend.py, train_h5_scalp_gold.py,
train_h6_mr_gold.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import EPOCHS, SYMBOL


def _load_factory(module_name: str, fn_name: str):
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


def cmd_gold(args: argparse.Namespace) -> int:
    # GOLD-only run. We reuse the mk4 _train_agent runner with a single
    # symbol; the multi-symbol scaffolding still works, it just trains
    # one model. The factory is the GNN single-node head — when
    # GNN_NODES == 1 it degenerates to a plain MLP, which is fine for a
    # single-symbol AI direction model.
    from _train_agent import run_agent

    # Hardware auto-detect happens inside the trainer; log it here too
    # so the user sees device choice up front.
    from hardware_detector import get as get_hw
    hw = get_hw()
    print(f"[m4Gold] hardware: {hw}")

    create_fn = _load_factory("models.exec_net", "create_exec_net")  # placeholder
    # The actual direction head used by _train_agent.run_agent is the one
    # passed in `create_dir_fn`. We re-purpose the GNN factory because it
    # is the single-node variant in METALS_SYMBOLS; here we fall back to a
    # lightweight directional MLP if the GNN module is unavailable.
    try:
        create_fn = _load_factory("models.gnn_metals", "create_gnn")
    except ModuleNotFoundError:
        # gnn_metals pruned in m4Gold; use a single-symbol direction MLP.
        from models.exec_net import create_exec_net as _ce
        create_fn = _ce

    results = run_agent(
        agent="GOLD",
        symbols=[SYMBOL],
        create_dir_fn=create_fn,
        epochs=args.epochs,
        skip_extract=args.skip_extract,
        seed=args.seed,
        mt5_features=getattr(args, "mt5_features", False),
        mt5_features_root=getattr(args, "mt5_features_root", None),
    )

    n_ok = sum(1 for r in results if r.get("onnx_ok"))
    if n_ok == 0 and results:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="MT5bot_m4Gold training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(__doc__ or "").split("USAGE")[-1],
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("gold", help=f"Train the GOLD AI direction head.")
    sp.add_argument("--epochs", type=int, default=EPOCHS)
    sp.add_argument("--skip-extract", action="store_true",
                    help="Reuse cached parquet bars, skip MT5 connect.")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--mt5-features", action="store_true",
                    help="Load features from MQL5-exported binary "
                         "instead of recomputing in Python.")
    sp.add_argument("--mt5-features-root", type=Path, default=None)
    sp.add_argument("--sampler", choices=["chronological", "random-window"],
                    default="chronological")
    sp.add_argument("--samples-per-epoch", type=int, default=100_000)
    sp.set_defaults(func=cmd_gold)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout, force=True,
    )
    args = build_parser().parse_args(argv)
    if getattr(args, "sampler", None):
        os.environ["HYDRA_SAMPLER"] = args.sampler
    if getattr(args, "samples_per_epoch", None):
        os.environ["HYDRA_SAMPLES_PER_EPOCH"] = str(args.samples_per_epoch)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
