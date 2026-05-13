"""
generate_deploy_playbook.py — read onnx_out/ and emit a deploy-ready Markdown.

Walks the per-strategy meta / spec JSONs, picks every cell with `deploy: True`,
and outputs:
  - A summary table of all deployable (strategy, symbol) cells
  - Per-cell EA configuration (the exact inputs to set on the chart)
  - Caveat sections (anti-signal flags, walk-forward consistency, MDD warnings)

Usage:
    python python/generate_deploy_playbook.py
    python python/generate_deploy_playbook.py --out DEPLOY_PLAYBOOK.md
    python python/generate_deploy_playbook.py --include-unstable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _load_meta_files():
    """Return dict {strategy: [meta dict, ...]} from onnx_out/."""
    from config import ONNX_OUTPUT_DIR
    by_strategy: dict[str, list[dict]] = {"H1": [], "H4": [], "H5": [], "H6": []}
    if not ONNX_OUTPUT_DIR.exists():
        return by_strategy
    for f in sorted(ONNX_OUTPUT_DIR.glob("M4GOLD_H1OF_*_meta.json")):
        try:    by_strategy["H1"].append(json.loads(f.read_text()))
        except Exception: pass
    for f in sorted(ONNX_OUTPUT_DIR.glob("M4GOLD_H4TREND_*_spec.json")):
        try:    by_strategy["H4"].append(json.loads(f.read_text()))
        except Exception: pass
    for f in sorted(ONNX_OUTPUT_DIR.glob("M4GOLD_H5SCALP_*_spec.json")):
        try:    by_strategy["H5"].append(json.loads(f.read_text()))
        except Exception: pass
    for f in sorted(ONNX_OUTPUT_DIR.glob("M4GOLD_H6MR_*_spec.json")):
        try:    by_strategy["H6"].append(json.loads(f.read_text()))
        except Exception: pass
    return by_strategy


def _h1_block(metas: list[dict], include_unstable: bool) -> str:
    out = ["### H1 — Tick-level order-flow imbalance\n"]
    deploys = [m for m in metas if m.get("deploy")]
    if not deploys:
        out.append("*No H1 cells cleared the skill gate in this run.*\n")
        # Note any anti-signal cells where the inversion was the winner
        inverted_top = [m for m in metas
                        if m.get("best_direction") == "inverted"]
        if inverted_top:
            out.append("\n**Anti-signal detected** in these symbols (model "
                       "is systematically wrong; the inverted-direction "
                       "evaluation was the best, just not deployable):\n\n")
            for m in inverted_top:
                out.append(f"- `{m['symbol']}` — inverted PF "
                           f"{m['best_pf']:.3f} N={m['best_n_trades']}\n")
        return "".join(out) + "\n"
    out.append("| Symbol | Direction | Threshold | PF | WR | N | Excess |\n")
    out.append("|---|---|---|---|---|---|---|\n")
    for m in deploys:
        out.append(
            f"| `{m['symbol']}` | {m.get('best_direction','normal')} | "
            f"{m['best_threshold']:.2f} | {m['best_pf']:.3f} | "
            f"{m['best_wr']:.3f} | {m['best_n_trades']} | "
            f"{m['excess_vs_passive']:+.3f} |\n")
    out.append("\n**EA config (per symbol):**\n```\n")
    for m in deploys:
        d = m.get("best_direction", "normal")
        out.append(f"# {m['symbol']}\n")
        out.append(f"InpEnabledStrategies = H1\n")
        out.append(f"InpH1Threshold       = {m['best_threshold']:.2f}\n")
        out.append(f"InpH1TicksPerBar     = {m['ticks_per_bar']}\n")
        if d == "inverted":
            out.append("# NOTE: anti-signal — invert the model output sign\n")
            out.append("InpH1InvertDirection = true\n")
        out.append("\n")
    out.append("```\n\n")
    return "".join(out)


def _h2_block(metas: list[dict], include_unstable: bool) -> str:
    # legacy stub kept for compatibility — H2 was dropped in m4Gold.
    return ""


def _h2_block_unused(metas: list[dict], include_unstable: bool) -> str:
    out = ["### H2 — Session-open Donchian breakout\n"]
    deploys = [m for m in metas if m.get("deploy")]
    if not deploys:
        out.append("*No H2 cells cleared the skill gate.*\n\n"
                   "The 2026-05-11 sweep showed every symbol's rule_PF "
                   "under 1.0 — the breakout rule itself has no edge on "
                   "this broker's quote stream at these session windows. "
                   "Consider re-examining session timing (LSE / NYSE / "
                   "Tokyo cash hours by symbol class) or trying a "
                   "tighter Donchian lookback.\n\n")
        return "".join(out)
    out.append("| Symbol | mode | rule PF | meta PF | excess |\n")
    out.append("|---|---|---|---|---|\n")
    for m in deploys:
        out.append(
            f"| `{m['symbol']}` | {m.get('deploy_mode','?')} | "
            f"{m['rule_pf']:.3f} | {m['meta_pf']:.3f} | "
            f"{max(m['rule_excess'], m['meta_excess_vs_rule']):+.3f} |\n")
    out.append("\n")
    return "".join(out)


def _compute_capital_allocation(deploys: list[dict]) -> list[dict]:
    """
    Risk-weight every deployable cell by:
        score   = max(0, mean_wf_sharpe) × wf_consistency
        weight  = score / sum(scores), capped at 50% per cell, floored at
                  5% per cell (so we never effectively benchmark out a
                  passing cell with a rounding error)
        renormalised so weights sum to 1.0

    Output for each cell:
        - risk_weight      — fraction of total risk budget
        - lot_at_1pct_risk — recommended lot per trade if you allow 1%
                              account drawdown per trade at the cell's
                              val_MDD. Bigger MDD -> smaller lot.

    The lot calculation is:
        lot_at_1pct_risk = (risk_weight × 0.01) / val_MDD
    which says: at our risk_weight share of total risk, what lot size
    keeps the expected drawdown at 1% of account. Treat this as the
    upper bound; live deploy should start at 25% of this.
    """
    scored = []
    for m in deploys:
        # H4 cells: walk_forward dict + summary[best_rule]
        wf = m.get("walk_forward", {})
        consistency = float(wf.get("consistency") or 0.0)
        mean_sharpe = float(wf.get("mean_sharpe") or 0.0)
        if "summary" in m and m.get("best_rule"):
            val = m["summary"][m["best_rule"]]["val"]
            val_mdd = max(0.001, float(val.get("mdd", 0.10)))
        else:
            # H1/H2 path
            val_mdd = max(0.001, float(m.get("mdd_val", 0.10)))
        score = max(0.0, mean_sharpe) * max(0.0, consistency)
        scored.append({"symbol": m.get("symbol", "?"),
                        "strategy": m.get("strategy", "?"),
                        "score": score,
                        "consistency": consistency,
                        "mean_sharpe": mean_sharpe,
                        "val_mdd": val_mdd,
                        "best_rule": m.get("best_rule"),
                        "timeframe": m.get("timeframe", "1h")})
    if not scored:
        return []
    total = sum(s["score"] for s in scored)
    if total <= 0:
        # Degenerate — all scores zero. Distribute equally.
        for s in scored:
            s["risk_weight"] = 1.0 / len(scored)
    else:
        # Initial proportional weighting
        for s in scored:
            s["risk_weight"] = s["score"] / total
        # Cap at 50% per cell, redistribute excess
        excess = 0.0
        for s in scored:
            if s["risk_weight"] > 0.50:
                excess += s["risk_weight"] - 0.50
                s["risk_weight"] = 0.50
        if excess > 0:
            uncapped = [s for s in scored if s["risk_weight"] < 0.50]
            if uncapped:
                share = excess / len(uncapped)
                for s in uncapped:
                    s["risk_weight"] += share
        # Floor at 5% so a passing cell isn't effectively benched
        for s in scored:
            if s["risk_weight"] < 0.05:
                s["risk_weight"] = 0.05
        total_w = sum(s["risk_weight"] for s in scored)
        for s in scored:
            s["risk_weight"] /= total_w
    # Lot calculation: 1% account risk per trade, scaled by share
    for s in scored:
        s["lot_at_1pct_risk"] = (s["risk_weight"] * 0.01) / s["val_mdd"]
    return scored


def _capital_block(by_strategy: dict[str, list[dict]]) -> str:
    """Render the capital-allocation table for all deployable cells."""
    all_deploys = []
    for sk, rows in by_strategy.items():
        all_deploys.extend(r for r in rows if r.get("deploy"))
    if not all_deploys:
        return ""
    alloc = _compute_capital_allocation(all_deploys)
    out = ["## Recommended capital allocation\n\n",
           "Each row is one deployable cell. The risk-weight column "
           "balances three signals: walk-forward consistency, mean "
           "Sharpe across sub-windows, and val MDD. Cells with worse "
           "MDD get a smaller lot to keep expected drawdown ~1% per "
           "trade at the suggested allocation.\n\n",
           "**Treat the lot column as an upper bound.** Start live "
           "trading at **25% of the suggested lot** for the first 30 "
           "demo+paper days, then scale up only if live PF holds up "
           "against the spec.\n\n",
           "| Symbol | Strategy | Rule | risk weight | mean WF Sharpe | "
           "WF consistency | val MDD | suggested lot @ 1% risk |\n",
           "|---|---|---|---|---|---|---|---|\n"]
    for s in sorted(alloc, key=lambda x: -x["risk_weight"]):
        out.append(
            f"| `{s['symbol']}` | {s['strategy']} | "
            f"`{s['best_rule'] or '?'}` | "
            f"**{s['risk_weight']:.1%}** | "
            f"{s['mean_sharpe']:+.2f} | "
            f"{s['consistency']:.0%} | "
            f"{100*s['val_mdd']:.1f}% | "
            f"**{s['lot_at_1pct_risk']:.3f}** |\n")
    out.append("\n*Lot column assumes 1% per-trade risk budget and is "
                "proportional to risk_weight. A 0.500 row means 0.50 "
                "lots at 1% risk; cut to 0.125 (25%) for the first 30 "
                "live demo days.*\n\n")
    return "".join(out)


def _h4_block(metas: list[dict], include_unstable: bool) -> str:
    out = ["### H4 — Long-horizon trend-following\n"]
    deploys = [m for m in metas if m.get("deploy")]
    unstable = [m for m in metas if m.get("deploy_unstable")]
    if not deploys and not unstable:
        out.append("*No H4 cells cleared the skill gate.*\n\n")
        return "".join(out)
    if deploys:
        out.append("**Robust deploys** (walk-forward consistency >= 50%):\n\n")
        out.append("| Symbol | Rule | TF | val Sharpe | val PF | val MDD | N | WF consistency |\n")
        out.append("|---|---|---|---|---|---|---|---|\n")
        for m in deploys:
            v = m["summary"][m["best_rule"]]["val"]
            wf = m.get("walk_forward", {})
            out.append(
                f"| `{m['symbol']}` | `{m['best_rule']}` | {m['timeframe']} | "
                f"{v['sharpe']:+.2f} | {v['pf']:.2f} | {100*v['mdd']:.1f}% | "
                f"{v['n_trades']} | "
                f"{wf.get('consistency', float('nan')):.0%} "
                f"({wf.get('sharpe_per_win', [])}) |\n")
        out.append("\n**EA config (per symbol):**\n```\n")
        for m in deploys:
            out.append(f"# {m['symbol']} (timeframe {m['timeframe']})\n")
            out.append(f"InpEnabledStrategies = H4\n")
            out.append(f"# spec auto-loaded from {m['symbol']}_spec.json\n")
            params = m["params"]
            if m["rule_kind"] == "ma_cross":
                out.append(f"# ma_cross fast={params['fast']} slow={params['slow']}\n")
            else:
                out.append(f"# momentum lookback={params['lookback']}\n")
            out.append("\n")
        out.append("```\n\n")
    if unstable and include_unstable:
        out.append("**Unstable cells** (passed single-split gate but walk-forward "
                   "shows the val edge is regime-specific — do NOT deploy without "
                   "additional out-of-sample evidence):\n\n")
        out.append("| Symbol | Rule | val Sharpe | WF consistency | min window Sharpe |\n")
        out.append("|---|---|---|---|---|\n")
        for m in unstable:
            v = m["summary"][m["best_rule"]]["val"]
            wf = m.get("walk_forward", {})
            out.append(
                f"| `{m['symbol']}` | `{m['best_rule']}` | {v['sharpe']:+.2f} | "
                f"{wf.get('consistency', float('nan')):.0%} | "
                f"{wf.get('min_sharpe', float('nan')):+.2f} |\n")
        out.append("\n")
    return "".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="DEPLOY_PLAYBOOK.md")
    p.add_argument("--include-unstable", action="store_true",
                   help="show unstable cells (passed single-split but failed "
                        "walk-forward) for visibility, with do-not-deploy flag")
    args = p.parse_args(argv)

    by_strategy = _load_meta_files()
    n_files = sum(len(v) for v in by_strategy.values())
    if n_files == 0:
        print("No M4GOLD_*_meta.json or _spec.json found in onnx_out/. "
              "Run train_strategies.py first.")
        return 1

    n_deploys = sum(1 for rows in by_strategy.values() for r in rows
                    if r.get("deploy"))
    body = [
        "# MT5bot_m4Gold — Deploy Playbook\n\n",
        f"Auto-generated from {n_files} meta files. ",
        f"**{n_deploys} cell{'s' if n_deploys != 1 else ''} deployable.**\n\n",
        "## Skill gate (the criteria each cell had to clear)\n\n",
        "- **H1**: model_PF ≥ 1.20, excess vs passive ≥ +0.10, N ≥ 30. Anti-signal "
        "(inverted direction) is evaluated in parallel; if the inversion wins, "
        "the EA must flip the model's output sign at runtime.\n",
        "- **H5/H6**: val PF ≥ 1.20, excess vs passive ≥ +0.10, N ≥ 30, "
        "walk-forward consistency ≥ 50%.\n",
        "- **H4**: val Sharpe ≥ 0.6, PF ≥ 1.10, MDD ≤ 30%, N ≥ 20, "
        "**plus walk-forward consistency ≥ 50%** (Sharpe>0 in majority of "
        "4 sub-windows). The WF gate was added after the 2026-05-11 sweep "
        "showed several cells passing single-split with train Sharpe ≈ 0 "
        "and val Sharpe > 1 — a regime-luck pattern that fails in live "
        "trading.\n\n",
        _h1_block(by_strategy["H1"], args.include_unstable),
        _h4_block(by_strategy["H4"], args.include_unstable),
        _capital_block(by_strategy),
        "## Operational checklist\n\n",
        "Before putting live capital on any deployable cell:\n\n",
        "1. **Demo for 30 days.** Compare live PF vs spec's val PF. Differences "
        "> 30% mean the broker's spread is materially worse than training "
        "assumed.\n",
        "2. **Paper-trade for 30 more days** with real broker quotes.\n",
        "3. **Start at 25% of the lot size** the EA suggests, scale up only "
        "after 3 months of consistent demo+paper performance.\n",
        "4. **Re-train monthly.** Trend regimes shift; a cell that deploys "
        "this month may fail next month's walk-forward check.\n",
    ]
    Path(args.out).write_text("".join(body), encoding="utf-8")
    print(f"Wrote {args.out}  ({n_files} cells inspected, {n_deploys} deployable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
