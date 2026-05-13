"""
cloud/notebook_run.py — universal notebook training entrypoint.

Paste into a cell on **any** of:

  - Kaggle Notebooks       (auto-detected via /kaggle/working)
  - Google Colab           (auto-detected via google.colab module)
  - Lightning AI Studios   (auto-detected as generic cloud)
  - Paperspace / RunPod / vast.ai / Lambda  (generic cloud)
  - Local Jupyter / VS Code / IPython

The script self-adjusts: working dir, data lookup, Drive mount, and
output dir are all chosen per-platform. You only need to edit the
constants in the EDIT section that match your platform — the others
are ignored.

Flow:
  1. Detect platform (kaggle | colab | cloud | local).
  2. Resolve REPO_DIR + OUT_DIR.
  3. Clone (or update) the HYDRA mk4 repo.
  4. Resolve where the parquet bundle lives (per-platform fallback chain).
  5. Hand off to cloud/runner.sh, which does install + audit + train +
     copy ONNX outputs to OUT_DIR.

Per-group training (4 cells, one per asset class)
-------------------------------------------------
Set TRAIN_GROUP below to 'forex' / 'metals' / 'indices' / 'crypto' to
train only that asset class. Use 'all' for end-to-end (default).

To run all four groups in parallel, open four notebook sessions (Kaggle
allows multiple kernels concurrently; Colab Pro allows multi-instance)
and set a different TRAIN_GROUP in each. Each kernel gets its own GPU,
giving real 4-way parallelism.

To run them sequentially in a single kernel, paste this script into four
cells, change TRAIN_GROUP per cell — the clone, install, and bundle
restore are idempotent (cached after first cell), so cells 2-4 jump
straight to training.

See cloud/CELLS_TEMPLATE.md for the ready-to-paste 4-cell layout.

Tick-mode training (mk4.7)
--------------------------
Set TRAIN_SAMPLER='random-window' below if you've extracted tick-bar
parquets with `extract_data.py --source ticks`. The trainer will then
treat the dataset as an unbounded random feed (SAMPLES_PER_EPOCH draws
per epoch) instead of a fixed chronological pass — closer to how the
live EA experiences the market. Default 'chronological' matches the
legacy M5-bar workflow.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ===========================================================================
# EDIT — every knob the cloud runner reads is below. Only the constants for
# *your* platform / your TRAIN_MODE are honoured; the rest are ignored.
# ===========================================================================

# --- Repo source ----------------------------------------------------------
REPO_URL          = "https://github.com/bongc4947/mtbotmk1.git"
REPO_BRANCH       = "master"

# --- What this run should train -------------------------------------------
# mk4.8: NAS100 has been removed from every roster — the broker doesn't
# quote it and no tick parquet exists. ALL_SYMBOLS (15) now matches the
# tick parquet set on disk exactly. Set TRAIN_MODE="strategies" for the
# H1+H2+H4 pivot; "rule_meta" remains for the legacy path.
#
# "strategies"       = mk4.8 production path: H1+H2+H4 hypothesis combo
#                     (tick order-flow + session breakout + trend-following)
#                     trained on raw ticks. Replaces rule_meta after the
#                     look-ahead leak post-mortem (see PRODUCTION_GAPS.md).
# "directional"      = PRISM/GNN/APEX/CE direction models (legacy path)
# "scalp"            = Phase 2 scalp model on tick-bar parquets
# "hedge"            = Phase 3 hedge: cointegration screen + per-pair train
# "scalp_and_hedge"  = scalp + hedge in one run
# "rule_meta"        = mk4.7 path: meta-classifier on the z-score rule's
#                     candidates. Kept compilable for diagnostic comparison;
#                     NOT recommended (built on the leaking premise).
TRAIN_MODE        = "strategies"

# --- Where to fetch the parquet bundle ------------------------------------
# Pick whichever applies to your platform; the rest are ignored.
KAGGLE_DATASET_SLUG = "bongcruz/hydra4-tick-data-bundle"   # Kaggle: dataset slug
BUNDLE_DRIVE_PATH   = "/content/drive/MyDrive/HYDRA4/HYDRA4_data_bundle.zip"  # Colab
BUNDLE_LOCAL_PATH   = ""    # generic cloud / local: e.g. "/workspace/HYDRA4_data_bundle.zip"
BUNDLE_HTTPS_URL    = ""    # any platform: e.g. "https://your.host/HYDRA4_data_bundle.zip"

# --- Common training knobs (apply to every TRAIN_MODE) --------------------
SEED              = 42
HYDRA_BATCH_SIZE  = ""           # "" = auto-detect from VRAM; or e.g. "131072"
# PARALLEL_TRAINING=True spawns sub-trainings concurrently INSIDE this cell so
# Kaggle's free GPU-hour quota ticks against wall-clock instead of 4×
# wall-clock. Splitting work across 4 separate cells in one notebook still
# runs sequentially (Jupyter constraint) — this is the fix.
#   directional + TRAIN_GROUP=all  -> prism + gnn + apex + ce in parallel
#   scalp                          -> N symbols capped at MAX_PARALLEL_WORKERS
#   hedge                          -> N pairs capped at MAX_PARALLEL_WORKERS
PARALLEL_TRAINING = True
MAX_PARALLEL_WORKERS = 4         # reduce if you OOM (T4 = 16 GB VRAM, fits 4)

# --- TRAIN_MODE = "directional" (legacy direction models) ------------------
# Asset class:
#   "all"     -> PRISM + GNN + APEX + CE + compliance
#   "forex"   -> PRISM only   (EURUSD, GBPUSD, USDJPY)
#   "metals"  -> GNN only     (GOLD, SILVER, PLATINUM, COPPER)
#   "indices" -> APEX only    (US_500, UK_100)
#   "crypto"  -> CE only      (BTCUSD, ETHUSD, LTCUSD, CrudeOIL, BRENT_OIL, NATURAL_GAS)
TRAIN_GROUP       = "all"
EPOCHS            = 60
SYMBOLS           = ""           # "" = all default for the chosen TRAIN_GROUP

# Sampler. "chronological" = order traversal with shuffle=True (M5 workflow).
# "random-window" = with-replacement random draws — pair with tick-bar
# parquets (extract_data.py --source ticks) for a near-live random feed.
TRAIN_SAMPLER     = "chronological"
SAMPLES_PER_EPOCH = 100_000

# --- TRAIN_MODE = "scalp" (Phase 2) ----------------------------------------
# Symbols to train scalp models on. Space-separated; empty -> "EURUSD GBPUSD USDJPY GOLD".
SCALP_SYMBOLS     = "EURUSD GBPUSD USDJPY GOLD"
SCALP_EPOCHS      = 30
SCALP_WINDOW      = 64           # tick-bars per training sample
SCALP_BATCH_SIZE  = 1024
SCALP_SHOULD_TRADE_THRESHOLD = 0.55  # gate: sigmoid(should_trade) > this -> fire

# --- TRAIN_MODE = "rule_meta" (mk4.7 production path) ----------------------
# Meta-classifier on top of the z-score fade rule. Trains per-symbol on the
# rule's historical candidates; predicts P(this candidate hits TP). Filter
# at meta_prob >= threshold to lift the rule's PF.
# Symbols default = all LIVE-READY on disk from BACKTEST_RESULTS.md. Override
# with a space-separated list. cost_pip = retail spread assumption; drop
# to 0.3 if you've switched to an ECN broker.
META_SYMBOLS       = ""             # empty -> all LIVE-READY on disk
META_ESTIMATORS    = 300
META_MAX_DEPTH     = 4
META_COST_PIP      = 2.0            # 2.0 retail, 0.3 ECN; tune for your broker

# --- TRAIN_MODE = "strategies" (mk4.8 hypothesis pivot) --------------------
# After train_rule_meta surfaced that the z-score rule was riding a look-
# ahead leak (see PRODUCTION_GAPS.md), the project pivoted to three
# independently testable hypotheses, all reusing the existing infra:
#
#   H1 -- tick-level order-flow imbalance directional model (XGBoost)
#   H2 -- session-open Donchian breakout (rule + optional meta filter)
#   H4 -- long-horizon trend-following (deterministic, no ML required)
#
# Pick any subset via STRATEGIES_SELECTED. Each clears the skill gate
# independently before being written to onnx_out/; combined manifest at
# HYDRA4_STRATEGIES_summary.json lists what shipped.
STRATEGIES_SYMBOLS       = ""             # empty -> every tick parquet on disk
# mk4.8.4: H2 dropped from the default after 0/15 deploys across three full
# Kaggle sweeps (best rule_PF was 0.603 on GOLD, none cleared the 1.2 gate).
# The Donchian session-open breakout rule is dead on this broker's quote
# feed. Re-add H2 to the selected list if you want a control run.
STRATEGIES_SELECTED      = "H1,H4"        # subset of H1,H2,H4
STRATEGIES_H1_TICKS_PER_BAR = 100
STRATEGIES_H1_HORIZON    = 10
# mk4.8.4: skip H1 on tick files larger than this. ETHUSD (1.8 GB) and
# GOLD (952 MB) reproducibly hit SIGKILL OOM on Kaggle's 30 GB box.
# Bump to 99 on fat-RAM boxes (RunPod / vast.ai / Lambda) to include them.
STRATEGIES_H1_MAX_TICK_FILE_GB = 1.0
STRATEGIES_H2_DONCHIAN   = 20
STRATEGIES_H2_TIMEOUT    = 12
STRATEGIES_H4_TIMEFRAME  = "1h"            # 1h | 4h | 1d
STRATEGIES_H4_FAST       = 50
STRATEGIES_H4_SLOW       = 200
STRATEGIES_H4_MOM        = 240             # legacy single-lookback knob
# mk4.8.3: multiple momentum lookbacks. Each becomes its own rule
# (mom_120, mom_240, ...) — wider coverage catches symbols where a
# different trend horizon fits.
STRATEGIES_H4_MOM_LOOKBACKS = "120,240"
# mk4.8.6: vol-regime filter for H4. 0.0 = off (legacy). Try 0.85-0.95
# to skip trend trades during chop sub-windows (the GOLD fresh-data
# run produced a -2.25 Sharpe sub-window the filter would have masked).
STRATEGIES_H4_VOL_FILTER_RATIO = 0.0
STRATEGIES_H4_VOL_FILTER_SHORT = 20
STRATEGIES_H4_VOL_FILTER_LONG  = 500
STRATEGIES_H4_NO_SHORT   = False           # long-only when True
STRATEGIES_ESTIMATORS    = 300
STRATEGIES_MAX_DEPTH     = 4
STRATEGIES_AUDIT_FIRST   = True            # run audit_strategies before training
# mk4.8.1: STRATEGIES_WORKERS=1 is the new default after a 14:17 UTC Kaggle
# run with workers=4 was killed by Kaggle's RAM watchdog — when 4 workers
# all build big tick-bar arrays concurrently (BRENT_OIL/BTCUSD/COPPER/CrudeOIL),
# total RAM crosses the per-kernel limit and Kaggle nukes the whole pool
# (BrokenProcessPool across every future, all 4 die together).
#
# Sequential = safe but slow:  ~75 min total wall-clock on 15 symbols.
# 2 workers   = often fine, OOM risk on 2 big symbols concurrent.
# 3-4 workers = OOM on Kaggle ~50% of the time (the trace above).
#
# Set this to >1 only if your platform has >40 GB RAM (RunPod / vast.ai /
# Lambda boxes typically do; Kaggle's 30 GB free tier does NOT).
STRATEGIES_WORKERS       = 1
# Use GPU for XGBoost training (H1 directional + H2 meta filter). Modest
# win: model + DMatrix move to VRAM (~50-200 MB per training), freeing
# equivalent RAM. Does NOT fix the tick-bar memory bottleneck (that's
# numpy/pandas on RAM, no GPU path without a major dependency rewrite).
# Default off because XGBoost on 200K x 16 tabular rows often runs as
# fast on CPU as GPU due to PCIe transfer overhead.
STRATEGIES_USE_GPU       = False

# --- TRAIN_MODE = "hedge" (Phase 3) ----------------------------------------
# "auto" runs the cointegration screen first, then trains every passing pair.
# Or specify explicit pairs: "GOLD/SILVER BTCUSD/ETHUSD CrudeOIL/BRENT_OIL"
HEDGE_PAIRS         = "auto"
HEDGE_EPOCHS        = 30
HEDGE_BATCH_SIZE    = 512
COINT_P_THRESHOLD   = 0.05       # Engle-Granger ADF p-value threshold per window
COINT_MIN_WINDOWS   = 3          # require >= this many consecutive windows to pass
COINT_WINDOW_BARS   = 10_000

# --- GitHub PAT for private repos -----------------------------------------
# Provide a Personal Access Token any of these ways:
#   - Kaggle : Add-ons -> Secrets -> Add name=GH_PAT
#   - Colab  : left sidebar -> Secrets -> +Secret name=GH_PAT (notebook access ON)
#   - Env    : export GH_PAT=ghp_...        (also accepts GITHUB_TOKEN)
#   - URL    : REPO_URL = "https://USER:TOKEN@github.com/USER/REPO.git"
GH_PAT_SECRET_NAME = "GH_PAT"

# group -> agent flag understood by python/train.py and cloud/runner.sh
_GROUP_TO_AGENT = {
    "all":     "all",
    "forex":   "prism",
    "metals":  "gnn",
    "indices": "apex",
    "crypto":  "ce",
}
# ===========================================================================


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
def detect_platform() -> str:
    if Path("/kaggle/working").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except Exception:
        pass
    if os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    # Cloud-machine heuristics: a /workspace directory we can write to and
    # no Kaggle marker. Covers RunPod, vast.ai, Lambda, Paperspace.
    if Path("/workspace").is_dir() and os.access("/workspace", os.W_OK):
        return "cloud"
    return "local"


PLATFORM = detect_platform()
print(f"[notebook_run] platform: {PLATFORM}")

_VALID_MODES = ("directional", "scalp", "hedge", "scalp_and_hedge",
                "rule_meta", "strategies")
if TRAIN_MODE not in _VALID_MODES:
    raise SystemExit(f"TRAIN_MODE={TRAIN_MODE!r} invalid. Expected one of: {_VALID_MODES}")
if TRAIN_GROUP not in _GROUP_TO_AGENT:
    raise SystemExit(
        f"TRAIN_GROUP={TRAIN_GROUP!r} invalid. "
        f"Expected one of: {sorted(_GROUP_TO_AGENT)}"
    )
if TRAIN_SAMPLER not in ("chronological", "random-window"):
    raise SystemExit(
        f"TRAIN_SAMPLER={TRAIN_SAMPLER!r} invalid. "
        f"Expected 'chronological' or 'random-window'."
    )
TRAIN_AGENT = _GROUP_TO_AGENT[TRAIN_GROUP]
print(f"[notebook_run] TRAIN_MODE={TRAIN_MODE}")
if TRAIN_MODE in ("directional",):
    print(f"[notebook_run] TRAIN_GROUP={TRAIN_GROUP}  -> TRAIN_AGENT={TRAIN_AGENT}  "
          f"sampler={TRAIN_SAMPLER}  samples_per_epoch={SAMPLES_PER_EPOCH}")
if TRAIN_MODE in ("scalp", "scalp_and_hedge"):
    print(f"[notebook_run] SCALP symbols=[{SCALP_SYMBOLS}]  epochs={SCALP_EPOCHS}  "
          f"window={SCALP_WINDOW}  batch={SCALP_BATCH_SIZE}")
if TRAIN_MODE in ("hedge", "scalp_and_hedge"):
    print(f"[notebook_run] HEDGE pairs=[{HEDGE_PAIRS}]  epochs={HEDGE_EPOCHS}  "
          f"p<{COINT_P_THRESHOLD}  min_windows={COINT_MIN_WINDOWS}")
if TRAIN_MODE == "rule_meta":
    print(f"[notebook_run] RULE_META symbols=[{META_SYMBOLS or 'all LIVE-READY on disk'}]  "
          f"estimators={META_ESTIMATORS}  max_depth={META_MAX_DEPTH}  cost_pip={META_COST_PIP}")
if TRAIN_MODE == "strategies":
    print(f"[notebook_run] STRATEGIES combo=[{STRATEGIES_SELECTED}]  "
          f"symbols=[{STRATEGIES_SYMBOLS or 'all tick parquets'}]  "
          f"H4_tf={STRATEGIES_H4_TIMEFRAME}  audit_first={STRATEGIES_AUDIT_FIRST}")


# ---------------------------------------------------------------------------
# Platform-specific working layout
# ---------------------------------------------------------------------------
if PLATFORM == "kaggle":
    WORK = Path("/kaggle/working")
elif PLATFORM == "colab":
    WORK = Path("/content")
elif PLATFORM == "cloud":
    WORK = Path("/workspace")
else:  # local
    # Sit a child folder next to wherever the notebook was launched, but
    # don't pollute the user's repo if they happen to launch from inside one.
    WORK = Path.cwd() / "hydra_mk4_run"
    WORK.mkdir(parents=True, exist_ok=True)

REPO_DIR = WORK / "MT5_bot_mk4"
OUT_DIR  = WORK / "onnx_out"
print(f"[notebook_run] WORK={WORK}  REPO_DIR={REPO_DIR}  OUT_DIR={OUT_DIR}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _redact(text: str) -> str:
    """Strip GitHub PATs from any string before printing."""
    import re
    text = re.sub(r"(ghp_|github_pat_|gho_|ghu_|ghs_)[A-Za-z0-9_]{20,255}", "<TOKEN>", text)
    text = re.sub(r"://[^:@\s/]+:[^@\s/]+@github\.com", "://<TOKEN>@github.com", text)
    text = re.sub(r"(Authorization:\s*(?:Bearer|Basic))\s+\S+", r"\1 <TOKEN>", text)
    return text


def _run(cmd, **kwargs):
    """Run a subprocess and surface stdout/stderr on failure (redacting PATs)."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0:
        safe_cmd = " ".join(_redact(str(c)) for c in cmd)
        print(f"\n>>> command failed (exit {r.returncode}): {safe_cmd}")
        if r.stdout: print(f"--- stdout ---\n{_redact(r.stdout)}")
        if r.stderr: print(f"--- stderr ---\n{_redact(r.stderr)}")
        raise SystemExit(
            "\nCommon causes:\n"
            "  - Internet disabled (Kaggle: Settings → Internet → On).\n"
            "  - Private repo, missing GH_PAT. Add a Kaggle/Colab Secret\n"
            "    named GH_PAT, or set env GH_PAT/GITHUB_TOKEN, or bake the\n"
            "    token into REPO_URL: https://USER:TOKEN@github.com/...\n"
            "  - Branch typo (master vs main).\n"
        )
    return r


def _get_github_pat() -> str | None:
    """Resolve a GitHub PAT from env / Kaggle Secrets / Colab Secrets."""
    pat = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
    if pat:
        print(f"[notebook_run] GitHub PAT: env (len={len(pat)})")
        return pat

    if PLATFORM == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient
            pat = UserSecretsClient().get_secret(GH_PAT_SECRET_NAME)
            if pat:
                print(f"[notebook_run] GitHub PAT: Kaggle Secret '{GH_PAT_SECRET_NAME}' (len={len(pat)})")
                return pat
        except Exception as e:
            print(f"[notebook_run] kaggle_secrets lookup failed: {e}")
            print(f"  Add the secret: Add-ons → Secrets → Add → name='{GH_PAT_SECRET_NAME}'")

    if PLATFORM == "colab":
        try:
            from google.colab import userdata
            pat = userdata.get(GH_PAT_SECRET_NAME)
            if pat:
                print(f"[notebook_run] GitHub PAT: Colab Secret '{GH_PAT_SECRET_NAME}' (len={len(pat)})")
                return pat
        except Exception as e:
            print(f"[notebook_run] colab userdata lookup failed: {e}")
            print(f"  Add the secret: left sidebar → Secrets → +Secret → name='{GH_PAT_SECRET_NAME}'")

    return None


def _authed_url(url: str, pat: str | None) -> str:
    """Embed a PAT into an https://github.com/... URL.

    Git-over-HTTPS uses HTTP Basic auth, not Bearer — so the documented
    GitHub approach is to put the token in the URL where curl/git's
    transport layer turns it into a Basic auth header automatically.

    We strip the PAT from .git/config immediately after each operation so
    persistent volumes don't end up with the token on disk.
    """
    if not pat or not url.startswith("https://github.com/"):
        return url
    if "@github.com/" in url:
        return url   # caller already embedded one
    # 'x-access-token' is GitHub's documented placeholder username for PATs
    # (any non-empty string works; matches the GitHub Actions checkout pattern).
    return url.replace("https://", f"https://x-access-token:{pat}@", 1)


def _show(p: Path, max_entries: int = 30) -> None:
    files = [c for c in p.rglob("*") if c.is_file()]
    for c in sorted(files)[:max_entries]:
        print(f"  {c.stat().st_size/1e6:>8.1f} MB  {c.relative_to(p)}")
    if len(files) > max_entries:
        print(f"  ... (+{len(files) - max_entries} more)")
    elif not files:
        print("  <empty>")


# ---------------------------------------------------------------------------
# 1 · Clone / update the repo (we need cloud/runner.sh from it)
# ---------------------------------------------------------------------------
GH_PAT = _get_github_pat()
AUTH_URL = _authed_url(REPO_URL, GH_PAT)

# Tell git not to prompt — without a PAT, fail fast instead of hanging on
# stdin (which on a notebook means "fatal: could not read Username").
_git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

if REPO_DIR.exists() and not (REPO_DIR / ".git").exists():
    shutil.rmtree(REPO_DIR)

if not REPO_DIR.exists():
    print(f"[notebook_run] clone: {REPO_URL} @ {REPO_BRANCH}")
    _run(["git", "clone", "--branch", REPO_BRANCH, "--single-branch",
          "--depth", "1", AUTH_URL, str(REPO_DIR)], env=_git_env)
    if GH_PAT:
        # Strip PAT from .git/config so persistent volumes don't leak it.
        _run(["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", REPO_URL],
             env=_git_env)
    # Disable file-mode tracking so subsequent `chmod +x` calls don't dirty
    # the working tree (was causing pull to fail on the next session).
    _run(["git", "-C", str(REPO_DIR), "config", "core.fileMode", "false"],
         env=_git_env)
else:
    # Cloud sessions are ephemeral — there's nothing in the working tree
    # worth preserving between runs. fetch + hard-reset is the bulletproof
    # alternative to `pull --ff-only`, which fails on the slightest dirt
    # (chmod bits, leftover output files, line-ending tweaks).
    print(f"[notebook_run] hard-reset to origin/{REPO_BRANCH}")
    if GH_PAT:
        _run(["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", AUTH_URL],
             env=_git_env)
    try:
        _run(["git", "-C", str(REPO_DIR), "config", "core.fileMode", "false"],
             env=_git_env)
        _run(["git", "-C", str(REPO_DIR), "fetch", "--quiet", "origin", REPO_BRANCH],
             env=_git_env)
        _run(["git", "-C", str(REPO_DIR), "checkout", "--quiet", REPO_BRANCH],
             env=_git_env)
        _run(["git", "-C", str(REPO_DIR), "reset", "--hard",
              f"origin/{REPO_BRANCH}", "--quiet"], env=_git_env)
    finally:
        if GH_PAT:
            _run(["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", REPO_URL],
                 env=_git_env)

# Print the repo state stamp so the user sees which commit is running.
# This shows up in the Kaggle / Colab cell output, no extra setup needed.
try:
    subprocess.run(
        ["python", str(REPO_DIR / "scripts" / "version.py")],
        check=False,
    )
except Exception:
    pass


# ---------------------------------------------------------------------------
# 2 · Resolve BUNDLE_URL per platform (with fallbacks)
# ---------------------------------------------------------------------------
def _resolve_bundle_url() -> str:
    # An explicit HTTPS URL is honoured on every platform.
    if BUNDLE_HTTPS_URL:
        return BUNDLE_HTTPS_URL

    if PLATFORM == "kaggle":
        kaggle_input = Path("/kaggle/input")
        if not kaggle_input.exists() or not any(kaggle_input.iterdir()):
            raise SystemExit(
                "Nothing mounted at /kaggle/input/. Attach the dataset via the "
                f"right sidebar: Add Data → search {KAGGLE_DATASET_SLUG!r} → +."
            )
        print("[notebook_run] /kaggle/input/ contents:")
        for c in sorted(kaggle_input.iterdir()):
            print(f"  {c.name}")

        zips      = list(kaggle_input.rglob("HYDRA4_data_bundle*.zip"))
        m5_pqs    = list(kaggle_input.rglob("HYDRA4_FEAT_*.parquet"))
        tbar_pqs  = list(kaggle_input.rglob("HYDRA4_TBARS_*.parquet"))
        tick_pqs  = list(kaggle_input.rglob("HYDRA4_TICKS_*.parquet"))
        # mk4.7: tick-bar parquets (HYDRA4_TBARS_*) are the new default for
        # the scalp / hedge pipelines. Old M5-bar parquets (HYDRA4_FEAT_*)
        # remain valid for the legacy directional path. Either is enough.
        # mk4.8: raw tick parquets (HYDRA4_TICKS_*) are the input the
        # strategies pipeline (H1/H2/H4) consumes — staged to data/ticks/.
        parquets = m5_pqs + tbar_pqs

        if zips:
            print(f"[notebook_run] found bundle zip: {zips[0]}")
            return f"file://{zips[0]}"

        # mk4.8: prefer raw ticks for the strategies pipeline. We stage them
        # into REPO_DIR/data/ticks/ via symlink (Kaggle /kaggle/input is
        # read-only but readable — symlinking avoids duplicating the 7.5 GB
        # tick payload into /kaggle/working).
        if tick_pqs and TRAIN_MODE == "strategies":
            target = REPO_DIR / "data" / "ticks"
            target.mkdir(parents=True, exist_ok=True)
            print(f"[notebook_run] staging {len(tick_pqs)} tick parquets → {target} (symlinks)")
            total_mb = 0
            for src in tick_pqs:
                dst = target / src.name
                total_mb += src.stat().st_size / 1e6
                if dst.exists() or dst.is_symlink():
                    continue
                try:
                    dst.symlink_to(src)
                except (OSError, NotImplementedError):
                    # Fallback: full copy (slower, uses /kaggle/working quota).
                    shutil.copy2(src, dst)
            print(f"[notebook_run]   staged {total_mb/1024:.1f} GB total")
            return "skip"

        if parquets:
            target = REPO_DIR / "data" / "parquet"
            target.mkdir(parents=True, exist_ok=True)
            print(f"[notebook_run] staging {len(parquets)} parquet "
                  f"({len(m5_pqs)} M5 + {len(tbar_pqs)} tick-bar) → {target}")
            for src in parquets:
                dst = target / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
            # Aux files best-effort
            data_root = REPO_DIR / "data"
            for name in ("economic_calendar.parquet", "fae_cache.parquet"):
                for src in kaggle_input.rglob(name):
                    shutil.copy2(src, data_root / name); break
            macro = data_root / "macro"; macro.mkdir(exist_ok=True)
            for name in ("cot_sentiment.parquet", "fred_macro.parquet"):
                for src in kaggle_input.rglob(name):
                    shutil.copy2(src, macro / name); break
            return "skip"

        # Last-resort fallback: tick parquets are present but the user
        # picked a non-strategies TRAIN_MODE. Stage them anyway so the
        # legacy directional path can rebuild M5 bars from them.
        if tick_pqs:
            target = REPO_DIR / "data" / "ticks"
            target.mkdir(parents=True, exist_ok=True)
            print(f"[notebook_run] staging {len(tick_pqs)} tick parquets → {target} "
                  f"(TRAIN_MODE={TRAIN_MODE!r} will need to build derived bars)")
            for src in tick_pqs:
                dst = target / src.name
                if dst.exists() or dst.is_symlink():
                    continue
                try:    dst.symlink_to(src)
                except Exception: shutil.copy2(src, dst)
            return "skip"

        print("[notebook_run] /kaggle/input/ tree:")
        _show(kaggle_input, max_entries=200)
        raise SystemExit(
            "No HYDRA4_data_bundle*.zip, HYDRA4_FEAT_*.parquet, "
            "HYDRA4_TBARS_*.parquet, or HYDRA4_TICKS_*.parquet under "
            "/kaggle/input/. Re-upload via the dataset's 'New Version' button."
        )

    if PLATFORM == "colab":
        # Mount Drive lazily, only if we actually need it.
        if BUNDLE_DRIVE_PATH and BUNDLE_DRIVE_PATH.startswith("/content/drive"):
            if not Path("/content/drive").exists():
                print("[notebook_run] mounting /content/drive (you'll be prompted)")
                from google.colab import drive
                drive.mount("/content/drive")
            if Path(BUNDLE_DRIVE_PATH).exists():
                return f"file://{BUNDLE_DRIVE_PATH}"
            raise SystemExit(
                f"BUNDLE_DRIVE_PATH not found: {BUNDLE_DRIVE_PATH}\n"
                "Upload HYDRA4_data_bundle.zip to that path in Drive, or set "
                "BUNDLE_HTTPS_URL / BUNDLE_LOCAL_PATH instead."
            )
        if BUNDLE_LOCAL_PATH and Path(BUNDLE_LOCAL_PATH).exists():
            return f"file://{BUNDLE_LOCAL_PATH}"
        raise SystemExit(
            "Set BUNDLE_DRIVE_PATH (recommended on Colab), BUNDLE_HTTPS_URL, "
            "or BUNDLE_LOCAL_PATH."
        )

    # cloud / local
    if BUNDLE_LOCAL_PATH:
        p = Path(BUNDLE_LOCAL_PATH).expanduser()
        if p.exists():
            return f"file://{p}"
        raise SystemExit(f"BUNDLE_LOCAL_PATH not found: {p}")

    # Last resort: maybe the user pre-extracted parquets into REPO_DIR/data/parquet/.
    target = REPO_DIR / "data" / "parquet"
    if target.exists() and any(target.glob("HYDRA4_FEAT_*.parquet")):
        print(f"[notebook_run] using pre-staged parquets in {target}")
        return "skip"

    raise SystemExit(
        f"On platform={PLATFORM!r} you must set one of: "
        "BUNDLE_HTTPS_URL, BUNDLE_LOCAL_PATH, or pre-stage parquet files into "
        f"{target}."
    )


bundle_url = _resolve_bundle_url()


# ---------------------------------------------------------------------------
# 3 · Hand off to runner.sh
# ---------------------------------------------------------------------------
env = {
    **os.environ,
    "REPO_URL":          REPO_URL,
    "REPO_BRANCH":       REPO_BRANCH,
    "REPO_DIR":          str(REPO_DIR),
    "OUT_DIR":           str(OUT_DIR),
    "BUNDLE_URL":        bundle_url,
    "EPOCHS":            str(EPOCHS),
    "SEED":              str(SEED),
    "RUN_AUDIT":         "1",
    "RUN_COMPLIANCE":    "1",
    "SYMBOLS":           SYMBOLS,
    "HYDRA_BATCH_SIZE":  HYDRA_BATCH_SIZE,
    "TRAIN_MODE":        TRAIN_MODE,
    "TRAIN_AGENT":       TRAIN_AGENT,
    "TRAIN_SAMPLER":     TRAIN_SAMPLER,
    "SAMPLES_PER_EPOCH": str(SAMPLES_PER_EPOCH),
    "PARALLEL_TRAINING": "1" if PARALLEL_TRAINING else "0",
    "MAX_PARALLEL_WORKERS": str(MAX_PARALLEL_WORKERS),
    # Phase 2 — scalp
    "SCALP_SYMBOLS":     SCALP_SYMBOLS,
    "SCALP_EPOCHS":      str(SCALP_EPOCHS),
    "SCALP_WINDOW":      str(SCALP_WINDOW),
    "SCALP_BATCH_SIZE":  str(SCALP_BATCH_SIZE),
    "SCALP_SHOULD_TRADE_THRESHOLD": str(SCALP_SHOULD_TRADE_THRESHOLD),
    # Phase 3 — hedge
    "HEDGE_PAIRS":       HEDGE_PAIRS,
    "HEDGE_EPOCHS":      str(HEDGE_EPOCHS),
    "HEDGE_BATCH_SIZE":  str(HEDGE_BATCH_SIZE),
    "COINT_P_THRESHOLD": str(COINT_P_THRESHOLD),
    "COINT_MIN_WINDOWS": str(COINT_MIN_WINDOWS),
    "COINT_WINDOW_BARS": str(COINT_WINDOW_BARS),
    # mk4.7 — rule_meta
    "META_SYMBOLS":      META_SYMBOLS,
    "META_ESTIMATORS":   str(META_ESTIMATORS),
    "META_MAX_DEPTH":    str(META_MAX_DEPTH),
    "META_COST_PIP":     str(META_COST_PIP),
    # mk4.8 — strategies (H1 + H2 + H4)
    "STRATEGIES_SYMBOLS":       STRATEGIES_SYMBOLS,
    "STRATEGIES_SELECTED":      STRATEGIES_SELECTED,
    "STRATEGIES_H1_TICKS_PER_BAR": str(STRATEGIES_H1_TICKS_PER_BAR),
    "STRATEGIES_H1_HORIZON":    str(STRATEGIES_H1_HORIZON),
    "STRATEGIES_H1_MAX_TICK_FILE_GB": str(STRATEGIES_H1_MAX_TICK_FILE_GB),
    "STRATEGIES_H2_DONCHIAN":   str(STRATEGIES_H2_DONCHIAN),
    "STRATEGIES_H2_TIMEOUT":    str(STRATEGIES_H2_TIMEOUT),
    "STRATEGIES_H4_TIMEFRAME":  STRATEGIES_H4_TIMEFRAME,
    "STRATEGIES_H4_FAST":       str(STRATEGIES_H4_FAST),
    "STRATEGIES_H4_SLOW":       str(STRATEGIES_H4_SLOW),
    "STRATEGIES_H4_MOM":        str(STRATEGIES_H4_MOM),
    "STRATEGIES_H4_MOM_LOOKBACKS": STRATEGIES_H4_MOM_LOOKBACKS,
    "STRATEGIES_H4_VOL_FILTER_RATIO": str(STRATEGIES_H4_VOL_FILTER_RATIO),
    "STRATEGIES_H4_VOL_FILTER_SHORT": str(STRATEGIES_H4_VOL_FILTER_SHORT),
    "STRATEGIES_H4_VOL_FILTER_LONG":  str(STRATEGIES_H4_VOL_FILTER_LONG),
    "STRATEGIES_H4_NO_SHORT":   "1" if STRATEGIES_H4_NO_SHORT else "0",
    "STRATEGIES_ESTIMATORS":    str(STRATEGIES_ESTIMATORS),
    "STRATEGIES_MAX_DEPTH":     str(STRATEGIES_MAX_DEPTH),
    "STRATEGIES_AUDIT_FIRST":   "1" if STRATEGIES_AUDIT_FIRST else "0",
    "STRATEGIES_WORKERS":       str(STRATEGIES_WORKERS),
    "STRATEGIES_USE_GPU":       "1" if STRATEGIES_USE_GPU else "0",
    "PYBIN":             sys.executable,
    # Inherit the resolved PAT so runner.sh can re-auth on its own git calls
    # (fetch/pull/etc.). Empty string if no PAT was found — runner.sh skips
    # auth in that case.
    "GH_PAT":            GH_PAT or "",
}

runner = REPO_DIR / "cloud" / "runner.sh"
subprocess.check_call(["chmod", "+x", str(runner)])
print(f"[notebook_run] handing off to {runner}")
subprocess.check_call(["bash", str(runner)], env=env)


# ---------------------------------------------------------------------------
# 4 · Surface the run manifest + download instructions
# ---------------------------------------------------------------------------
manifest = OUT_DIR / "run_manifest.txt"
if manifest.exists():
    print("\n=== run manifest ===")
    print(manifest.read_text())

# Find the latest ONNX bundle zip the runner produced.
zips = sorted(OUT_DIR.glob("HYDRA4_onnx_*.zip"),
              key=lambda p: p.stat().st_mtime, reverse=True)
zip_path = zips[0] if zips else None
zip_size_mb = (zip_path.stat().st_size / 1e6) if zip_path else 0.0

print(f"\nDone. Artifacts in {OUT_DIR}")
if zip_path is not None:
    print(f"ONNX bundle: {zip_path.name}  ({zip_size_mb:.1f} MB)")
print()
print("=" * 70)
print(" HOW TO DOWNLOAD THE TRAINED ONNX MODELS")
print("=" * 70)
if PLATFORM == "kaggle":
    print("Kaggle:")
    print("  1. After this cell finishes, click 'Save Version' → 'Save & Run All")
    print("     (Commit)' so /kaggle/working/ outputs get persisted.")
    print("  2. Once the version completes, open the notebook viewer page.")
    print("  3. Right sidebar → 'Output' tab → expand 'onnx_out/' → click any")
    print("     file's three-dot menu → Download. Or right-click the .zip:")
    if zip_path is not None:
        print(f"        onnx_out/{zip_path.name}")
    print("  4. CLI alternative: `kaggle kernels output <user>/<kernel-slug>`")
elif PLATFORM == "colab":
    print("Colab:")
    print("  Run this in a NEW cell to trigger a browser download:")
    print("      from google.colab import files")
    if zip_path is not None:
        print(f"      files.download('{zip_path}')")
    else:
        print(f"      files.download('{OUT_DIR}/HYDRA4_onnx_*.zip')")
    print("  Or copy to Drive (assuming /content/drive is mounted):")
    print(f"      !cp -r {OUT_DIR} /content/drive/MyDrive/HYDRA4_models/")
elif PLATFORM == "cloud":
    print("Cloud (RunPod / vast.ai / Lambda / Paperspace):")
    print(f"  scp -r <user>@<host>:{OUT_DIR}/  ./local_dir/")
    print(f"  rsync -avz <user>@<host>:{OUT_DIR}/  ./local_dir/")
    print(f"  Or upload to S3:  aws s3 cp {zip_path or OUT_DIR} s3://<bucket>/  --recursive")
else:
    print("Local:")
    print(f"  Files are already on disk at: {OUT_DIR}")
    print(f"  Drop them into MT5_COMMON_DIR (auto-detected by the EA).")
print("=" * 70)
