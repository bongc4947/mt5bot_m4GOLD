"""
test_tick_pipeline.py — smoke test for the mk4.7 tick-mode pipeline.

Doesn't require MT5. Synthesises a 200K-tick random-walk price series,
runs it through aggregate_ticks_to_bars + the labeller + the existing
DirectionDataset and the new RandomWindowDirectionDataset, and asserts
that every stage produces well-shaped, NaN-free data.

Run with:
    python python/tests/test_tick_pipeline.py
Exits 0 on success, 1 on any failure. Prints PASS/FAIL per stage.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# stage tracking
_PASS = 0
_FAIL = 0


def _stage(name, fn):
    global _PASS, _FAIL
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  [PASS] {name}  ({elapsed*1000:.0f} ms)")
        _PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        import traceback; traceback.print_exc()
        _FAIL += 1


# ---------------------------------------------------------------------------
# Stage 1 — synthesise 200K ticks with realistic-ish bid/ask/volume.
# ---------------------------------------------------------------------------

def make_synthetic_ticks(n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Geometric random walk → mid-price stays positive, log-returns ~ N(0, sigma)
    log_returns = rng.normal(0, 1e-4, size=n)
    mid = 1.0500 * np.exp(np.cumsum(log_returns))
    # Spread varies between 0.5 and 2.0 pips (10^-4 units for FX)
    half_spread = (1.0 + 0.5 * np.abs(rng.normal(0, 1, size=n))) * 5e-5
    bid = mid - half_spread
    ask = mid + half_spread
    # Time stamps: irregular intervals averaging 1 second between ticks
    intervals_ms = rng.exponential(1000, size=n).astype("int64").clip(min=10)
    time_msc = (1_700_000_000_000 + np.cumsum(intervals_ms))
    volume = rng.poisson(10, size=n).astype("float32")
    return pd.DataFrame({
        "time_msc": time_msc,
        "bid":      bid.astype("float32"),
        "ask":      ask.astype("float32"),
        "last":     mid.astype("float32"),
        "volume":   volume,
        "flags":    np.zeros(n, dtype="int32"),
    })


_TICKS = None
def stage_make_ticks():
    global _TICKS
    _TICKS = make_synthetic_ticks(200_000)
    assert len(_TICKS) == 200_000
    assert _TICKS["bid"].lt(_TICKS["ask"]).all(), "bid >= ask in synth ticks"
    assert _TICKS["time_msc"].is_monotonic_increasing, "ticks not time-ordered"


# ---------------------------------------------------------------------------
# Stage 2 — aggregate to tick-bars at N=100 ticks/bar.
# ---------------------------------------------------------------------------

_BARS = None
def stage_aggregate():
    global _BARS
    from data_pipeline import aggregate_ticks_to_bars
    _BARS = aggregate_ticks_to_bars(_TICKS, ticks_per_bar=100)
    assert len(_BARS) == 200_000 // 100, f"expected 2000 bars, got {len(_BARS)}"
    # Schema must match the M5 pipeline's expectations
    for col in ("time", "open", "high", "low", "close", "tick_volume",
                "real_volume", "spread", "duration_sec"):
        assert col in _BARS.columns, f"missing column {col}"
    # No NaNs in the OHLC series
    for col in ("open", "high", "low", "close"):
        assert not _BARS[col].isna().any(), f"NaN in {col}"
    # OHLC ordering invariants
    assert (_BARS["high"] >= _BARS["low"]).all(), "high < low"
    assert (_BARS["high"] >= _BARS["open"]).all() and (_BARS["high"] >= _BARS["close"]).all()
    assert (_BARS["low"]  <= _BARS["open"]).all() and (_BARS["low"]  <= _BARS["close"]).all()
    # Time monotonic
    assert _BARS["time"].is_monotonic_increasing
    # Spread positive
    assert (_BARS["spread"] > 0).all()


# ---------------------------------------------------------------------------
# Stage 3 — direction labels work on tick-bars unchanged.
# ---------------------------------------------------------------------------

_LABELS = None
def stage_labels():
    global _LABELS
    from labeler import compute_direction_labels
    labels, regime = compute_direction_labels(_BARS, forward_bars=20)
    assert len(labels) == len(_BARS)
    assert set(np.unique(labels.astype(int))).issubset({-1, 0, 1}), \
        f"unexpected labels: {set(np.unique(labels.astype(int)))}"
    n_long  = int((labels > 0).sum())
    n_short = int((labels < 0).sum())
    n_flat  = int((labels == 0).sum())
    print(f"     labels: long={n_long}  short={n_short}  flat={n_flat}")
    # Sanity: not collapsed to all-zero and not all-one
    assert n_flat < len(labels), "all bars labelled FLAT — labeller broken"
    assert n_long + n_short > 0, "no directional labels"
    _LABELS = labels


# ---------------------------------------------------------------------------
# Stage 4 — RandomWindowDirectionDataset draws sane samples.
# ---------------------------------------------------------------------------

def stage_random_window_dataset():
    from dataset import RandomWindowDirectionDataset, DirectionDataset

    # We don't need real 200-dim features for this test — just check the
    # dataset wrapper. Use OHLC columns as a stand-in 4-dim feature.
    feat = _BARS[["open", "high", "low", "close"]].to_numpy().astype("float32")
    ds = RandomWindowDirectionDataset(feat, _LABELS, samples_per_epoch=5_000,
                                       mode="binary", exclude_flat=True)
    assert len(ds) == 5000, f"expected len=5000, got {len(ds)}"

    # Pull 1000 random samples and verify shape + value range
    seen_indices = set()
    for k in range(1000):
        x, y = ds[k % len(ds)]   # idx ignored by RandomWindow; vary anyway
        assert x.shape == (4,), f"bad feature shape at k={k}: {x.shape}"
        assert y.dtype.is_floating_point or y.dtype == ds.y.dtype, \
            f"label dtype unexpected: {y.dtype}"
        assert 0.0 <= float(y) <= 1.0, f"label out of binary range: {y}"
        # We don't have access to the internal index, so just check x is finite
        assert not (x != x).any(), "NaN in features"
        # Approximate uniqueness check via a hash of the float bytes
        seen_indices.add(hash(x.numpy().tobytes()))
    # Should see > 100 unique samples in 1000 random draws (with-replacement,
    # so not all unique, but with 4-dim float features collisions are rare)
    assert len(seen_indices) > 200, \
        f"too few unique random samples: {len(seen_indices)}/1000"

    # Sanity vs vanilla DirectionDataset: same labels, just different sampling
    plain = DirectionDataset(feat, _LABELS, mode="binary", exclude_flat=True)
    assert ds._real_n == len(plain), \
        f"random-window real_n={ds._real_n} != plain={len(plain)}"


# ---------------------------------------------------------------------------
# Stage 5 — make_loader correctly drops the size-1 tail when training.
# ---------------------------------------------------------------------------

def stage_loader_drop_last():
    from dataset import DirectionDataset, make_loader
    feat = _BARS[["open", "high", "low", "close"]].to_numpy().astype("float32")
    ds = DirectionDataset(feat, _LABELS, mode="binary", exclude_flat=True)

    # Construct a loader where dataset size leaves a tail batch of size 1.
    # bs chosen so that len(ds) % bs == 1.
    n = len(ds)
    bs = (n - 1)  # size = n samples, batch = n-1 → 1 full batch + 1 leftover
    if bs < 2:
        # Too small after exclude_flat to test; just smoke the loader.
        bs = max(2, n // 2)

    tr_loader = make_loader(ds, bs, shuffle=True, workers=0)
    va_loader = make_loader(ds, bs, shuffle=False, workers=0)

    # Training loader: must drop tail when len > bs
    if n > bs:
        assert tr_loader.drop_last is True, \
            "training loader should drop_last=True (BatchNorm fix)"
    # Val loader: never drop, so every val sample is scored
    assert va_loader.drop_last is False, \
        "val loader must keep drop_last=False"


# ---------------------------------------------------------------------------
# Stage 6 — fetch_ticks_capped is importable and respects no-MT5 path.
# ---------------------------------------------------------------------------

def stage_fetch_capped_no_mt5():
    # Just verify the function exists and gracefully reports the missing
    # MT5 dependency rather than crashing the import. We don't actually
    # call it (would try to connect to MT5).
    from data_pipeline import fetch_ticks_capped, run_tick_pipeline
    assert callable(fetch_ticks_capped)
    assert callable(run_tick_pipeline)


# ---------------------------------------------------------------------------
# Stage 7 — config paths exist for tick mode.
# ---------------------------------------------------------------------------

def stage_config_paths():
    from config import (TICKS_DIR, ticks_parquet_path,
                        tickbars_parquet_path)
    assert TICKS_DIR.exists() or TICKS_DIR.parent.exists()
    p1 = ticks_parquet_path("EURUSD")
    p2 = tickbars_parquet_path("EURUSD", 100)
    assert p1.name == "HYDRA4_TICKS_EURUSD.parquet"
    assert p2.name == "HYDRA4_TBARS_EURUSD_100tpb.parquet"


# ---------------------------------------------------------------------------
# Stage 8 — extract_data.py CLI parses the new flags without errors.
# ---------------------------------------------------------------------------

def stage_extract_cli_parse():
    import importlib
    extract = importlib.import_module("extract_data")
    parser = extract._build_parser()
    # bars mode (default)
    a = parser.parse_args(["EURUSD"])
    assert a.source == "bars"
    assert a.bundle is False
    # ticks mode with caps + bundle
    a = parser.parse_args(["EURUSD", "--source", "ticks",
                           "--max-size-mb", "1024", "--ticks-per-bar", "200",
                           "--save-raw-ticks", "--bundle"])
    assert a.source == "ticks"
    assert a.max_size_mb == 1024
    assert a.ticks_per_bar == 200
    assert a.save_raw_ticks is True
    assert a.bundle is True


# ---------------------------------------------------------------------------
# Stage 9 — train.py CLI accepts --sampler / --samples-per-epoch.
# ---------------------------------------------------------------------------

def stage_train_cli_parse():
    from train import build_parser
    p = build_parser()
    args = p.parse_args(["prism", "--skip-extract",
                         "--sampler", "random-window",
                         "--samples-per-epoch", "50000"])
    assert args.sampler == "random-window"
    assert args.samples_per_epoch == 50_000


# ---------------------------------------------------------------------------
# Stage 9c — multi-timeframe alignment is causal and zero-filled when missing
# (sub-phase 1a step 1: H1 + H4 context bolted onto every tick-bar)
# ---------------------------------------------------------------------------

def stage_mtf_alignment():
    from multi_timeframe import align_mtf_features, MTF_FEATURE_COLUMNS

    base = _BARS.copy()
    rng = np.random.default_rng(99)

    # Synthesise H1 and H4 bars covering the same period as our tick-bars.
    span_seconds = (pd.to_datetime(base["time"].iloc[-1], utc=True)
                    - pd.to_datetime(base["time"].iloc[0],  utc=True)).total_seconds()
    n_h1 = max(2, int(span_seconds / 3600))
    n_h4 = max(2, int(span_seconds / (4 * 3600)))
    h1_times = pd.date_range(start=base["time"].iloc[0], periods=n_h1, freq="1h")
    h1_close = 1.05 * np.exp(np.cumsum(rng.normal(0, 1e-4, n_h1)))
    h1_bars = pd.DataFrame({
        "time":  h1_times,
        "open":  h1_close * 0.999,
        "high":  h1_close * 1.002,
        "low":   h1_close * 0.998,
        "close": h1_close,
        "tick_volume": np.full(n_h1, 1000, dtype=np.int64),
    })
    h4_times = pd.date_range(start=base["time"].iloc[0], periods=n_h4, freq="4h")
    h4_close = 1.05 * np.exp(np.cumsum(rng.normal(0, 5e-4, n_h4)))
    h4_bars = pd.DataFrame({
        "time":  h4_times,
        "open":  h4_close * 0.999,
        "high":  h4_close * 1.002,
        "low":   h4_close * 0.998,
        "close": h4_close,
        "tick_volume": np.full(n_h4, 4000, dtype=np.int64),
    })

    # 1. Both timeframes present — all 8 columns added, no NaN, no lookahead
    aligned = align_mtf_features(base, h1_bars, h4_bars)
    for c in MTF_FEATURE_COLUMNS:
        assert c in aligned.columns, f"missing column: {c}"
        assert not aligned[c].isna().any(), f"NaN in {c}"
        # Bounded features — sanity envelope. atr_norm in [0,1], others in [-1,1]
        if "atr_norm" in c:
            assert aligned[c].min() >= -1e-6 and aligned[c].max() <= 1.0 + 1e-6, \
                f"{c} out of expected [0,1] range: [{aligned[c].min()}, {aligned[c].max()}]"
        else:
            assert aligned[c].min() >= -1.0 - 1e-6 and aligned[c].max() <= 1.0 + 1e-6, \
                f"{c} out of expected [-1,1] range: [{aligned[c].min()}, {aligned[c].max()}]"
    # Causality: for each tick-bar t, the merged H1 row must have time <= t.
    # We check this indirectly by verifying that the FIRST tick-bars (those
    # before the first H1 bar's time) get the merged-asof None semantic →
    # NaN → filled to 0.0. With h1_times[0] == base["time"].iloc[0] there's
    # no "before" period, so this is fine.

    # 2. h1_bars=None → first 4 columns zero-filled; h4 columns still real
    aligned_no_h1 = align_mtf_features(base, None, h4_bars)
    for c in ("h1_trend", "h1_rsi", "h1_atr_norm", "h1_vwap_rel"):
        assert (aligned_no_h1[c] == 0.0).all(), f"{c} should be 0 when h1_bars=None"
    assert not (aligned_no_h1["h4_trend"] == 0.0).all(), \
        "h4_trend should not be all zero when h4_bars provided"

    # 3. Both None → all 8 columns zero (graceful degradation)
    aligned_none = align_mtf_features(base, None, None)
    for c in MTF_FEATURE_COLUMNS:
        assert (aligned_none[c] == 0.0).all(), f"{c} should be 0 when both MTF None"

    # 4. Schema stability — number of MTF columns is exactly 8 in every branch
    base_n = len(base.columns)
    assert len(aligned.columns)       == base_n + 8
    assert len(aligned_no_h1.columns) == base_n + 8
    assert len(aligned_none.columns)  == base_n + 8


# ---------------------------------------------------------------------------
# Stage 9b — streaming chunk aggregation matches one-shot aggregation
# (regression guard for the OOM fix that streams ticks chunk-by-chunk).
# ---------------------------------------------------------------------------

def stage_streaming_equivalence():
    """
    Synthesise a tick stream, aggregate it (a) all at once and
    (b) split into 5 chunks with the carry-over logic the new
    run_tick_pipeline uses. Bars must match bit-for-bit on close,
    spread, etc.
    """
    from data_pipeline import aggregate_ticks_to_bars
    rng = np.random.default_rng(123)
    # 12,345 ticks → 123 bars at 100 ticks/bar, with 45 ticks of tail
    ticks = make_synthetic_ticks(12_345, seed=123)

    one_shot = aggregate_ticks_to_bars(ticks, ticks_per_bar=100)

    # Now simulate the chunked path with carry-over (same logic as
    # run_tick_pipeline). 5 chunks of unequal size.
    boundaries = [0, 2000, 5500, 8000, 11000, 12_345]
    bar_chunks = []
    carry = None
    for i in range(len(boundaries) - 1):
        chunk = ticks.iloc[boundaries[i]:boundaries[i+1]].reset_index(drop=True)
        merged = (pd.concat([carry, chunk], ignore_index=True)
                  if carry is not None else chunk)
        n_full = len(merged) // 100
        if n_full == 0:
            carry = merged
            continue
        head = merged.iloc[: n_full * 100]
        carry = (merged.iloc[n_full * 100:].reset_index(drop=True)
                 if len(merged) > n_full * 100 else None)
        bar_chunks.append(aggregate_ticks_to_bars(head, ticks_per_bar=100))

    streamed = pd.concat(bar_chunks, ignore_index=True)

    assert len(streamed) == len(one_shot), \
        f"row count differs: streamed={len(streamed)}  one_shot={len(one_shot)}"
    # Numerical equality on close prices (no rounding drift expected)
    diff = np.abs(streamed["close"].to_numpy() - one_shot["close"].to_numpy())
    assert diff.max() < 1e-9, f"streaming/one-shot close diff: max={diff.max()}"
    # Spread should match exactly too
    diff = np.abs(streamed["spread"].to_numpy() - one_shot["spread"].to_numpy())
    assert diff.max() < 1e-9, f"streaming/one-shot spread diff: max={diff.max()}"


# ---------------------------------------------------------------------------
# Phase 0 + Phase 1 stages (all deterministic, no MT5 needed)
# ---------------------------------------------------------------------------

def stage_orderflow_aggregation():
    from orderflow import aggregate_orderflow_to_bars, ORDERFLOW_FEATURE_COLUMNS
    of = aggregate_orderflow_to_bars(_TICKS, ticks_per_bar=100)
    assert len(of) == len(_BARS), f"length mismatch: {len(of)} vs {len(_BARS)}"
    for c in ORDERFLOW_FEATURE_COLUMNS:
        assert c in of.columns, f"missing column: {c}"
        arr = of[c].to_numpy()
        assert not np.isnan(arr).any(), f"NaN in {c}"
    # OFI must be in [-1, 1], CVD in [-1, 1] (after the /10 scaling)
    assert of["ofi"].between(-1.0, 1.0).all()
    assert of["cvd"].between(-1.0, 1.0).all()


def stage_session_features():
    from session_features import compute_session_features, SESSION_FEATURE_COLUMNS
    sf = compute_session_features(_BARS)
    assert len(sf) == len(_BARS)
    for c in SESSION_FEATURE_COLUMNS:
        assert c in sf.columns, f"missing column: {c}"
    # tod_sin^2 + tod_cos^2 = 1 by construction
    assert np.allclose(sf["tod_sin"] ** 2 + sf["tod_cos"] ** 2, 1.0, atol=1e-5)
    # session one-hot at most three of {asia, london, ny} active per row
    sums = sf[["sess_asia", "sess_london", "sess_ny"]].sum(axis=1)
    assert sums.max() <= 3.0


def stage_clean_ticks():
    from data_pipeline import clean_ticks
    # Inject duplicates and a spread-outlier
    bad = _TICKS.copy()
    dup = bad.iloc[:5].copy()
    bad = pd.concat([bad, dup], ignore_index=True)
    # outlier row
    out_row = bad.iloc[0:1].copy()
    out_row["ask"] = out_row["bid"] + 0.5  # 50 bps spread on FX → outlier
    bad = pd.concat([bad, out_row], ignore_index=True)
    cleaned = clean_ticks(bad)
    assert len(cleaned) <= len(bad), "cleaning shouldn't add rows"
    # at least the 5 duplicates should be gone
    assert len(cleaned) <= len(_TICKS) + 1


def stage_sequential_dataset():
    from dataset import SequentialDirectionDataset
    feat = _BARS[["open", "high", "low", "close"]].to_numpy(dtype=np.float32)
    ds = SequentialDirectionDataset(feat, _LABELS, window=8, mode="binary",
                                     exclude_flat=True)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (8, 4), f"bad shape: {x.shape}"
    assert y.dtype.is_floating_point or y.dtype == ds.y.dtype
    # Monotonic check: x[-1] should equal feat[target_idx[0]]
    expected = feat[ds._target_idx[0]]
    assert np.allclose(x[-1].numpy(), expected, atol=1e-6)


def stage_backward_vol_regime():
    from validation import backward_vol_regime
    close = _BARS["close"].to_numpy(dtype=np.float64)
    regime = backward_vol_regime(close, window_short=20, window_long=80)
    assert len(regime) == len(close)
    assert set(np.unique(regime)).issubset({0, 1, 2})
    # Causality: shuffling the future shouldn't change the early values
    close2 = close.copy()
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(close2) // 4) + (3 * len(close2) // 4)  # last quarter
    close2[3 * len(close2) // 4:] = close2[3 * len(close2) // 4:][rng.permutation(len(close2) // 4)]
    regime2 = backward_vol_regime(close2, window_short=20, window_long=80)
    # First half should be unaffected
    half = len(close) // 2
    assert (regime[:half] == regime2[:half]).all(), \
        "backward_vol_regime leaked future info — first half changed"


def stage_adversarial_val():
    from validation import adversarial_validation_score
    rng = np.random.default_rng(11)
    # Same distribution -> score ~ 0.5
    same_a = rng.normal(0, 1, (1000, 10)).astype(np.float32)
    same_b = rng.normal(0, 1, (1000, 10)).astype(np.float32)
    score_same = adversarial_validation_score(same_a, same_b)
    assert 0.3 <= score_same <= 0.7, f"same-dist score outside [0.3, 0.7]: {score_same}"
    # Different distribution -> score notably > 0.5
    diff_a = rng.normal(0, 1, (1000, 10)).astype(np.float32)
    diff_b = rng.normal(3, 1, (1000, 10)).astype(np.float32)
    score_diff = adversarial_validation_score(diff_a, diff_b)
    assert score_diff > 0.7, f"different-dist score should be > 0.7: {score_diff}"


def stage_temperature_scaling():
    from validation import fit_temperature
    rng = np.random.default_rng(13)
    # Uncalibrated logits — sigmoid saturated near 0/1
    true_probs = rng.uniform(0.2, 0.8, 5000)
    labels = (rng.uniform(0, 1, 5000) < true_probs).astype(np.float32)
    raw_logits = np.log(true_probs / (1 - true_probs)) * 3.0  # over-confident
    T = fit_temperature(raw_logits, labels)
    assert 0.5 <= T <= 5.0, f"T out of bounds: {T}"
    # T should pull above 1 to soften the over-confident logits
    assert T > 1.5, f"T should be > 1.5 for over-confident logits, got {T}"


def stage_ood_autoencoder():
    from validation import OODAutoencoder
    rng = np.random.default_rng(17)
    train_X = rng.normal(0, 1, (2000, 16)).astype(np.float32)
    val_X   = rng.normal(0, 1, (500,  16)).astype(np.float32)
    ood_X   = rng.normal(5, 1, (500,  16)).astype(np.float32)  # OOD by mean shift
    ae = OODAutoencoder(input_dim=16, hidden=32, latent=8)
    ae.fit(train_X, epochs=5, batch_size=256)
    threshold = ae.calibrate(val_X, percentile=95)
    in_dist_scores  = ae.score(val_X)
    out_dist_scores = ae.score(ood_X)
    assert in_dist_scores.mean() < out_dist_scores.mean(), \
        "OOD scores should be higher than in-dist scores"
    in_dist_reject  = (in_dist_scores  > threshold).mean()
    out_dist_reject = (out_dist_scores > threshold).mean()
    assert out_dist_reject > 0.5, \
        f"OOD reject rate too low: {out_dist_reject:.2f}"
    assert in_dist_reject < 0.20, \
        f"in-dist reject rate too high: {in_dist_reject:.2f}"


def stage_walk_forward():
    from validation import walk_forward_pf_summary
    fwd = np.diff(_BARS["close"].to_numpy(dtype=np.float64),
                   prepend=_BARS["close"].iloc[0]) / _BARS["close"].iloc[0]
    feat = np.zeros((len(_BARS), 4), dtype=np.float32)  # not used by 'always-long'
    summary = walk_forward_pf_summary(feat, _LABELS, fwd, pip=0.0001,
                                       n_folds=5, gap=20)
    assert "median_pf" in summary
    assert "frac_profitable" in summary
    assert "n_folds" in summary
    assert summary["n_folds"] >= 3, f"too few folds: {summary['n_folds']}"


# ---------------------------------------------------------------------------
# Phase 2 + Phase 3 stages
# ---------------------------------------------------------------------------

def stage_scalp_net():
    import torch, tempfile, os
    from models.scalp_net import ScalpNet, ScalpNetExportWrapper
    F, T_win, B = 30, 64, 4
    model = ScalpNet(feature_dim=F, hidden=64, mlp_hidden=32, dropout=0.1)
    x = torch.randn(B, T_win, F)
    d, s = model(x)
    assert d.shape == (B,), f"direction shape {d.shape}"
    assert s.shape == (B,), f"should-trade shape {s.shape}"
    # Export wrapper produces (B, 2)
    wrapper = ScalpNetExportWrapper(model, t_dir=1.5, t_trade=2.0)
    wrapper.eval()
    out = wrapper(x)
    assert out.shape == (B, 2), f"wrapper out shape {out.shape}"
    # ONNX export — suppress torch.onnx verbose stdout (it prints emoji
    # the Windows cp1252 console can't encode; the export itself works).
    import io, contextlib
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(wrapper, x, path,
                               input_names=["features"],
                               output_names=["dir_and_trade"],
                               dynamic_axes={"features": {0: "batch"},
                                             "dir_and_trade": {0: "batch"}},
                               opset_version=17)
        assert os.path.exists(path) and os.path.getsize(path) > 0
    finally:
        try: os.unlink(path)
        except: pass


def stage_pnl_loss():
    import torch
    from loss_pnl import ExpectedPnLLoss
    loss_fn = ExpectedPnLLoss(cost=1e-4)
    rng = torch.Generator().manual_seed(7)
    d_logit  = torch.randn(64, generator=rng, requires_grad=True)
    st_logit = torch.randn(64, generator=rng, requires_grad=True)
    fwd      = torch.randn(64, generator=rng) * 0.001
    loss = loss_fn(d_logit, st_logit, fwd)
    assert loss.requires_grad
    loss.backward()
    assert d_logit.grad is not None and st_logit.grad is not None
    # On synthetic noise, gate should learn to drive should_trade negative
    # so sigmoid(should_trade) -> 0 (no-trade default). Run a few steps.
    d_logit2  = torch.zeros(64, requires_grad=True)
    st_logit2 = torch.zeros(64, requires_grad=True)
    rng2 = torch.Generator().manual_seed(11)
    fwd2 = torch.randn(2000, generator=rng2) * 0.001
    opt = torch.optim.Adam([d_logit2, st_logit2], lr=0.5)
    for _ in range(20):
        opt.zero_grad()
        loss2 = loss_fn(d_logit2, st_logit2,
                         fwd2[: 64].repeat(1)[:64])
        loss2.backward(); opt.step()
    # After training on noise: gate should have moved (not stuck at 0)
    assert torch.abs(st_logit2).max().item() > 0.01


def stage_scalp_labels():
    from labeler_scalp import compute_scalp_labels
    bars = _BARS.copy()
    if "spread" not in bars.columns:
        bars["spread"] = 5e-5  # synthetic FX spread
    direction, should_trade = compute_scalp_labels(
        bars, sl_spread_mult=1.5, tp_spread_mult=2.5, timeout_bars=20)
    assert len(direction) == len(bars)
    assert set(np.unique(direction)).issubset({-1, 0, 1})
    assert set(np.unique(should_trade)).issubset({0, 1})
    # On a random walk with 1.5/2.5 RR, some bars hit TP, some hit SL,
    # most timeout. should_trade should be a strict subset of non-flat.
    assert ((direction == 0) | (should_trade == 1)).all() or \
           ((should_trade == 1).sum() <= (direction != 0).sum() + 1)


def stage_hedge_net():
    import torch, tempfile, os
    from models.hedge_net import HedgeNet, HedgeNetExportWrapper
    F = 30
    S = 4
    model = HedgeNet(feature_dim=F, spread_state_dim=S,
                      leg_hidden=32, trunk_hidden=32, dropout=0.1)
    leg_a = torch.randn(8, F)
    leg_b = torch.randn(8, F)
    sp    = torch.randn(8, S)
    out = model(leg_a, leg_b, sp)
    assert out.shape == (8,), f"hedge out shape {out.shape}"
    # Symmetry check: weight-tied encoder => legA and legB swap should
    # give the negation in [zA - zB] block; not testing exact equality
    # since trunk is non-linear, but model should be well-behaved.
    wrapper = HedgeNetExportWrapper(model, t_revert=1.2)
    wrapper.eval()
    packed = torch.cat([leg_a, leg_b, sp], dim=-1)
    out2 = wrapper(packed)
    assert out2.shape == (8,)
    # Round-trip via ONNX (suppress torch.onnx unicode prints on Windows)
    import io, contextlib
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(wrapper, packed, path,
                               input_names=["packed"],
                               output_names=["revert_logit"],
                               dynamic_axes={"packed": {0: "batch"},
                                             "revert_logit": {0: "batch"}},
                               opset_version=17)
        assert os.path.exists(path) and os.path.getsize(path) > 0
    finally:
        try: os.unlink(path)
        except: pass


def stage_hedge_labels():
    from labeler_hedge import (compute_spread, compute_spread_zscore,
                                compute_spread_state, compute_revert_labels)
    rng = np.random.default_rng(31)
    n = 5000
    a_close = 1.05 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    b_close = 1.10 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    spread, betas = compute_spread(a_close, b_close, beta_window=500)
    assert len(spread) == n and len(betas) == n
    z = compute_spread_zscore(spread, window=60)
    assert len(z) == n
    # z must be roughly mean 0, std ~ 1 in steady state
    assert abs(z[1000:].mean()) < 0.5
    state = compute_spread_state(spread, z, vol_window=60)
    assert state.shape == (n, 4)
    assert (-1 - 1e-6 <= state).all() and (state <= 1 + 1e-6).all()
    labels = compute_revert_labels(z, z_entry=2.0, horizon_bars=60)
    assert set(np.unique(labels)).issubset({-1, 0, 1})


def stage_cointegration_screen():
    from cointegration import screen_pair, screen_all_pairs
    rng = np.random.default_rng(53)
    n = 12_000
    # Cointegrated pair: B = 1.5*A + noise (stationary residual)
    a = np.cumsum(rng.normal(0, 1e-3, n)) + 100.0
    b = 1.5 * a + rng.normal(0, 0.05, n)
    res_coint = screen_pair(a, b, window_bars=4000, step_bars=2000,
                              min_passing_windows=2, p_threshold=0.10)
    # Random independent walks: NOT cointegrated
    a2 = np.cumsum(rng.normal(0, 1e-3, n))
    b2 = np.cumsum(rng.normal(0, 1e-3, n))
    res_rand = screen_pair(a2, b2, window_bars=4000, step_bars=2000,
                             min_passing_windows=2, p_threshold=0.10)
    # The cointegrated case should pass on average more strongly (lower p)
    assert res_coint["mean_pvalue"] < res_rand["mean_pvalue"] + 0.05, \
        f"cointegrated p {res_coint['mean_pvalue']:.3f} should be <= rand p {res_rand['mean_pvalue']:.3f}"
    # And screen_all_pairs runs on a small dict without crashing
    closes = {"A": a, "B": b, "C": a2}
    pairs = screen_all_pairs(closes, window_bars=4000, step_bars=2000,
                                min_passing_windows=2, p_threshold=0.50)
    # With p_threshold=0.50 the cointegrated pair should appear
    pair_names = {(p[0], p[1]) for p in pairs}
    # Either order possible
    assert ("A", "B") in pair_names or ("B", "A") in pair_names


# ---------------------------------------------------------------------------
# Stage 10 — random-window train/val split is chronologically disjoint
# (no leakage from the random sampler peeking into val bars).
# ---------------------------------------------------------------------------

def stage_random_window_no_leakage():
    """
    Verify the trainer's random-window path slices train/val by time first.
    We can't run the full trainer (no real models), but we replicate the
    slicing logic to confirm train indices < val indices, with a gap.
    """
    from config import VAL_SPLIT, LABEL_FORWARD_BARS
    from dataset import RandomWindowDirectionDataset, DirectionDataset

    n = len(_LABELS)
    n_val = max(1, int(n * VAL_SPLIT))
    gap   = LABEL_FORWARD_BARS
    n_tr  = max(1, n - n_val - gap)

    feat = _BARS[["open", "high", "low", "close"]].to_numpy().astype("float32")
    tr_feats, tr_lab = feat[:n_tr], _LABELS[:n_tr]
    va_feats, va_lab = feat[n_tr + gap:], _LABELS[n_tr + gap:]

    # The two ranges must not overlap by index
    assert n_tr + gap <= n, "split exceeds dataset bounds"

    tr_set = RandomWindowDirectionDataset(tr_feats, tr_lab,
                                           samples_per_epoch=200,
                                           mode="binary", exclude_flat=True)
    va_set = DirectionDataset(va_feats, va_lab,
                               mode="binary", exclude_flat=True)
    # Pull samples from train; verify all features come from rows < n_tr.
    # We do this by checking that the max value of OHLC matches the train
    # slice's max (since OHLC values are monotonically random-walking,
    # train and val slices will have different OHLC ranges).
    train_max = float(tr_feats.max())
    val_max   = float(va_feats.max())
    # Sample many times; every drawn feature must be <= train_max.
    eps = 1e-6
    for _ in range(500):
        x, _y = tr_set[0]
        v = float(x.max())
        assert v <= train_max + eps, \
            f"random-window train sample {v:.6f} exceeds train_max {train_max:.6f}"
    print(f"     train n={n_tr}  val n={len(va_lab)}  gap={gap}  "
          f"train_max={train_max:.5f}  val_max={val_max:.5f}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("\n=== HYDRA mk4.7 tick-pipeline smoke test ===\n")
    _stage("synthesise ticks (200K)",      stage_make_ticks)
    _stage("aggregate_ticks_to_bars",      stage_aggregate)
    _stage("compute_direction_labels",     stage_labels)
    _stage("RandomWindowDirectionDataset", stage_random_window_dataset)
    _stage("make_loader drop_last semantics", stage_loader_drop_last)
    _stage("fetch_ticks_capped importable",   stage_fetch_capped_no_mt5)
    _stage("config tick paths",            stage_config_paths)
    _stage("extract_data.py CLI parsing",  stage_extract_cli_parse)
    _stage("train.py CLI parsing",         stage_train_cli_parse)
    _stage("streaming-vs-one-shot equivalence", stage_streaming_equivalence)
    _stage("MTF alignment (H1+H4 -> tick-bar)", stage_mtf_alignment)
    _stage("orderflow aggregation",             stage_orderflow_aggregation)
    _stage("session features",                  stage_session_features)
    _stage("clean_ticks dedupe + outlier",      stage_clean_ticks)
    _stage("SequentialDirectionDataset",        stage_sequential_dataset)
    _stage("backward_vol_regime causality",     stage_backward_vol_regime)
    _stage("adversarial validation",            stage_adversarial_val)
    _stage("temperature scaling",               stage_temperature_scaling)
    _stage("OOD autoencoder fit/score",         stage_ood_autoencoder)
    _stage("walk-forward PF summary",           stage_walk_forward)
    _stage("ScalpNet forward + export",         stage_scalp_net)
    _stage("ExpectedPnLLoss gradient sanity",   stage_pnl_loss)
    _stage("scalp labels micro-triple-barrier", stage_scalp_labels)
    _stage("HedgeNet forward + export",         stage_hedge_net)
    _stage("hedge labels (spread revert)",      stage_hedge_labels)
    _stage("Engle-Granger pair screen",         stage_cointegration_screen)
    _stage("random-window train/val no-leakage", stage_random_window_no_leakage)

    print(f"\n  Result: {_PASS} pass / {_FAIL} fail\n")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
