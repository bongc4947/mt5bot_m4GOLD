"""
test_strategies_smoke.py — pytest-discoverable smoke tests.

Same coverage as audit_strategies.py::dynamic_audit but exposed as discrete
pytest cases so CI / IDE can run them individually. The synthetic-data
fixtures sit in audit_strategies; we just re-use them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import audit_strategies  # noqa: E402
import strategies_common as sc  # noqa: E402
import train_h1_orderflow as h1  # noqa: E402
import train_h2_session   as h2  # noqa: E402
import train_h4_trend     as h4  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_ticks():
    return audit_strategies._gen_synthetic_ticks(20000, seed=0)


@pytest.fixture(scope="module")
def synthetic_tickbars(synthetic_ticks):
    return sc.ticks_to_tickbars(synthetic_ticks, ticks_per_bar=100)


@pytest.fixture(scope="module")
def synthetic_m5():
    return audit_strategies._gen_synthetic_m5(3000, seed=1)


@pytest.fixture(scope="module")
def synthetic_h1_bars():
    bars = audit_strategies._gen_synthetic_m5(3000, seed=2)
    bars["time"] = pd.date_range("2020-01-01", periods=len(bars), freq="1h",
                                  tz="UTC")
    return bars


# ---------------------------------------------------------------------------
# Common-utility tests
# ---------------------------------------------------------------------------

class TestCommon:
    def test_pip_size_majors(self):
        assert sc.pip_size("EURUSD") == 1e-4
        assert sc.pip_size("USDJPY") == 1e-2
        assert sc.pip_size("SILVER") == 1e-2
        assert sc.pip_size("UK_100") == 1.0
        assert sc.pip_size("COPPER") == 1e-3
        assert sc.pip_size("BTCUSD") == 1.0

    def test_chronological_split_no_overlap(self):
        tr, va = sc.chronological_split(1000, val_frac=0.30, gap=20)
        assert tr.stop + 20 <= va.start
        # train + gap + val == n
        assert va.stop == 1000

    def test_passive_pf_returns_two_values(self):
        r = np.array([1.0, -1.0, 1.0, -1.0])
        pl, ps = sc.passive_pf(r, cost_per_bar=0.0)
        # Symmetric returns -> long PF = short PF = 1
        assert abs(pl - 1.0) < 1e-9 and abs(ps - 1.0) < 1e-9

    def test_skill_gate_pass(self):
        ok, ex = sc.skill_gate(model_pf=1.50, passive_long_pf=1.20,
                                 passive_short_pf=1.10, n_trades=100)
        assert ok and abs(ex - 0.30) < 1e-9

    def test_skill_gate_fails_low_n(self):
        ok, _ = sc.skill_gate(model_pf=2.0, passive_long_pf=1.0,
                               passive_short_pf=1.0, n_trades=10)
        assert not ok


# ---------------------------------------------------------------------------
# H1 — order-flow imbalance
# ---------------------------------------------------------------------------

class TestH1:
    def test_features_shape(self, synthetic_tickbars):
        f = h1.build_h1_features(synthetic_tickbars)
        assert f.shape == (len(synthetic_tickbars), h1.H1_FEATURE_DIM)
        assert np.all(np.isfinite(f))

    def test_features_deterministic(self, synthetic_tickbars):
        f1 = h1.build_h1_features(synthetic_tickbars)
        f2 = h1.build_h1_features(synthetic_tickbars)
        np.testing.assert_allclose(f1, f2)

    def test_features_causal(self, synthetic_tickbars):
        ok = audit_strategies._causality_test(
            h1.build_h1_features, synthetic_tickbars,
            row_under_test=len(synthetic_tickbars) // 2)
        assert ok, "H1 features depend on future rows — leak"

    def test_labels_are_forward_looking(self, synthetic_tickbars):
        y, _ = h1.make_labels_and_pnl(synthetic_tickbars, horizon=10,
                                       symbol="TEST")
        # The label IS allowed to look ahead — confirm it does, by
        # scrambling future and checking labels change.
        scrambled = synthetic_tickbars.copy()
        rng = np.random.default_rng(13)
        n = len(scrambled)
        scrambled.loc[n // 2 + 1:, "mid"] = rng.normal(1.1, 1e-4,
                                                         size=n - n // 2 - 1)
        y2, _ = h1.make_labels_and_pnl(scrambled, horizon=10, symbol="TEST")
        assert not np.array_equal(y, y2), \
            "Labels did not change when future was scrambled — label code is " \
            "not looking ahead, which means we're not learning anything"


# ---------------------------------------------------------------------------
# H2 — session-open Donchian breakout
# ---------------------------------------------------------------------------

class TestH2:
    def test_candidates_have_session_id(self, synthetic_m5):
        cand = h2.generate_session_candidates(synthetic_m5, donchian_window=20,
                                                sl_atr=0.5, tp_atr=1.5,
                                                timeout_bars=12)
        if len(cand) > 0:
            assert set(cand["session_id"].unique()).issubset({0, 1})

    def test_features_shape_when_candidates_exist(self, synthetic_m5):
        cand = h2.generate_session_candidates(synthetic_m5, donchian_window=20,
                                                sl_atr=0.5, tp_atr=1.5,
                                                timeout_bars=12)
        if len(cand) < 5:
            pytest.skip("not enough candidates on synthetic data")
        f = h2.build_h2_features(synthetic_m5, cand)
        assert f.shape == (len(cand), h2.H2_FEATURE_DIM)
        assert np.all(np.isfinite(f))

    def test_outcome_columns_sum_to_n(self, synthetic_m5):
        cand = h2.generate_session_candidates(synthetic_m5, donchian_window=20,
                                                sl_atr=0.5, tp_atr=1.5,
                                                timeout_bars=12)
        if len(cand) == 0:
            pytest.skip("no candidates")
        assert (cand["tp_hit"] + cand["sl_hit"] + cand["timeout"]).eq(1).all()


# ---------------------------------------------------------------------------
# H4 — trend-following H1 / H4
# ---------------------------------------------------------------------------

class TestH4:
    def test_ma_cross_positions_causal(self, synthetic_h1_bars):
        close = synthetic_h1_bars["close"].to_numpy()
        pos = h4._ma_cross_positions(close, fast=20, slow=50, allow_short=True)
        # Scramble future, recompute, position at middle must match
        test_row = len(close) // 2
        scrambled = close.copy()
        rng = np.random.default_rng(11)
        scrambled[test_row + 1:] = rng.normal(close[test_row], 1e-3,
                                                size=len(close) - test_row - 1)
        pos2 = h4._ma_cross_positions(scrambled, fast=20, slow=50, allow_short=True)
        assert pos[test_row] == pos2[test_row]

    def test_momentum_positions_in_range(self, synthetic_h1_bars):
        close = synthetic_h1_bars["close"].to_numpy()
        pos = h4._momentum_positions(close, lookback=24, allow_short=True)
        assert set(np.unique(pos)).issubset({-1.0, 0.0, 1.0})

    def test_random_walk_no_edge(self, synthetic_h1_bars):
        """A pure random-walk H1 series should NOT pass the H4 deploy gate.
        If it does, the metric is broken."""
        close = synthetic_h1_bars["close"].to_numpy()
        pos = h4._ma_cross_positions(close, fast=20, slow=50, allow_short=True)
        bar_ret, _ = h4._backtest_positions(close, pos, cost_per_turn=1e-4)
        s = sc.sharpe(bar_ret, periods_per_year=h4.PERIODS_PER_YEAR["1h"])
        # Allow some slack — 3000 bars of Gaussian noise can produce
        # spurious Sharpe but it should be small.
        assert abs(s) < 2.0, f"Random-walk Sharpe={s:.2f} — likely a leak"
