"""
cv/purged_kfold.py — leakage-free cross-validation for financial series.

Naive K-fold leaks: a label built from a forward horizon (triple-barrier
over LABEL_HORIZON_BARS) means sample i's outcome overlaps samples
i+1 .. i+horizon. If those land in the test fold while i is in train,
the model has effectively seen the future.

Two estimators, both from López de Prado, *Advances in Financial Machine
Learning* (2018), Ch.7:

  PurgedKFold
    Standard K contiguous test folds. For each, training samples whose
    label horizon overlaps the test window are PURGED, and an EMBARGO
    gap after the test window is also dropped (serial correlation leaks
    forward even without horizon overlap).

  CombinatorialPurgedCV
    Picks every C(N, k) combination of N groups as the test set, giving
    a *distribution* of backtest paths instead of a single number — far
    harder to overfit a strategy to.

Both operate on integer row indices + a scalar label horizon (in bars),
matching this repo's numpy-first style. No pandas Series of t1 required.
"""

from __future__ import annotations

import itertools
from typing import Iterator

import numpy as np


def _purge_train(train_mask: np.ndarray, test_lo: int, test_hi: int,
                 horizon: int, embargo: int) -> np.ndarray:
    """Zero out training rows that leak into [test_lo, test_hi)."""
    n = len(train_mask)
    # A train row t leaks if its label window [t, t+horizon] reaches the
    # test block, or if it sits inside the embargo right after the block.
    purge_lo = max(0, test_lo - horizon)
    purge_hi = min(n, test_hi + embargo)
    train_mask[purge_lo:purge_hi] = False
    return train_mask


class PurgedKFold:
    """
    K-fold with horizon purging + embargo.

    Parameters
    ----------
    n_splits : number of contiguous test folds.
    horizon  : label forward horizon in bars (triple-barrier window).
    embargo_pct : embargo gap as a fraction of dataset length.
    """

    def __init__(self, n_splits: int = 6, horizon: int = 20,
                 embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.horizon = int(horizon)
        self.embargo_pct = float(embargo_pct)

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) integer arrays."""
        embargo = int(n_samples * self.embargo_pct)
        edges = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)
        all_idx = np.arange(n_samples)
        for k in range(self.n_splits):
            test_lo, test_hi = edges[k], edges[k + 1]
            test_idx = all_idx[test_lo:test_hi]
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_lo:test_hi] = False
            train_mask = _purge_train(train_mask, test_lo, test_hi,
                                      self.horizon, embargo)
            train_idx = all_idx[train_mask]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (López de Prado, AFML Ch.12).

    Splits the series into `n_groups` contiguous groups, then uses every
    combination of `n_test_groups` groups as the test set. Yields
    C(n_groups, n_test_groups) splits — a distribution of backtest paths.
    """

    def __init__(self, n_groups: int = 8, n_test_groups: int = 2,
                 horizon: int = 20, embargo_pct: float = 0.01):
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.horizon = int(horizon)
        self.embargo_pct = float(embargo_pct)

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        embargo = int(n_samples * self.embargo_pct)
        edges = np.linspace(0, n_samples, self.n_groups + 1, dtype=int)
        all_idx = np.arange(n_samples)
        for combo in itertools.combinations(range(self.n_groups),
                                            self.n_test_groups):
            test_mask = np.zeros(n_samples, dtype=bool)
            train_mask = np.ones(n_samples, dtype=bool)
            for g in combo:
                lo, hi = edges[g], edges[g + 1]
                test_mask[lo:hi] = True
                train_mask[lo:hi] = False
            # Purge around every test block in the combo.
            for g in combo:
                lo, hi = edges[g], edges[g + 1]
                train_mask = _purge_train(train_mask, lo, hi,
                                          self.horizon, embargo)
            train_idx = all_idx[train_mask]
            test_idx = all_idx[test_mask]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        from math import comb
        return comb(self.n_groups, self.n_test_groups)


if __name__ == "__main__":
    # Smoke test — purged folds must never overlap the test horizon.
    pk = PurgedKFold(n_splits=5, horizon=20, embargo_pct=0.01)
    for i, (tr, te) in enumerate(pk.split(10_000)):
        gap = te.min() - tr[tr < te.min()].max() if (tr < te.min()).any() else 99
        print(f"fold {i}: train={len(tr):5d} test={len(te):5d} "
              f"left-gap={gap}")
    cp = CombinatorialPurgedCV(n_groups=8, n_test_groups=2, horizon=20)
    print(f"CombinatorialPurgedCV -> {cp.get_n_splits()} paths")
