"""
baselines/xgb_direction.py — XGBoost 3-class direction baseline.

Gradient-boosted trees are the strongest classical control for tabular
financial features (calibrated probabilities, robust to feature scale,
no train/serve skew). AURUM must beat this on purged CV to justify the
deep stack.

Exports to ONNX via onnxmltools so it can run inside MT5 exactly like the
deep model — same deployment path, fair comparison.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import (
    FLAT_INPUT_DIM, N_DIRECTION_CLASSES, META_N_ESTIMATORS,
    META_MAX_DEPTH, META_LR, SEED,
)

log = logging.getLogger(__name__)


class XGBDirectionBaseline:
    """Thin wrapper around XGBClassifier with a fixed 3-class contract."""

    def __init__(self, n_estimators: int = META_N_ESTIMATORS,
                 max_depth: int = META_MAX_DEPTH, lr: float = META_LR,
                 use_gpu: bool = False):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            objective="multi:softprob",
            num_class=N_DIRECTION_CLASSES,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            n_jobs=0,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: tuple | None = None,
            balance: bool = True) -> "XGBDirectionBaseline":
        kw = {}
        if eval_set is not None:
            kw["eval_set"] = [eval_set]
            kw["verbose"] = False
        if balance:
            # Inverse-frequency sample weights — triple-barrier labels are
            # skewed, so balance the baseline exactly as AURUM is balanced.
            counts = np.bincount(y, minlength=N_DIRECTION_CLASSES).astype(np.float64)
            w_per_class = counts.sum() / np.maximum(counts, 1.0)
            kw["sample_weight"] = w_per_class[y]
        self.model.fit(X, y, **kw)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def export_onnx(self, path: Path) -> bool:
        """Export to ONNX. Returns True on success."""
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError:
            log.warning("onnxmltools not installed — skipping XGB ONNX export")
            return False
        if not self._fitted:
            raise RuntimeError("fit() before export_onnx()")
        onnx_model = convert_xgboost(
            self.model,
            initial_types=[("input", FloatTensorType([1, FLAT_INPUT_DIM]))],
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        log.info("[xgb-baseline] ONNX -> %s", path)
        return True


def create_xgb_baseline(use_gpu: bool = False) -> XGBDirectionBaseline:
    return XGBDirectionBaseline(use_gpu=use_gpu)
