"""
models/xgb_head.py — XGBoost direction head with ONNX export.

mk4.4 #2: Gradient-boosted trees usually outperform MLPs on
tabular features in the 200-dim / 500K-1.4M-sample regime, give
better-calibrated probabilities (so CONF_THRESHOLD actually filters
by quality rather than saturation), and produce free feature
importance for diagnostics.

Same input shape as PRISM (200-dim float32) and same output (1-dim
probability), so this drops in as a peer model class. ONNX export
goes via onnxmltools.convert_xgboost which produces an opset-12
compatible file the MT5 ONNX runtime can load.

USAGE
-----
    from models.xgb_head import XGBDirectionHead, train_and_export
    head = XGBDirectionHead.train(X_train, y_train, X_val, y_val,
                                   n_estimators=500, max_depth=6,
                                   learning_rate=0.05)
    head.export_onnx("HYDRA4_PRISM_EURUSD_dir_det.onnx", n_features=200)
    val_acc = head.score(X_val, y_val)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class XGBDirectionHead:
    """Thin wrapper around an XGBClassifier with a trainer + ONNX exporter."""

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(cls,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val:   np.ndarray, y_val:   np.ndarray,
              *,
              n_estimators: int = 500,
              max_depth:    int = 6,
              learning_rate: float = 0.05,
              min_child_weight: int = 5,
              reg_alpha:  float = 0.1,
              reg_lambda: float = 1.0,
              subsample:  float = 0.85,
              colsample_bytree: float = 0.85,
              early_stopping_rounds: int = 25,
              use_gpu: bool = True,
              seed: int = 42,
              ) -> "XGBDirectionHead":
        """
        Binary direction classifier. y in {0, 1} where 1 = LONG profitable.

        On a Colab/Kaggle T4 with 500K samples × 200 dims, full training
        takes ~60-180 s. Early-stopping watches val log-loss.
        """
        from xgboost import XGBClassifier

        kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds,
            verbosity=1,
        )
        if use_gpu:
            kwargs["tree_method"] = "hist"
            kwargs["device"]      = "cuda"
        else:
            kwargs["tree_method"] = "hist"

        clf = XGBClassifier(**kwargs)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        log.info("XGB train done: best_iter=%d  best_val_logloss=%.5f",
                 clf.best_iteration, clf.best_score)
        return cls(clf)

    # ------------------------------------------------------------------
    # Inference / scoring
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of LONG (class 1)."""
        p = self.model.predict_proba(X)
        # XGBClassifier returns shape (N, 2) for binary
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.reshape(-1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        proba = self.predict_proba(X)
        preds = (proba > 0.5).astype(np.int8)
        return float((preds == y.astype(np.int8)).mean())

    def feature_importance(self, top_k: int = 20) -> list[tuple[int, float]]:
        """Top-k (feature_index, gain) tuples — free diagnostic for free."""
        booster = self.model.get_booster()
        scores = booster.get_score(importance_type="gain")  # {"f0": 0.12, ...}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(int(k.lstrip("f")), v) for k, v in ranked[:top_k]]

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------
    def export_onnx(self, out_path: str | Path, n_features: int,
                    target_opset: int = 12) -> Path:
        """
        Convert the trained XGBClassifier to ONNX.

        n_features must match the model's input dim (200 in mk4.3+). The
        resulting file produces a single float probability output —
        compatible with the MT5 ONNX runtime path the EA already uses
        for the PRISM / APEX / GNN / CE direction heads.
        """
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError as exc:
            raise SystemExit(
                "onnxmltools not installed. Add to requirements-train.txt:\n"
                "    onnxmltools>=1.12.0\n"
                f"({exc})"
            )

        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(self.model, initial_types=initial_type,
                                     target_opset=target_opset)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            f.write(onnx_model.SerializeToString())

        # Sanity-check the export round-trips.
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out_path),
                                         providers=["CPUExecutionProvider"])
            _ = sess.run(None, {"input": np.zeros((1, n_features), np.float32)})
            log.info("XGB ONNX exported + validated: %s", out_path)
        except Exception as e:
            log.warning("XGB ONNX export written but validation failed: %s", e)
        return out_path
