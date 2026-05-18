"""
aurum/meta_label.py — L4 meta-label gate.

Meta-labeling (López de Prado, *Advances in Financial ML* 2018, Ch.3):
the primary model (AurumNet) decides *direction*; a separate, smaller
model decides whether to *act* on that direction at all. The meta-model
is trained on a binary target — "was the primary signal profitable?" —
and lifts precision sharply without touching recall of the primary.

The meta-features are cheap, derived entirely from the primary model's
own output plus light market context, so the EA can assemble them at
runtime with no extra ONNX call beyond the meta-model itself:

  [dir_p_short, dir_p_flat, dir_p_long,        # primary direction probs
   dir_conf,                                    # max prob - second prob
   q10, q50, q90,                               # quantile head
   regime_p (4),                                # regime head
   realized_vol, atr_norm]                      # market context  -> 13 dims

Output: P(act). The EA fires only when P(act) >= META_ACT_THRESHOLD.
Exports to ONNX via onnxmltools, same path as the XGBoost baseline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import (
    META_N_ESTIMATORS, META_MAX_DEPTH, META_LR, SEED,
)

log = logging.getLogger(__name__)

# Meta-feature vector width — must match _meta_features() and AurumAgent.mqh.
META_FEATURE_DIM = 13


def build_meta_features(primary_out: np.ndarray,
                        realized_vol: np.ndarray,
                        atr_norm: np.ndarray) -> np.ndarray:
    """
    Assemble the meta-feature matrix from primary-model outputs.

    primary_out : float[N, OUTPUT_DIM]  raw AurumNet output (post-softmax on
                  the dir/regime slices — i.e. AurumExportWrapper output).
    realized_vol, atr_norm : float[N]   market context at each sample.
    """
    from aurum.aurum_config import SLICE_DIR, SLICE_QUANT, SLICE_REGIME
    d0, d1 = SLICE_DIR
    q0, q1 = SLICE_QUANT
    r0, r1 = SLICE_REGIME
    dir_p = primary_out[:, d0:d1]
    quant = primary_out[:, q0:q1]
    regime = primary_out[:, r0:r1]
    sorted_p = np.sort(dir_p, axis=1)
    dir_conf = (sorted_p[:, -1] - sorted_p[:, -2]).reshape(-1, 1)
    feats = np.concatenate([
        dir_p, dir_conf, quant, regime,
        realized_vol.reshape(-1, 1), atr_norm.reshape(-1, 1),
    ], axis=1).astype(np.float32)
    assert feats.shape[1] == META_FEATURE_DIM, (feats.shape, META_FEATURE_DIM)
    return feats


def build_meta_target(primary_dir: np.ndarray, y_dir: np.ndarray,
                       y_ret: np.ndarray) -> np.ndarray:
    """
    Binary meta-target: 1 if acting on the primary direction would have
    been profitable, else 0.

    A primary 'flat' call is never acted on -> target 0.
    A long call is profitable if y_ret > 0; a short if y_ret < 0.
    """
    target = np.zeros(len(primary_dir), dtype=np.int64)
    long_ok = (primary_dir == 2) & (y_ret > 0)
    short_ok = (primary_dir == 0) & (y_ret < 0)
    target[long_ok | short_ok] = 1
    return target


class MetaLabelGate:
    """XGBoost binary classifier — P(act)."""

    def __init__(self, use_gpu: bool = False):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=META_N_ESTIMATORS,
            max_depth=META_MAX_DEPTH,
            learning_rate=META_LR,
            objective="binary:logistic",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=SEED,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            n_jobs=0,
        )
        self._fitted = False

    def fit(self, feats: np.ndarray, target: np.ndarray) -> "MetaLabelGate":
        self.model.fit(feats, target)
        self._fitted = True
        return self

    def predict_act_prob(self, feats: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(feats)[:, 1]

    def save(self, path: Path) -> None:
        """Persist the fitted booster so the export step can re-run it."""
        if not self._fitted:
            raise RuntimeError("fit() before save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> "MetaLabelGate":
        self.model.load_model(str(path))
        self._fitted = True
        return self

    def export_onnx(self, path: Path) -> bool:
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError:
            log.warning("onnxmltools missing — skipping meta-gate ONNX export")
            return False
        if not self._fitted:
            raise RuntimeError("fit() before export_onnx()")
        onnx_model = convert_xgboost(
            self.model,
            initial_types=[("input",
                            FloatTensorType([1, META_FEATURE_DIM]))],
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        log.info("[meta] ONNX -> %s", path)
        return True


def create_meta_gate(use_gpu: bool = False) -> MetaLabelGate:
    return MetaLabelGate(use_gpu=use_gpu)
