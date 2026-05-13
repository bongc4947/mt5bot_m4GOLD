"""
validation.py — Phase 1 validation infrastructure.

One module, four pieces:

  1. backward_vol_regime  — replaces the forward-looking vol regime in
     labeler.py with a strictly causal version (last-N vs last-4N realised
     vol). Closes the train/live parity bug (PRODUCTION_GAPS B-section
     "backward-looking realised volatility").

  2. adversarial_validation_score — fits a tiny logistic classifier to
     distinguish train features from val features. Score > 0.7 means
     train and val are statistically different distributions; the val
     PnL number can't be trusted as a generalization estimate.

  3. fit_temperature — 1-D scalar that calibrates a model's logits on
     the val set so sigmoid output thresholds (e.g. 0.55) have
     consistent semantics across symbols (PRODUCTION_GAPS B2).

  4. OODAutoencoder — small encoder/decoder trained on training feature
     vectors. At inference, reconstruction error > P95(train_errors)
     means "input doesn't look like anything we've seen" → EA refuses
     to trade. The single biggest robustness lever for synthetic /
     regime-shift / black-swan data.

Phase-1 walk-forward gating is implemented in eval_harness.walk_forward_splits
already; the wiring into _train_agent.train_symbol happens in step
"validation.wf_gate" (also defined here to keep the surface unified).
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Backward-causal volatility regime
# ---------------------------------------------------------------------------

def backward_vol_regime(close: np.ndarray,
                         window_short: int = 20,
                         window_long: int = 80,
                         high_quantile: float = 0.66,
                         low_quantile:  float = 0.33,
                         ) -> np.ndarray:
    """
    Strictly causal LOW / MED / HIGH vol regime tag (codes 0/1/2).

    For each bar i:
        vol_short[i] = std(close[i-window_short+1 : i+1])
        vol_long[i]  = std(close[i-window_long+1  : i+1])
        ratio[i]     = vol_short[i] / vol_long[i]

    Then bin `ratio` against its own rolling distribution: high_quantile
    → HIGH (2), below low_quantile → LOW (0), else MED (1). All
    quantiles computed on a rolling window of the last `window_long * 5`
    bars so the labels adapt to regime shifts without leaking future
    info.

    Drop-in replacement for the forward-vol regime in labeler.py.
    """
    import pandas as pd
    n = len(close)
    if n < window_long + 1:
        return np.full(n, 1, dtype=np.int8)  # MED everywhere on tiny data

    log_ret = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=0.0)
    vs = pd.Series(log_ret).rolling(window_short, min_periods=2).std().to_numpy()
    vl = pd.Series(log_ret).rolling(window_long,  min_periods=2).std().to_numpy()
    ratio = np.where(vl > 1e-12, vs / vl, 1.0)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

    qhi = pd.Series(ratio).rolling(window_long * 5, min_periods=window_long).quantile(high_quantile).to_numpy()
    qlo = pd.Series(ratio).rolling(window_long * 5, min_periods=window_long).quantile(low_quantile ).to_numpy()
    qhi = np.nan_to_num(qhi, nan=1.0)
    qlo = np.nan_to_num(qlo, nan=1.0)

    regime = np.full(n, 1, dtype=np.int8)
    regime[ratio >= qhi] = 2  # HIGH
    regime[ratio <= qlo] = 0  # LOW
    return regime


# ---------------------------------------------------------------------------
# 2. Adversarial validation
# ---------------------------------------------------------------------------

def adversarial_validation_score(train_X: np.ndarray,
                                  val_X:   np.ndarray,
                                  max_samples: int = 50_000,
                                  ) -> float:
    """
    Fit a tiny logistic regression on (label=0 if from train, label=1
    if from val) and return its accuracy on a held-out 20% slice.

    Interpretation:
      ~0.50 — train and val are statistically indistinguishable: val
              PnL is a believable generalization estimate.
      ~0.70 — distributions differ substantially: val period had a
              regime shift relative to train. Don't trust the val PnL.
      ~0.90 — distributions are very different: train ≠ val by
              construction. Either re-shuffle data or expect
              out-of-distribution failure modes in live.

    Uses sklearn if present; falls back to a tiny torch logistic head.
    """
    rng = np.random.default_rng(42)
    n_tr = min(len(train_X), max_samples // 2)
    n_va = min(len(val_X),   max_samples // 2)
    tr_idx = rng.choice(len(train_X), size=n_tr, replace=False)
    va_idx = rng.choice(len(val_X),   size=n_va, replace=False)
    X = np.concatenate([train_X[tr_idx], val_X[va_idx]], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n_tr), np.ones(n_va)]).astype(np.int64)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_split = int(len(X) * 0.8)
    X_tr, y_tr = X[:n_split], y[:n_split]
    X_va, y_va = X[n_split:], y[n_split:]
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, n_jobs=1).fit(X_tr, y_tr)
        return float(clf.score(X_va, y_va))
    except Exception:
        # Tiny torch fallback so we don't add a hard sklearn dependency
        import torch
        Xt = torch.from_numpy(X_tr); yt = torch.from_numpy(y_tr).float().unsqueeze(1)
        Xv = torch.from_numpy(X_va); yv = torch.from_numpy(y_va).float().unsqueeze(1)
        w = torch.zeros(X.shape[1], 1, requires_grad=True)
        b = torch.zeros(1,         requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt)
            loss.backward(); opt.step()
        with torch.no_grad():
            pred = (torch.sigmoid(Xv @ w + b) > 0.5).float()
            return float((pred == yv).float().mean().item())


# ---------------------------------------------------------------------------
# 3. Temperature scaling (calibration)
# ---------------------------------------------------------------------------

def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                     max_iter: int = 200) -> float:
    """
    Fit a single scalar T > 0 such that sigmoid(logit / T) is
    well-calibrated against `labels` on the val set. Closes the
    "CONF_THRESHOLD = 0.55 fires n=0" issue — after dividing logits by
    the fitted T, prob > 0.55 has consistent semantics across symbols.

    Uses NLL minimization. Bounded to [0.5, 5.0] so a degenerate val
    set can't push T to extreme values.
    """
    import torch
    z = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    # Parameterize T = exp(s) so it stays > 0 without bounds.
    s = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([s], lr=0.5, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(s).clamp(0.5, 5.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / T, y)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(torch.exp(s).clamp(0.5, 5.0).item())
    return T


# ---------------------------------------------------------------------------
# 4. OOD autoencoder
# ---------------------------------------------------------------------------

class OODAutoencoder:
    """
    Tiny MLP autoencoder for out-of-distribution detection on tabular
    feature vectors. Trained on the *training-set* feature distribution;
    at inference, reconstruction MSE > p95(training-MSE) flags the input
    as OOD and the EA refuses to trade.

    Default architecture: F → 64 → 16 → 64 → F, ReLU, ~5K params for
    F=200. Cheap to export to ONNX and run on CPU per-bar.

    Public API:
        ood = OODAutoencoder(input_dim=200)
        ood.fit(train_features, epochs=20)
        threshold = ood.calibrate(val_features, percentile=95)
        is_ood = ood.score(live_features) > threshold
        ood.export_onnx(path)
    """

    def __init__(self, input_dim: int, hidden: int = 64, latent: int = 16):
        import torch
        import torch.nn as nn
        self.input_dim = int(input_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,    latent), nn.ReLU(),
            nn.Linear(latent,    hidden), nn.ReLU(),
            nn.Linear(hidden,    input_dim),
        )
        self.threshold_ = None  # set by calibrate()

    def fit(self, X: np.ndarray, epochs: int = 20, batch_size: int = 1024,
            lr: float = 1e-3, device: str = "cpu") -> None:
        import torch
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.model = self.model.to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = len(Xt)
        for ep in range(epochs):
            perm = torch.randperm(n)
            losses: list[float] = []
            for i in range(0, n, batch_size):
                batch = Xt[perm[i:i+batch_size]].to(device)
                opt.zero_grad()
                recon = self.model(batch)
                loss  = ((recon - batch) ** 2).mean()
                loss.backward(); opt.step()
                losses.append(float(loss.item()))
            log.info("OOD epoch %2d/%d  recon_mse=%.5f", ep + 1, epochs,
                     float(np.mean(losses)))

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            recon = self.model(Xt)
            mse = ((recon - Xt) ** 2).mean(dim=1).cpu().numpy()
        return mse

    def calibrate(self, X_val: np.ndarray, percentile: float = 95.0) -> float:
        scores = self.score(X_val)
        self.threshold_ = float(np.percentile(scores, percentile))
        return self.threshold_

    def export_onnx(self, path: str) -> None:
        import torch
        self.model.eval()
        dummy = torch.zeros(1, self.input_dim)
        torch.onnx.export(
            self.model, dummy, path,
            input_names=["features"], output_names=["reconstruction"],
            dynamic_axes={"features": {0: "batch"}, "reconstruction": {0: "batch"}},
            opset_version=17,
        )


# ---------------------------------------------------------------------------
# 5. Walk-forward gating helper
# ---------------------------------------------------------------------------

def walk_forward_pf_summary(features: np.ndarray, labels: np.ndarray,
                             forward_returns: np.ndarray,
                             pip: float,
                             *,
                             n_folds: int = 5,
                             gap: int = 20,
                             ) -> dict:
    """
    Run a deterministic 'always-long' baseline across walk-forward
    folds; return median PF + fold-consistency (fraction of folds with
    PF > 1.0). The model-side equivalent gets called from
    _train_agent after the model is trained.

    Used as a model-free reference: if "always long" already produces
    the same median PF as the trained model, the model has zero alpha.
    """
    from eval_harness import walk_forward_splits, compute_pnl_metrics
    fold_pfs: list[float] = []
    for tr_idx, va_idx in walk_forward_splits(len(labels), n_folds=n_folds, gap=gap):
        if len(va_idx) < 50:
            continue
        pred = np.ones(len(va_idx), dtype=np.int8)
        m = compute_pnl_metrics(pred=pred,
                                 forward_returns=forward_returns[va_idx],
                                 bar="M5",
                                 commission=pip, slippage=pip)
        if np.isfinite(m.profit_factor):
            fold_pfs.append(float(m.profit_factor))
    if not fold_pfs:
        return {"median_pf": float("nan"), "frac_profitable": 0.0, "n_folds": 0}
    median_pf = float(np.median(fold_pfs))
    frac_prof = float(np.mean([pf > 1.0 for pf in fold_pfs]))
    return {"median_pf": median_pf, "frac_profitable": frac_prof,
            "n_folds": len(fold_pfs), "fold_pfs": fold_pfs}
