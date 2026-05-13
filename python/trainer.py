"""
trainer.py — Hardware-aware training loop for all HYDRA mk4 models.
Supports direction models (binary BCE) and execution models (multi-task).
"""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*",
                        category=UserWarning)

from config import (
    EPOCHS, RETRAIN_EPOCHS, PATIENCE, LR, WEIGHT_DECAY,
    FOCAL_GAMMA, FOCAL_ALPHA, MIXUP_ALPHA, MIN_SAMPLES_MIXUP,
    BATCH_SIZE, WORKERS, progress_json_path,
)
from hardware_detector import HardwareConfig, get as get_hw
from dataset import (
    DirectionDataset, ExecutionDataset, ModifyDataset,
    RandomWindowDirectionDataset,
    train_val_split, make_loader,
)


# ---------------------------------------------------------------------------
# Effective-batch sizing
# ---------------------------------------------------------------------------
def _effective_batch_size(dataset_size: int, hw_batch: int,
                          target_batches_per_epoch: int = 32,
                          floor: int = 1024) -> int:
    """
    Pick a batch size that gives ~target_batches_per_epoch gradient steps.

    The hardware-detector batch is sized to fill VRAM. For small datasets
    (e.g. 300 K samples after exclude_flat) that ratio collapses to ~1
    batch / epoch and Adam never accumulates meaningful momentum. Cap the
    batch by dataset_size / target_batches_per_epoch so the optimizer
    actually sees enough updates.

    Args:
        dataset_size: # training samples (post split, post exclude_flat).
        hw_batch    : the auto-detected VRAM-aware batch size.
        target_batches_per_epoch: aim for at least this many mini-batches
            per epoch. 32 is a balanced default — enough Adam steps to
            converge in 30-100 epochs without making each step too noisy.
        floor       : never go below this size (CUDA launch + memory
                      transfer overhead dominates below ~1024).

    Returns: the effective batch size, log-printed if it differs from hw_batch.
    """
    if dataset_size <= 0:
        return hw_batch
    by_size = max(floor, dataset_size // target_batches_per_epoch)
    eff    = min(hw_batch, by_size)
    if eff != hw_batch:
        log.info("Batch sized for dataset: VRAM-cap=%d -> effective=%d "
                 "(%d batches/epoch on %d samples)",
                 hw_batch, eff,
                 max(1, dataset_size // eff), dataset_size)
    return int(eff)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal BCE loss — down-weights easy examples, focuses on hard ones
# ---------------------------------------------------------------------------

class FocalBCELoss(nn.Module):
    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = nn.functional.binary_cross_entropy_with_logits(
                    logit, target.float(), reduction="none")
        prob = torch.sigmoid(logit)
        pt   = torch.where(target.float() == 1, prob, 1 - prob)
        at   = torch.where(target.float() == 1,
                           torch.full_like(prob, self.alpha),
                           torch.full_like(prob, 1 - self.alpha))
        return (at * (1 - pt).pow(self.gamma) * bce).mean()


class AsymmetricFocalBCELoss(nn.Module):
    """
    Focal BCE with dynamic class weighting based on per-class accuracy.
    Automatically increases weight for the class with lower accuracy.
    """
    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA,
                 long_bias: float = 1.5, update_freq: int = 100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.long_bias = long_bias
        self.update_freq = update_freq
        self.step_count = 0
        self.long_acc_ema = 0.5
        self.short_acc_ema = 0.5

    def update_class_accuracy(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update EMA of per-class accuracy."""
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            targets_f = targets.float()

            # Per-class accuracy
            long_mask = (targets_f == 1)
            short_mask = (targets_f == 0)

            if long_mask.sum() > 0:
                long_acc = (preds[long_mask] == targets_f[long_mask]).float().mean().item()
                self.long_acc_ema = 0.95 * self.long_acc_ema + 0.05 * long_acc

            if short_mask.sum() > 0:
                short_acc = (preds[short_mask] == targets_f[short_mask]).float().mean().item()
                self.short_acc_ema = 0.95 * self.short_acc_ema + 0.05 * short_acc

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.step_count += 1

        # Update class accuracy periodically
        if self.step_count % self.update_freq == 0:
            self.update_class_accuracy(logit, target)

        # Compute dynamic alpha based on accuracy imbalance
        # If LONG accuracy is lower, increase its weight
        acc_ratio = self.short_acc_ema / (self.long_acc_ema + 1e-10)
        dynamic_long_bias = self.long_bias * min(acc_ratio, 2.0)  # Cap at 2x

        bce  = nn.functional.binary_cross_entropy_with_logits(
                    logit, target.float(), reduction="none")
        prob = torch.sigmoid(logit)
        pt   = torch.where(target.float() == 1, prob, 1 - prob)

        # Dynamic alpha: higher weight for underperforming class
        at = torch.where(target.float() == 1,
                        torch.full_like(prob, self.alpha * dynamic_long_bias),
                        torch.full_like(prob, 1 - self.alpha))

        return (at * (1 - pt).pow(self.gamma) * bce).mean()


# ---------------------------------------------------------------------------
# Execution model loss
# ---------------------------------------------------------------------------

def exec_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred / target: [B, 5]  [timing, sl_pips, tp_pips, vol_mult, session_gate]
    """
    p = pred.float();  t = target.float()
    timing_loss  = nn.functional.binary_cross_entropy(p[:, 0].clamp(0, 1), t[:, 0])
    sl_loss      = nn.functional.huber_loss(p[:, 1], t[:, 1], delta=5.0)
    tp_loss      = nn.functional.huber_loss(p[:, 2], t[:, 2], delta=5.0)
    vol_loss     = nn.functional.mse_loss(p[:, 3], t[:, 3])
    session_loss = nn.functional.binary_cross_entropy(p[:, 4].clamp(0, 1), t[:, 4])
    return 2.0 * session_loss + 1.5 * timing_loss + sl_loss + tp_loss + vol_loss


# ---------------------------------------------------------------------------
# Modification model loss
# ---------------------------------------------------------------------------

def modify_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred / target: [B, 3]  [move_sl_to_be, trail_sl_pips, close_now]
    """
    p = pred.float();  t = target.float()
    be_loss    = nn.functional.binary_cross_entropy(p[:, 0].clamp(0, 1), t[:, 0])
    trail_loss = nn.functional.huber_loss(p[:, 1], t[:, 1], delta=3.0)
    close_loss = nn.functional.binary_cross_entropy(p[:, 2].clamp(0, 1), t[:, 2])
    return 1.5 * be_loss + trail_loss + 1.5 * close_loss


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

def _binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    return float((preds == labels).float().mean().item())


def _write_progress(info: Dict[str, Any]):
    try:
        path = progress_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HardwareAwareTrainer
# ---------------------------------------------------------------------------

class HardwareAwareTrainer:
    """
    Unified trainer for direction, execution, and modification models.
    Detects hardware once on construction and adapts batch / AMP / workers.
    """

    def __init__(self, hw: Optional[HardwareConfig] = None):
        self.hw = hw or get_hw()
        self.device = torch.device(self.hw.device)

    # ------------------------------------------------------------------
    # Direction model training (binary BCE)
    # ------------------------------------------------------------------

    def train_direction(self, model: nn.Module,
                        features: np.ndarray,
                        labels: np.ndarray,
                        epochs: int = EPOCHS,
                        lr: float = LR,
                        patience: int = PATIENCE,
                        checkpoint_path: Optional[Path] = None,
                        symbol: str = "") -> Dict[str, Any]:
        """
        Train a direction model. Returns metrics dict.
        labels: float array +1=LONG, -1=SHORT, 0=FLAT

        Sampling: HYDRA_SAMPLER env var:
          unset / "chronological" — DirectionDataset, shuffle=True (default).
          "random-window" — RandomWindowDirectionDataset, with-replacement
              draws of HYDRA_SAMPLES_PER_EPOCH per epoch (default 100K).
              Use this when training on tick-bars (mk4.7 --source ticks)
              so the trainer sees the data as an unbounded random feed of
              decision moments rather than a fixed chronological sequence.
        """
        import os as _os
        from config import VAL_SPLIT, LABEL_FORWARD_BARS
        sampler_mode = _os.environ.get("HYDRA_SAMPLER", "chronological").lower()
        spe = int(_os.environ.get("HYDRA_SAMPLES_PER_EPOCH", "100000"))

        if sampler_mode == "random-window":
            # Slice features/labels chronologically *first*, then wrap only
            # the train side in the random-window dataset. The val side
            # remains a plain DirectionDataset so train and val draw from
            # *disjoint* time ranges — no leakage. Without this split, both
            # halves would randomly sample the full underlying pool.
            n = len(labels)
            n_val = max(1, int(n * VAL_SPLIT))
            gap   = LABEL_FORWARD_BARS
            n_tr  = max(1, n - n_val - gap)
            tr_feats = features[:n_tr]
            tr_lab   = labels[:n_tr]
            va_feats = features[n_tr + gap:]
            va_lab   = labels[n_tr + gap:]
            tr_set = RandomWindowDirectionDataset(tr_feats, tr_lab,
                                                   samples_per_epoch=spe,
                                                   mode="binary", exclude_flat=True)
            va_set = DirectionDataset(va_feats, va_lab,
                                       mode="binary", exclude_flat=True)
            log.info("Random-window split: train=%d (random feed, %d draws/epoch)  "
                     "gap=%d  val=%d (chronological)",
                     n_tr, spe, gap, len(va_lab))
        else:
            dataset = DirectionDataset(features, labels, mode="binary",
                                       exclude_flat=True)
            tr_set, va_set = train_val_split(dataset)
        bs = _effective_batch_size(len(tr_set), self.hw.batch_size)
        tr_loader = make_loader(tr_set, bs, shuffle=True,
                                workers=0, balanced=True)
        va_loader = make_loader(va_set, bs, shuffle=False, workers=0)

        return self._fit(
            model, tr_loader, va_loader,
            loss_fn=AsymmetricFocalBCELoss(),
            metric_fn=lambda logits, y: _binary_accuracy(logits, y),
            metric_name="val_acc",
            epochs=epochs, lr=lr, patience=patience,
            checkpoint_path=checkpoint_path,
            symbol=symbol,
            use_mixup=True,
        )

    # ------------------------------------------------------------------
    # Execution model training
    # ------------------------------------------------------------------

    def train_execution(self, model: nn.Module,
                        features: np.ndarray,
                        exec_labels: np.ndarray,
                        epochs: int = EPOCHS,
                        lr: float = LR,
                        patience: int = PATIENCE,
                        checkpoint_path: Optional[Path] = None,
                        symbol: str = "") -> Dict[str, Any]:
        dataset = ExecutionDataset(features, exec_labels)
        tr_set, va_set = train_val_split(dataset)
        bs = _effective_batch_size(len(tr_set), self.hw.batch_size)
        tr_loader = make_loader(tr_set, bs, shuffle=True, workers=0)
        va_loader = make_loader(va_set, bs, shuffle=False, workers=0)

        return self._fit(
            model, tr_loader, va_loader,
            loss_fn=None,   # use exec_loss
            metric_fn=None,
            metric_name="val_loss",
            epochs=epochs, lr=lr, patience=patience,
            checkpoint_path=checkpoint_path,
            symbol=symbol,
            loss_override=exec_loss,
        )

    # ------------------------------------------------------------------
    # Modification model training
    # ------------------------------------------------------------------

    def train_modify(self, model: nn.Module,
                     features: np.ndarray,
                     mod_labels: np.ndarray,
                     epochs: int = EPOCHS,
                     lr: float = LR,
                     patience: int = PATIENCE,
                     checkpoint_path: Optional[Path] = None,
                     symbol: str = "") -> Dict[str, Any]:
        dataset = ModifyDataset(features, mod_labels)
        tr_set, va_set = train_val_split(dataset)
        bs = _effective_batch_size(len(tr_set), self.hw.batch_size)
        tr_loader = make_loader(tr_set, bs, shuffle=True, workers=0)
        va_loader = make_loader(va_set, bs, shuffle=False, workers=0)

        return self._fit(
            model, tr_loader, va_loader,
            loss_fn=None,
            metric_fn=None,
            metric_name="val_loss",
            epochs=epochs, lr=lr, patience=patience,
            checkpoint_path=checkpoint_path,
            symbol=symbol,
            loss_override=modify_loss,
        )

    # ------------------------------------------------------------------
    # Core fit loop
    # ------------------------------------------------------------------

    def _fit(self, model: nn.Module,
             tr_loader, va_loader,
             loss_fn: Optional[nn.Module],
             metric_fn: Optional[Callable],
             metric_name: str,
             epochs: int,
             lr: float,
             patience: int,
             checkpoint_path: Optional[Path],
             symbol: str,
             loss_override: Optional[Callable] = None,
             use_mixup: bool = False) -> Dict[str, Any]:

        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = GradScaler("cuda", enabled=self.hw.amp)

        best_metric = -float("inf") if metric_name == "val_acc" else float("inf")
        best_state = None
        no_improve = 0
        history: Dict[str, list] = {"train_loss": [], metric_name: []}

        # Read the effective batch size from the loader (it may be smaller
        # than self.hw.batch_size if the dataset was small — see
        # _effective_batch_size). Reporting the actual loader batch is more
        # honest than the auto-detected VRAM cap.
        eff_bs = getattr(tr_loader, "batch_size", self.hw.batch_size) or self.hw.batch_size
        log.info("Training %s on %s  amp=%s  batch=%d  epochs=%d  batches/epoch=%d",
                 symbol, self.hw.device, self.hw.amp, eff_bs, epochs,
                 max(1, len(tr_loader.dataset) // eff_bs))

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            tr_loss = 0.0
            n_batches = 0

            for X_b, y_b in tr_loader:
                X_b = X_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)

                # Mixup augmentation (direction model only, skip when dataset is tiny)
                if use_mixup and MIXUP_ALPHA > 0 and len(tr_loader.dataset) >= MIN_SAMPLES_MIXUP:
                    lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                    idx = torch.randperm(X_b.size(0), device=self.device)
                    X_b = lam * X_b + (1 - lam) * X_b[idx]
                    y_b = lam * y_b.float() + (1 - lam) * y_b[idx].float()

                optimizer.zero_grad()

                with autocast("cuda", enabled=self.hw.amp):
                    out = model(X_b)
                if loss_override is not None:
                    loss = loss_override(out.float(), y_b.float())
                else:
                    with autocast("cuda", enabled=self.hw.amp):
                        # Binary: out shape [B,1], y_b shape [B]
                        loss = loss_fn(out.squeeze(1), y_b)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                tr_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_tr_loss = tr_loss / max(n_batches, 1)

            # Validation
            model.eval()
            va_loss = 0.0
            va_metric_sum = 0.0
            va_batches = 0
            with torch.no_grad():
                for X_b, y_b in va_loader:
                    X_b = X_b.to(self.device)
                    y_b = y_b.to(self.device)
                    out = model(X_b)
                    if loss_override is not None:
                        lv = loss_override(out.float(), y_b.float())
                    else:
                        lv = loss_fn(out.squeeze(1), y_b)
                    va_loss += lv.item()
                    if metric_fn is not None:
                        va_metric_sum += metric_fn(out.squeeze(1), y_b)
                    va_batches += 1

            avg_va_loss = va_loss / max(va_batches, 1)
            avg_va_metric = (va_metric_sum / max(va_batches, 1)
                             if metric_fn is not None else avg_va_loss)

            history["train_loss"].append(avg_tr_loss)
            history[metric_name].append(avg_va_metric)

            elapsed = time.time() - t0
            log.info("Epoch %3d/%d  tr=%.4f  va=%.4f  %s=%.4f  %.1fs",
                     epoch, epochs, avg_tr_loss, avg_va_loss,
                     metric_name, avg_va_metric, elapsed)

            # Early stopping
            improved = (avg_va_metric > best_metric if metric_name == "val_acc"
                        else avg_va_metric < best_metric)
            if improved:
                best_metric = avg_va_metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                if checkpoint_path:
                    torch.save(best_state, checkpoint_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info("Early stop at epoch %d  best_%s=%.4f",
                             epoch, metric_name, best_metric)
                    break

            _write_progress({
                "symbol": symbol,
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": avg_tr_loss,
                metric_name: float(avg_va_metric),
                "best": float(best_metric),
                "ts": time.time(),
            })

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        result = {
            metric_name: float(best_metric),
            "history": history,
            "trained_bars": len(tr_loader.dataset) + len(va_loader.dataset),
            "epochs_run": epoch,
        }
        log.info("Training done: best_%s=%.4f  bars=%d",
                 metric_name, best_metric, result["trained_bars"])
        return result
