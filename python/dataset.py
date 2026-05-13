"""
dataset.py — PyTorch Dataset + DataLoader for direction and execution models.
Handles balanced sampling across regimes and train/val split.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import VAL_SPLIT, WORKERS, LABEL_FORWARD_BARS, FEATURE_DIM_EXEC, FEATURE_DIM_MOD, CLASS_BALANCE_RATIO

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Direction Dataset
# ---------------------------------------------------------------------------

class DirectionDataset(Dataset):
    """
    Holds (features [N, FEAT_DIM], labels [N]) for direction model training.
    Labels are mapped: -1→0, 0→0.5 (flat — but excluded), 1→1.
    Flat labels (0) are included as negative class by default (binary: +1 vs rest).
    For 3-class: use mode='multiclass'.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 mode: str = "binary", exclude_flat: bool = True):
        """
        mode: 'binary' — labels to 0/1 (short+flat=0, long=1)
              'ternary' — labels as 0/1/2 (short, flat, long)
        exclude_flat: skip flat bars entirely (increases signal but reduces data)
        """
        assert len(features) == len(labels)
        self.mode = mode

        if exclude_flat:
            mask = labels != 0
            # For mmap arrays, fancy-index to get a concrete array for label/index ops
            labels   = np.asarray(labels)[mask]
            features = features[mask]   # may still be mmap — ok for index

        # If features is a float16 mmap, keep it as-is and convert per-item in
        # __getitem__ so we never materialise the full float32 array in RAM.
        # If it's already float32 (small in-memory array), convert now as before.
        if hasattr(features, 'dtype') and features.dtype == np.float16:
            self._mmap = True
            self.X = features           # disk-backed float16 memmap
        else:
            self._mmap = False
            self.X = torch.from_numpy(np.asarray(features).astype(np.float32))

        if mode == "binary":
            y = (labels > 0).astype(np.float32)
        else:
            y = (labels + 1).astype(np.int64)

        self.y = torch.from_numpy(y)
        self.labels_raw = np.asarray(labels).copy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._mmap:
            # Read only the requested rows from disk, convert to float32 on the fly
            x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        else:
            x = self.X[idx]
        return x, self.y[idx]

    def class_weights(self, max_ratio: float = CLASS_BALANCE_RATIO,
                      time_half_life: float = 0.0) -> torch.Tensor:
        """
        Soft-capped + optionally recency-weighted sampling weights for
        WeightedRandomSampler.

        max_ratio=1.5 → at most 60/40 majority/minority split after resampling.

        time_half_life > 0 → exponential recency bias applied on top of class
        weights.  Expressed as a fraction of dataset length: a sample at position
        (1 - half_life) gets half the weight of the last sample.

        Note: a previous LONG_BIAS=1.5 multiplier was removed in mk4.7. It
        rewarded models for leaning long, which silently boosted accuracy on
        bull-trending assets (BTC/SPX/Metals) while hurting genuinely
        sideways markets (FX). FX is the canary for real directional skill.
        """
        labels = self.y.numpy()
        n = len(labels)
        classes, counts = np.unique(labels, return_counts=True)

        # Scale down minority target so majority/minority ≤ max_ratio
        majority_count = counts.max()
        target_counts  = counts.copy().astype(np.float64)
        for i, c in enumerate(counts):
            if c < majority_count:
                target_counts[i] = max(c, majority_count / max_ratio)

        freq  = target_counts / target_counts.sum()
        w_map = {cls: 1.0 / (f + 1e-10) for cls, f in zip(classes, freq)}
        weights = np.array([w_map.get(int(l), 1.0) for l in labels], dtype=np.float64)

        if time_half_life > 0.0:
            # positions in [0, 1] — sample 0 is oldest, sample n-1 is most recent
            positions = np.arange(n, dtype=np.float64) / max(n - 1, 1)
            # exp decay: weight(pos) = 2^((pos-1)/half_life)
            # → weight(1.0) = 1.0, weight(1-half_life) = 0.5
            decay = np.exp(np.log(2.0) * (positions - 1.0) / time_half_life)
            weights = weights * decay

        return torch.from_numpy(weights.astype(np.float32))


class SequentialDirectionDataset(Dataset):
    """
    Returns (T, F) sliding-window samples ending at index idx instead of
    single (F,) snapshots. Each __getitem__ yields the last `window`
    timesteps of features as a tensor of shape (window, F) plus the
    label at the final timestep.

    Required by the scalp model architecture (Phase 2): scalp signals
    live in *how* features evolved over the last 30-60 ticks, not in
    their final snapshot. A single-timestamp MLP cannot learn this.

    exclude_flat: drops bars whose label is 0 (FLAT). Default True to
    match DirectionDataset semantics. The window itself can include
    flat-bar timesteps (they're features, not training targets) — only
    the *target* timestep gets the FLAT-skip filter.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 window: int = 64,
                 mode: str = "binary",
                 exclude_flat: bool = True):
        assert len(features) == len(labels), \
            f"features {len(features)} != labels {len(labels)}"
        assert window >= 2, "window must be >= 2"
        self.window = int(window)

        # Float32 features matrix (no copy if already correct dtype).
        if hasattr(features, "dtype") and features.dtype == np.float32:
            self.X = torch.from_numpy(np.ascontiguousarray(features))
        else:
            self.X = torch.from_numpy(np.asarray(features).astype(np.float32))

        labels_np = np.asarray(labels)

        # Build the index of *valid* target timesteps: those at idx >=
        # window-1 (so a full window fits) AND (if exclude_flat) those
        # whose label != 0. We store these target indices and __getitem__
        # maps idx -> target_indices[idx].
        valid_mask = np.ones(len(labels_np), dtype=bool)
        valid_mask[: self.window - 1] = False  # need full window
        if exclude_flat:
            valid_mask &= (labels_np != 0)
        self._target_idx = np.where(valid_mask)[0]
        if len(self._target_idx) == 0:
            raise ValueError(
                f"No valid samples after window={self.window} and "
                f"exclude_flat={exclude_flat} filters")

        # Binary y at every original timestep (we look it up by index).
        if mode == "binary":
            y = (labels_np > 0).astype(np.float32)
        else:
            y = (labels_np + 1).astype(np.int64)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self._target_idx)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self._target_idx[idx])
        # Slice the last `window` timesteps ending at t (inclusive).
        x = self.X[t - self.window + 1 : t + 1]   # shape (window, F)
        return x, self.y[t]


class RandomWindowDirectionDataset(DirectionDataset):
    """
    Same contract as DirectionDataset but __getitem__(idx) returns a
    *random* sample regardless of idx, and __len__ is whatever the user
    set as `samples_per_epoch`. This decouples "epoch length" from
    "dataset length" and makes training look like an unbounded random
    feed of decision moments — closer to how the live EA experiences
    the market (which never loops over a fixed dataset).

    Why bother when DataLoader already does shuffle=True?
    * shuffle=True traverses every sample exactly once per epoch.
    * RandomWindow draws WITH replacement: some samples may be revisited
      within an epoch, others skipped. Over multiple epochs the expected
      coverage is uniform but the trainer never sees the dataset as a
      fixed sequence.
    * Decouples len() from the dataset, so you can run e.g. 100K random
      draws/epoch on a 50K-sample tick-bar set, giving the optimizer
      ~3× more steps per epoch without re-loading data.
    * Pairs naturally with tick-bar training (mk4.7 --source ticks):
      tick-bars are information-uniform, the random sampler makes them
      temporally-uniform too.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 samples_per_epoch: int = 100_000,
                 mode: str = "binary", exclude_flat: bool = True,
                 seed: int = 42):
        super().__init__(features, labels, mode=mode, exclude_flat=exclude_flat)
        self.samples_per_epoch = int(samples_per_epoch)
        # Each worker gets its own RNG so num_workers > 0 doesn't yield
        # duplicate samples across workers in a batch. Re-seed lazily on
        # first __getitem__ inside a worker via worker_init_fn upstream.
        self._rng = np.random.default_rng(seed)
        self._real_n = super().__len__()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # idx is ignored; we draw a fresh random index each call.
        i = int(self._rng.integers(0, self._real_n))
        return super().__getitem__(i)


# ---------------------------------------------------------------------------
# Execution Dataset
# ---------------------------------------------------------------------------

class ExecutionDataset(Dataset):
    """
    Holds (exec_features [N, FEATURE_DIM_EXEC=1120], exec_labels [N, 5]).
    Labels: [timing, sl_pips, tp_pips, vol_mult, session_gate]
    Accepts float16 mmap arrays — converts to float32 per-item to avoid RAM spike.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        assert features.shape[1] == FEATURE_DIM_EXEC, \
            f"Expected exec features dim={FEATURE_DIM_EXEC}, got {features.shape[1]}"
        assert labels.shape[1] == 5

        if hasattr(features, 'dtype') and features.dtype == np.float16:
            self._mmap = True
            self.X = features           # disk-backed float16 memmap
        else:
            self._mmap = False
            self.X = torch.from_numpy(np.asarray(features).astype(np.float32))

        self.y = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._mmap:
            x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        else:
            x = self.X[idx]
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Modification Dataset
# ---------------------------------------------------------------------------

class ModifyDataset(Dataset):
    """
    Holds (mod_features [N, FEATURE_DIM_MOD=1008], mod_labels [N, 3]).
    Labels: [move_sl_to_be, trail_sl_pips, close_now]
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(features.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ModifyFromCacheDataset(Dataset):
    """
    Wraps a dir_feat float16 mmap (N, 1000) and lazily appends 8 position-context
    zeros per item — identical to the training-time mod_feat construction but
    without ever materializing the full (N, 1008) float32 array in RAM.
    Labels: mod_labels (N, 3).
    """

    def __init__(self, dir_feat: np.ndarray, labels: np.ndarray):
        assert labels.shape[1] == 3
        self.X = dir_feat   # float16 mmap (N, 1000) — disk-backed
        self.y = torch.from_numpy(labels.astype(np.float32))
        self._pad = torch.zeros(8, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        return torch.cat([x, self._pad]), self.y[idx]


# ---------------------------------------------------------------------------
# Train/Val split + DataLoader factory
# ---------------------------------------------------------------------------

def train_val_split(dataset: Dataset,
                    val_frac: float = VAL_SPLIT) -> Tuple[Dataset, Dataset]:
    """
    Chronological split with a label-forward gap between train and val.

    Without the gap, bars near the boundary have forward windows that reach
    into the val period — their labels are computed using val-period prices,
    which is look-ahead leakage.  Inserting a gap of LABEL_FORWARD_BARS (20)
    ensures no training label depends on any val-period bar.

    Layout: [--- train ---|-- gap(LABEL_FORWARD_BARS={LABEL_FORWARD_BARS}) --|--- val ---]
    """
    N     = len(dataset)
    n_val = max(1, int(N * val_frac))
    gap   = LABEL_FORWARD_BARS          # bars — matches labeling look-ahead (see config.py)
    n_tr  = max(1, N - n_val - gap)

    tr_idx = list(range(n_tr))
    va_idx = list(range(n_tr + gap, N))  # skip the gap

    tr = torch.utils.data.Subset(dataset, tr_idx)
    va = torch.utils.data.Subset(dataset, va_idx)
    log.info("Chronological split: train=%d  gap=%d  val=%d", n_tr, gap, len(va_idx))
    return tr, va


def make_loader(dataset: Dataset,
                batch_size: int,
                shuffle: bool = True,
                workers: int = WORKERS,
                balanced: bool = False,
                time_decay: float = 0.0) -> DataLoader:
    """
    time_decay: if > 0, applies exponential recency weighting via class_weights()
    (TIME_DECAY_HALFLIFE from config).  Only active when balanced=True and the
    underlying dataset is a DirectionDataset.
    """
    sampler = None
    if balanced:
        # Unwrap Subset → underlying DirectionDataset
        base = (dataset.dataset
                if isinstance(dataset, torch.utils.data.Subset) else dataset)
        if isinstance(base, DirectionDataset):
            all_weights = base.class_weights(time_half_life=time_decay)
            if isinstance(dataset, torch.utils.data.Subset):
                weights = all_weights[torch.tensor(dataset.indices)]
            else:
                weights = all_weights
            sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                            replacement=True)
            shuffle = False

    # drop_last=True when shuffling/training: a tail batch of size 1 makes
    # BatchNorm raise "Expected more than 1 value per channel". Only safe to
    # drop on training loaders (shuffle=True or balanced sampler) — never on
    # the val loader, which must score every example exactly once.
    is_training = shuffle or sampler is not None
    drop_last = bool(is_training and len(dataset) > batch_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=0,   # Windows subprocess safety: spawned DataLoader workers deadlock under `train.py all`
        pin_memory=True,
        drop_last=drop_last,
    )
