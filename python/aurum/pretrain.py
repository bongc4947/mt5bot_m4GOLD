"""
aurum/pretrain.py — L0 self-supervised pretraining.

Financial labels are scarce and noisy; unlabelled GOLD history is
abundant. Pretrain the per-timeframe patch encoders on two objectives:

  1. Masked-patch reconstruction (PatchTST self-supervised, Nie et al.
     2023). A random SSL_MASK_RATIO of patch tokens is replaced with a
     learned mask token; a lightweight linear decoder reconstructs the
     original patch values. Loss = MSE on the masked patches only.

  2. Contrastive view consistency (TS2Vec, Yue et al. 2022). Two augmented
     views of the same window (jitter + scaling) should map to nearby
     pooled embeddings; an NT-Xent loss pulls positives together and
     pushes negatives apart.

The trained encoders are saved per timeframe and later loaded frozen (then
unfrozen after a warmup) by the fine-tuning stage.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import (
    TIMEFRAMES, N_CHANNELS, PATCH_LEN, D_MODEL, SSL_MASK_RATIO, SSL_EPOCHS,
    SSL_LR, SSL_CONTRASTIVE_WEIGHT, SSL_TEMPERATURE, SEED,
)
from aurum.backbone import PatchEncoder, n_patches

log = logging.getLogger(__name__)


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Jitter + per-channel scaling — cheap, label-free TS augmentation."""
    jitter = torch.randn_like(x) * 0.02
    scale = 1.0 + torch.randn(x.shape[0], 1, x.shape[2],
                              device=x.device) * 0.05
    return x * scale + jitter


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    """Normalised temperature-scaled cross-entropy (SimCLR / TS2Vec)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)               # [2B, D]
    sim = z @ z.t() / temp                       # [2B, 2B]
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


class _SSLModule(nn.Module):
    """Encoder + mask token + linear patch decoder for one timeframe."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.encoder = PatchEncoder(seq_len)
        self.mask_token = nn.Parameter(torch.randn(D_MODEL) * 0.02)
        self.decoder = nn.Linear(D_MODEL, PATCH_LEN)
        self.n_patch = n_patches(seq_len)

    def forward(self, x: torch.Tensor):
        """Returns (recon_loss, pooled_embedding) for the masked-patch task."""
        B = x.shape[0]
        tok = self.encoder.embed(x)                 # [B, C*nP, D]
        target = self.encoder.raw_patches(x)        # [B, C*nP, PATCH_LEN]
        # Random patch mask.
        n_tok = tok.shape[1]
        n_mask = max(1, int(n_tok * SSL_MASK_RATIO))
        mask = torch.zeros(B, n_tok, dtype=torch.bool, device=x.device)
        for b in range(B):
            idx = torch.randperm(n_tok, device=x.device)[:n_mask]
            mask[b, idx] = True
        tok = torch.where(mask.unsqueeze(-1), self.mask_token, tok)
        tok = self.encoder.encode_tokens(tok)       # pos + blocks + norm
        recon = self.decoder(tok)                   # [B, C*nP, PATCH_LEN]
        loss = F.mse_loss(recon[mask], target[mask])
        pooled = tok.mean(dim=1)
        return loss, pooled


def pretrain_timeframe(tf: str, X: np.ndarray, *, epochs: int = SSL_EPOCHS,
                       device: str = "cpu", batch_size: int = 256,
                       out_dir: Path | None = None) -> Path:
    """
    Pretrain one timeframe's encoder on its unlabelled windows.

    X : float32[N, L_tf, C]   — windowed feature tensors from datamodule.
    Returns the path to the saved encoder state-dict.
    """
    torch.manual_seed(SEED)
    seq_len = TIMEFRAMES[tf]
    mod = _SSLModule(seq_len).to(device)
    opt = torch.optim.AdamW(mod.parameters(), lr=SSL_LR, weight_decay=1e-5)
    Xt = torch.from_numpy(X.astype(np.float32))
    n = len(Xt)
    log.info("[ssl:%s] %d windows  epochs=%d  device=%s", tf, n, epochs, device)

    for ep in range(epochs):
        mod.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            xb = Xt[perm[i:i + batch_size]].to(device)
            if len(xb) < 4:
                continue
            recon_loss, emb1 = mod(_augment(xb))
            _, emb2 = mod(_augment(xb))
            contrastive = _nt_xent(emb1, emb2, SSL_TEMPERATURE)
            loss = recon_loss + SSL_CONTRASTIVE_WEIGHT * contrastive
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step()
            tot += loss.item()       # .item() detaches — no grad-tensor warning
            nb += 1
        if ep % 5 == 0 or ep == epochs - 1:
            log.info("[ssl:%s] epoch %d/%d  loss=%.5f", tf, ep + 1, epochs,
                     tot / max(1, nb))

    out_dir = out_dir or (Path(__file__).parent.parent.parent / "onnx_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"aurum_ssl_encoder_{tf}.pt"
    torch.save(mod.encoder.state_dict(), path)
    log.info("[ssl:%s] encoder -> %s", tf, path.name)
    return path


def pretrain_all(dataset: dict, *, device: str = "cpu",
                 epochs: int = SSL_EPOCHS) -> dict[str, Path]:
    """Pretrain every timeframe encoder. `dataset` from datamodule.build_dataset."""
    t0 = time.time()
    paths = {}
    for tf in TIMEFRAMES:
        paths[tf] = pretrain_timeframe(tf, dataset["X"][tf],
                                       epochs=epochs, device=device)
    log.info("[ssl] all timeframes pretrained in %.0fs", time.time() - t0)
    return paths
