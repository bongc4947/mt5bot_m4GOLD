"""
aurum/backbone.py — L1 patch-transformer encoder (PatchTST-style).

One encoder per timeframe. Following Nie et al., ICLR 2023
("A Time Series is Worth 64 Words"):

  * Patching — the channel series is split into overlapping patches of
    PATCH_LEN bars. A patch is the transformer "token". This shortens the
    attention sequence and gives each token local semantic content.
  * Channel independence — every channel is encoded by the SAME weights,
    independently. Regularises hard; the biggest win in the PatchTST
    ablation.

Patch embedding is a Conv1d with kernel=PATCH_LEN, stride=PATCH_STRIDE,
applied to a [B*C, 1, L] reshape so the weights are shared across
channels. Conv1d exports to ONNX cleanly — `Tensor.unfold` does NOT
(its dynamic last-dim breaks ONNX shape inference), which is why the
deployed path never uses unfold. `raw_patches()` (unfold-based) exists
only for self-supervised reconstruction targets, which run in PyTorch
and are never exported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import (
    N_CHANNELS, PATCH_LEN, PATCH_STRIDE, D_MODEL, DEPTH, N_HEADS,
    FFN_DIM, DROPOUT,
)


def n_patches(seq_len: int) -> int:
    """How many patches a series of `seq_len` bars yields."""
    return max(1, (seq_len - PATCH_LEN) // PATCH_STRIDE + 1)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block — all ops ONNX-exportable."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS,
                                          dropout=DROPOUT, batch_first=True)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_DIM), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(FFN_DIM, D_MODEL),
        )
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class PatchEncoder(nn.Module):
    """
    Channel-independent patch-transformer for ONE timeframe.

    Input  : [B, L, C]   (L bars, C channels)
    Output : [B, D_MODEL] pooled embedding;
             optionally [B, C*nP, D_MODEL] per-patch tokens.
    """

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.n_patch = n_patches(seq_len)
        # Conv1d patch embedding — channel-shared via [B*C,1,L] reshape.
        self.patch_conv = nn.Conv1d(1, D_MODEL, kernel_size=PATCH_LEN,
                                    stride=PATCH_STRIDE)
        self.pos = nn.Parameter(
            torch.randn(1, self.n_patch * N_CHANNELS, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList(_TransformerBlock() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(D_MODEL)

    # -- patch helpers -----------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, C] -> patch tokens [B, C*nP, D_MODEL] (no pos added)."""
        B, L, C = x.shape
        xc = x.transpose(1, 2).reshape(B * C, 1, L)        # [B*C, 1, L]
        emb = self.patch_conv(xc)                          # [B*C, D, nP]
        nP = emb.shape[-1]
        emb = emb.reshape(B, C, D_MODEL, nP)
        emb = emb.permute(0, 1, 3, 2).reshape(B, C * nP, D_MODEL)
        return emb

    def raw_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, L, C] -> raw patch values [B, C*nP, PATCH_LEN].
        Uses unfold — SSL-only (never exported to ONNX).
        """
        B, L, C = x.shape
        xc = x.transpose(1, 2)                              # [B, C, L]
        p = xc.unfold(-1, PATCH_LEN, PATCH_STRIDE)          # [B,C,nP,P]
        nP = p.shape[2]
        return p.reshape(B, C * nP, PATCH_LEN)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run pos + transformer blocks + norm on prepared tokens."""
        tokens = tokens + self.pos[:, :tokens.shape[1], :]
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        tok = self.encode_tokens(self.embed(x))             # [B, C*nP, D]
        pooled = tok.mean(dim=1)                            # [B, D]
        if return_tokens:
            return pooled, tok
        return pooled


def create_encoder(seq_len: int) -> PatchEncoder:
    return PatchEncoder(seq_len)


if __name__ == "__main__":
    enc = create_encoder(128)
    x = torch.randn(4, 128, N_CHANNELS)
    p, t = enc(x, return_tokens=True)
    print("pooled:", p.shape, "tokens:", t.shape,
          "raw_patches:", enc.raw_patches(x).shape)
