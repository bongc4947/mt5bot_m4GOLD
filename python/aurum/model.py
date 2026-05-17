"""
aurum/model.py — AurumNet assembly + ONNX export wrapper.

AurumNet wires the three per-timeframe patch encoders (L1) into the
cross-timeframe fusion (L2) and the multi-task heads (L3).

The deployed contract is a single flat input so MT5's ONNX runtime sees
trivial I/O:

  input  : float[B, FLAT_INPUT_DIM]   layout [M5(128×8), M15(64×8), H1(64×8)]
  output : float[B, OUTPUT_DIM]       layout [dir(3), quant(3), exec(3), regime(4)]

Per-channel normalisation (mean/std from the training set) is baked in as
a buffer, so the EA feeds raw channels and the model self-normalises —
no separate scaler to keep in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurum.aurum_config import (
    TIMEFRAMES, N_CHANNELS, FLAT_INPUT_DIM, OUTPUT_DIM,
)
from aurum.backbone import PatchEncoder
from aurum.fusion import CrossTimeframeFusion
from aurum.heads import MultiTaskHeads


class AurumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tf_order = list(TIMEFRAMES.keys())
        self.encoders = nn.ModuleDict(
            {tf: PatchEncoder(L) for tf, L in TIMEFRAMES.items()})
        self.fusion = CrossTimeframeFusion()
        self.heads = MultiTaskHeads()
        # Per-channel norm — overwritten by load_norm() after dataset build.
        self.register_buffer("ch_mean", torch.zeros(N_CHANNELS))
        self.register_buffer("ch_std", torch.ones(N_CHANNELS))

    # -- utilities ---------------------------------------------------------
    def load_norm(self, norm: dict) -> None:
        self.ch_mean.copy_(torch.tensor(norm["mean"], dtype=torch.float32))
        self.ch_std.copy_(torch.tensor(norm["std"], dtype=torch.float32))

    def load_encoders(self, paths: dict[str, Path]) -> None:
        """Load SSL-pretrained encoder weights per timeframe."""
        for tf, p in paths.items():
            self.encoders[tf].load_state_dict(torch.load(p, map_location="cpu"))

    def set_encoder_grad(self, requires_grad: bool) -> None:
        for enc in self.encoders.values():
            for p in enc.parameters():
                p.requires_grad = requires_grad

    # -- forward -----------------------------------------------------------
    def _split_flat(self, x_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flat [B, 2048] -> dict {tf: [B, L, C]} with per-channel norm."""
        B = x_flat.shape[0]
        out, off = {}, 0
        for tf in self.tf_order:
            L = TIMEFRAMES[tf]
            block = L * N_CHANNELS
            seg = x_flat[:, off:off + block].reshape(B, L, N_CHANNELS)
            off += block
            out[tf] = (seg - self.ch_mean) / self.ch_std
        return out

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Returns the concatenated [B, OUTPUT_DIM] deployed output."""
        windows = self._split_flat(x_flat)
        embeds = {tf: self.encoders[tf](windows[tf]) for tf in self.tf_order}
        fused = self.fusion(embeds)
        h = self.heads(fused)
        return torch.cat([h["direction"], h["quantile"],
                          h["exec"], h["regime"]], dim=1)

    def forward_dict(self, x_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Same as forward() but returns the per-task dict (for training)."""
        windows = self._split_flat(x_flat)
        embeds = {tf: self.encoders[tf](windows[tf]) for tf in self.tf_order}
        fused = self.fusion(embeds)
        return self.heads(fused)


class AurumExportWrapper(nn.Module):
    """
    Inference wrapper for ONNX export. Applies softmax to the direction
    and regime logits so the EA reads calibrated probabilities directly.
    """

    def __init__(self, net: AurumNet):
        super().__init__()
        self.net = net

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        from aurum.aurum_config import SLICE_DIR, SLICE_REGIME
        out = self.net(x_flat)
        out = out.clone()
        d0, d1 = SLICE_DIR
        r0, r1 = SLICE_REGIME
        out[:, d0:d1] = torch.softmax(out[:, d0:d1], dim=1)
        out[:, r0:r1] = torch.softmax(out[:, r0:r1], dim=1)
        return out


def create_aurum_net() -> AurumNet:
    return AurumNet()


if __name__ == "__main__":
    net = create_aurum_net()
    x = torch.randn(2, FLAT_INPUT_DIM)
    print("AurumNet out:", net(x).shape)            # [2, 13]
    print("export out:", AurumExportWrapper(net)(x).shape)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params:,}")
