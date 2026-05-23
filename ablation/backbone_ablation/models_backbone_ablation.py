from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from main_model.gpr_fishnet import (
    AttentionResidualPredictor,
    ContextAwareRouter,
    MultiScalePersistence,
    STLSTMCell,
)


BACKBONE_VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "baseline": "STLSTM only with a plain output head.",
    "plus_arp": "STLSTM + ARP.",
    "plus_mssp": "STLSTM + MSSP.",
    "plus_arp_mssp": "STLSTM + ARP + MSSP with fixed two-branch fusion.",
    "full": "STLSTM + ARP + MSSP + CAR (Context-Aware Router), reused from the current main-model results.",
}

BACKBONE_VARIANTS: Tuple[str, ...] = tuple(BACKBONE_VARIANT_DESCRIPTIONS.keys())


class PlainOutputHead(nn.Module):
    def __init__(self, fusion_dim: int, hidden_dim: int):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(fusion_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_net(x)


class BackboneAblationNet(nn.Module):
    def __init__(
        self,
        variant: str = "baseline",
        in_chans: int = 8,
        hidden_dim: int = 64,
        img_size: Tuple[int, int] = (64, 96),
        num_layers: int = 2,
    ):
        super().__init__()
        if variant not in BACKBONE_VARIANTS:
            raise ValueError(f"Unknown backbone ablation variant: {variant}. Valid variants: {BACKBONE_VARIANTS}")

        self.variant = variant
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.h, self.w = img_size
        self.apply_output_relu = True

        self.cell_list = nn.ModuleList(
            [
                STLSTMCell(in_chans if i == 0 else hidden_dim, hidden_dim, self.h, self.w, 3, 1)
                for i in range(num_layers)
            ]
        )
        self.fusion_dim = hidden_dim * 2 if num_layers > 1 else hidden_dim

        self.output_head: nn.Module | None = None
        self.predictor: nn.Module | None = None
        self.persister: nn.Module | None = None
        self.router: nn.Module | None = None

        if variant == "baseline":
            self.output_head = PlainOutputHead(self.fusion_dim, hidden_dim)
        elif variant == "plus_arp":
            self.predictor = AttentionResidualPredictor(self.fusion_dim, hidden_dim)
        elif variant == "plus_mssp":
            self.persister = MultiScalePersistence(1 + self.fusion_dim, hidden_dim)
        elif variant == "plus_arp_mssp":
            self.predictor = AttentionResidualPredictor(self.fusion_dim, hidden_dim)
            self.persister = MultiScalePersistence(1 + self.fusion_dim, hidden_dim)
        elif variant == "full":
            self.predictor = AttentionResidualPredictor(self.fusion_dim, hidden_dim)
            self.persister = MultiScalePersistence(1 + self.fusion_dim, hidden_dim)
            self.router = ContextAwareRouter(self.fusion_dim + 1)
        else:
            raise RuntimeError(f"Unsupported variant setup: {variant}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        device = x.device

        h_t = [torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=device) for _ in range(self.num_layers)]
        memory = torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=device)

        for t in range(seq_len):
            for i in range(self.num_layers):
                layer_input = x[:, t] if i == 0 else h_t[i - 1]
                h_t[i], c_t[i], memory = self.cell_list[i](layer_input, h_t[i], c_t[i], memory)

        return torch.cat([h_t[0], h_t[-1]], dim=1) if self.num_layers > 1 else h_t[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fusion_h = self._encode(x)
        last_obs = x[:, -1, -1:]

        if self.variant == "baseline":
            assert self.output_head is not None
            pred_final = self.output_head(fusion_h)
        elif self.variant == "plus_arp":
            assert self.predictor is not None
            _, pred_final = self.predictor(fusion_h)
        elif self.variant == "plus_mssp":
            assert self.persister is not None
            pred_final = self.persister(fusion_h, last_obs)
        elif self.variant == "plus_arp_mssp":
            assert self.predictor is not None
            assert self.persister is not None
            h_attn, pred_gen = self.predictor(fusion_h)
            pred_pers = self.persister(h_attn, last_obs)
            pred_final = 0.5 * pred_gen + 0.5 * pred_pers
        elif self.variant == "full":
            assert self.predictor is not None
            assert self.persister is not None
            assert self.router is not None
            h_attn, pred_gen = self.predictor(fusion_h)
            pred_pers = self.persister(h_attn, last_obs)
            route_ctx = torch.cat([h_attn, last_obs], dim=1)
            pred_final = self.router(route_ctx, pred_gen, pred_pers)
        else:
            raise RuntimeError(f"Unsupported variant in forward: {self.variant}")

        if self.apply_output_relu:
            pred_final = F.relu(pred_final)
        return pred_final.unsqueeze(1)
