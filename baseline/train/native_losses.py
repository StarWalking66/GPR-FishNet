from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def native_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def native_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def predrnn_v2_native_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    decouple_loss: Optional[torch.Tensor] = None,
    decouple_beta: float = 0.1,
) -> torch.Tensor:
    loss = native_mse_loss(pred, target)
    if decouple_loss is not None:
        loss = loss + decouple_beta * decouple_loss
    return loss


class _SpectralEnergyLoss(nn.Module):
    def _prepare_fft(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_2d = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
        target_2d = target.reshape(-1, target.shape[-2], target.shape[-1])
        pred_fft = torch.fft.fft2(pred_2d.float())
        target_fft = torch.fft.fft2(target_2d.float())
        return pred_fft, target_fft


class FrequencyAmplitudeLoss(_SpectralEnergyLoss):
    """
    Adapted from the official FACL implementation used by exPreCast.
    """

    def __init__(self, alpha: float = 1.0, ave_spectrum: bool = False, log_matrix: bool = False):
        super().__init__()
        self.alpha = alpha
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft, target_fft = self._prepare_fft(pred, target)
        pred_amp = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2 + 1e-8)
        target_amp = torch.sqrt(target_fft.real ** 2 + target_fft.imag ** 2 + 1e-8)

        if self.ave_spectrum:
            pred_amp = pred_amp.mean(dim=0, keepdim=True)
            target_amp = target_amp.mean(dim=0, keepdim=True)

        matrix_tmp = (pred_amp - target_amp) ** 2
        if self.log_matrix:
            matrix_tmp = torch.log(matrix_tmp + 1.0)

        return matrix_tmp.mean()


class FrequencyCorrelationLoss(_SpectralEnergyLoss):
    """
    Adapted from the official FACL implementation used by exPreCast.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft, target_fft = self._prepare_fft(pred, target)

        pred_real = pred_fft.real.flatten(1)
        pred_imag = pred_fft.imag.flatten(1)
        target_real = target_fft.real.flatten(1)
        target_imag = target_fft.imag.flatten(1)

        pred_real = pred_real - pred_real.mean(dim=1, keepdim=True)
        pred_imag = pred_imag - pred_imag.mean(dim=1, keepdim=True)
        target_real = target_real - target_real.mean(dim=1, keepdim=True)
        target_imag = target_imag - target_imag.mean(dim=1, keepdim=True)

        pred_norm = torch.sqrt(pred_real.pow(2).sum(dim=1) + pred_imag.pow(2).sum(dim=1) + 1e-8)
        target_norm = torch.sqrt(target_real.pow(2).sum(dim=1) + target_imag.pow(2).sum(dim=1) + 1e-8)

        real_term = (pred_real * target_real).sum(dim=1)
        imag_term = (pred_imag * target_imag).sum(dim=1)
        corr = (real_term + imag_term) / (pred_norm * target_norm + 1e-8)
        return (1.0 - corr).mean()


@dataclass
class _FACLState:
    count: int = 0
    total_count: int = 1


class ExPreCastFACL(nn.Module):
    """
    Frequency-Aware Curriculum Learning objective used by the official exPreCast code.

    The official repository instantiates `FACL(args.n_steps)`.
    This implementation follows the published FACL scheduling logic while
    remaining lightweight enough for the current benchmark trainer.
    """

    def __init__(
        self,
        total_steps: int,
        *,
        micro_batch: int = 1,
        const_ratio: float = 0.4,
    ):
        super().__init__()
        self.fal = FrequencyAmplitudeLoss()
        self.fcl = FrequencyCorrelationLoss()
        self.micro_batch = max(1, micro_batch)
        self.const_ratio = float(const_ratio)
        self.state = _FACLState(count=0, total_count=max(1, math.ceil(total_steps / self.micro_batch)))

    def reset(self) -> None:
        self.state.count = 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.state.count += 1
        progress = min(self.state.count / max(self.state.total_count, 1), 1.0)
        if progress <= self.const_ratio:
            use_fcl = False
        else:
            phase = (progress - self.const_ratio) / max(1.0 - self.const_ratio, 1e-8)
            use_fcl = random.random() < phase

        return self.fcl(pred, target) if use_fcl else self.fal(pred, target)
