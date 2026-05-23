from __future__ import annotations

from typing import Optional, Tuple

import torch


def validate_sequence_grid_input(
    x: torch.Tensor,
    *,
    model_name: str,
    seq_len: Optional[int] = None,
    in_chans: Optional[int] = None,
    img_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int, int]:
    if x.ndim != 5:
        raise ValueError(
            f"{model_name} expects input shaped [B, T, C, H, W], "
            f"but received ndim={x.ndim} with shape={tuple(x.shape)}."
        )

    batch_size, steps, channels, height, width = x.shape

    if seq_len is not None and steps != seq_len:
        raise ValueError(f"{model_name} expects seq_len={seq_len}, but received {steps}.")

    if in_chans is not None and channels != in_chans:
        raise ValueError(f"{model_name} expects in_chans={in_chans}, but received {channels}.")

    if img_size is not None and (height, width) != tuple(img_size):
        raise ValueError(
            f"{model_name} expects img_size={tuple(img_size)}, but received {(height, width)}."
        )

    return batch_size, steps, channels, height, width


def format_prediction_grid(pred: torch.Tensor, *, model_name: str) -> torch.Tensor:
    if pred.ndim != 4:
        raise ValueError(
            f"{model_name} must produce [B, T_pred, H, W] before formatting, "
            f"but received shape={tuple(pred.shape)}."
        )
    return pred.unsqueeze(2).contiguous()
