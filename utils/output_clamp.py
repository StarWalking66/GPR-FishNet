import torch


def apply_output_clamp(pred: torch.Tensor, min_value: float = 0.0, max_value: float = 1.0) -> torch.Tensor:
    return torch.clamp(pred, min_value, max_value)
