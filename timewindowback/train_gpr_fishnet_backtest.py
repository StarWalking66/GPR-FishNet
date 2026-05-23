import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from main_model.gpr_fishnet import GPRFishNet
from utils.multi_seed_experiment import DEFAULT_SEEDS, resolve_seed_save_dir
from utils.output_clamp import apply_output_clamp


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ST_FishNet_Features"
DEFAULT_MASK_PATH = DEFAULT_DATA_DIR / "all_vars_train_mask_intersection.npy"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "model_outcomes" / "checkpoints_gpr_fishnet_backtest"
DEFAULT_ENV_VARS = ["thetao", "chl", "uo", "vo", "so", "zos", "o2"]
MODEL_LABEL = "GPR-FishNet"
BEST_CHECKPOINT_FILENAME = "best_gpr_fishnet_backtest.pth"
ROLL_OUT_PREFIX = "rollout_12m"
ONE_STEP_PREFIX = "test_one_step"


@dataclass(frozen=True)
class WindowConfig:
    name: str
    test_year: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def rollout_start_index(self) -> int:
        return self.test_start

    @property
    def rollout_horizon(self) -> int:
        return self.test_end - self.test_start


WINDOW_CONFIGS: Dict[int, WindowConfig] = {
    2022: WindowConfig(
        name="window_2022",
        test_year=2022,
        train_start=0,
        train_end=96,
        val_start=96,
        val_end=120,
        test_start=120,
        test_end=132,
    ),
    2023: WindowConfig(
        name="window_2023",
        test_year=2023,
        train_start=0,
        train_end=108,
        val_start=108,
        val_end=132,
        test_start=132,
        test_end=144,
    ),
    2024: WindowConfig(
        name="window_2024",
        test_year=2024,
        train_start=0,
        train_end=120,
        val_start=120,
        val_end=144,
        test_start=144,
        test_end=156,
    ),
}


class TemporalWindowDataset(Dataset):
    def __init__(
        self,
        env_full: np.ndarray,
        ais_full: np.ndarray,
        split_start: int,
        split_end: int,
        seq_len: int,
        pred_len: int,
        split_name: str,
    ) -> None:
        super().__init__()

        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split_name: {split_name}")
        if seq_len <= 0 or pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive integers.")
        if split_start < 0 or split_end > len(env_full) or split_start >= split_end:
            raise ValueError(f"Invalid split range: [{split_start}, {split_end})")

        context_start = split_start if split_name == "train" else split_start - seq_len
        if context_start < 0:
            raise ValueError(
                f"{split_name} split requires at least seq_len={seq_len} history before start={split_start}."
            )

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.split_name = split_name
        self.split_start = int(split_start)
        self.split_end = int(split_end)
        self.context_start = int(context_start)
        self.env_segment = env_full[self.context_start : self.split_end].astype(np.float32, copy=False)
        self.ais_segment = ais_full[self.context_start : self.split_end].astype(np.float32, copy=False)

        self.total_samples = int(len(self.env_segment) - self.seq_len - self.pred_len + 1)
        if self.total_samples <= 0:
            raise ValueError(
                f"{split_name} split does not have enough frames for seq_len={seq_len}, "
                f"pred_len={pred_len}, frames={len(self.env_segment)}"
            )

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hist_env = self.env_segment[index : index + self.seq_len]
        hist_ais = self.ais_segment[index : index + self.seq_len]
        targets = self.ais_segment[index + self.seq_len : index + self.seq_len + self.pred_len]
        inputs = np.concatenate([hist_env, hist_ais], axis=1).astype(np.float32, copy=False)
        return torch.from_numpy(inputs), torch.from_numpy(targets.astype(np.float32, copy=False))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj: Dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def parse_int_list(raw: str, default: Sequence[int]) -> List[int]:
    if not raw.strip():
        return list(default)
    values: List[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(int(text))
    return values


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def threshold_tag(hotspot_threshold: float) -> str:
    return f"{hotspot_threshold:.10f}".rstrip("0").rstrip(".").replace(".", "p")


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def build_seed_run_row(
    seed: int,
    best_val_loss: float,
    best_epoch: int,
    one_step_metrics: Dict[str, float],
    rollout_metrics: Dict[str, float],
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "seed": int(seed),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
    }
    row.update(flatten_metrics(ONE_STEP_PREFIX, one_step_metrics))
    row.update(flatten_metrics(ROLL_OUT_PREFIX, rollout_metrics))
    return row


def expand_mask_like(ref_tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask = mask.float().to(ref_tensor.device)
    if mask.ndim == 2:
        while mask.ndim < ref_tensor.ndim:
            mask = mask.unsqueeze(0)
    elif mask.ndim == 3:
        if ref_tensor.ndim == 4:
            mask = mask.unsqueeze(1)
        elif ref_tensor.ndim == 5:
            mask = mask.unsqueeze(1).unsqueeze(1)
    return (mask > 0).float().expand_as(ref_tensor)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return torch.mean((pred - target) ** 2)
    expanded_mask = expand_mask_like(pred, mask)
    diff2 = ((pred - target) ** 2) * expanded_mask
    return diff2.sum() / expanded_mask.sum().clamp_min(1.0)


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return torch.mean(torch.abs(pred - target))
    expanded_mask = expand_mask_like(pred, mask)
    diff1 = torch.abs(pred - target) * expanded_mask
    return diff1.sum() / expanded_mask.sum().clamp_min(1.0)


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    mask_tensor: Optional[torch.Tensor] = None,
    hotspot_threshold: float = 0.2755,
) -> Dict[str, float]:
    preds_t = torch.from_numpy(preds).float()
    targets_t = torch.from_numpy(targets).float()

    valid_mask = torch.ones_like(preds_t) if mask_tensor is None else expand_mask_like(preds_t, mask_tensor.cpu())
    valid_mask_bool = valid_mask > 0.5

    p = preds_t[valid_mask_bool].numpy().astype(np.float64)
    t = targets_t[valid_mask_bool].numpy().astype(np.float64)
    if p.size == 0:
        return {"MAE": 0.0, "MSE": 0.0, "RMSE": 0.0, "R2": 0.0, "SSIM": 0.0, "CSI": 0.0, "F1": 0.0}

    diff = p - t
    mae = float(np.abs(diff).mean())
    mse = float((diff ** 2).mean())
    rmse = float(np.sqrt(mse))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2) + 1e-8
    r2 = float(1.0 - ss_res / ss_tot)

    mu_x, mu_y = np.mean(p), np.mean(t)
    var_x, var_y = np.var(p), np.var(t)
    cov_xy = np.mean((p - mu_x) * (t - mu_y))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = float(
        ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2))
        / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    )

    pred_bin = (p >= hotspot_threshold).astype(np.uint8)
    target_bin = (t >= hotspot_threshold).astype(np.uint8)
    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))

    csi = float(tp / (tp + fp + fn + 1e-8))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "SSIM": ssim, "CSI": csi, "F1": f1}


def evaluate(
    model: GPRFishNet,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    mask_tensor: Optional[torch.Tensor] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = apply_output_clamp(model(inputs))
            if loss_name.lower() == "mae":
                loss = masked_mae_loss(preds, targets, mask_tensor)
            else:
                loss = masked_mse_loss(preds, targets, mask_tensor)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def test_and_save_one_step(
    model: GPRFishNet,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    mask_tensor: Optional[torch.Tensor],
    hotspot_threshold: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_inputs: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"[{ONE_STEP_PREFIX.upper()}] Inference"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = apply_output_clamp(model(inputs))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0).astype(np.float32)
    targets_np = np.concatenate(all_targets, axis=0).astype(np.float32)
    inputs_np = np.concatenate(all_inputs, axis=0).astype(np.float32)

    np.save(save_dir / f"{ONE_STEP_PREFIX}_preds.npy", preds_np)
    np.save(save_dir / f"{ONE_STEP_PREFIX}_targets.npy", targets_np)
    np.save(save_dir / f"{ONE_STEP_PREFIX}_inputs.npy", inputs_np)

    metrics = compute_metrics(preds_np, targets_np, mask_tensor, hotspot_threshold)
    save_json(metrics, save_dir / f"{ONE_STEP_PREFIX}_metrics.json")
    return metrics, preds_np, targets_np


def save_example_predictions(save_dir: Path, preds: np.ndarray, targets: np.ndarray, prefix: str, num_examples: int = 5) -> None:
    num_examples = min(num_examples, preds.shape[0])
    example_dir = save_dir / f"{prefix}_samples"
    example_dir.mkdir(parents=True, exist_ok=True)
    for index in range(num_examples):
        np.save(example_dir / f"{prefix}_{index}_pred.npy", preds[index])
        np.save(example_dir / f"{prefix}_{index}_target.npy", targets[index])


def load_full_series(data_dir: Path, env_vars: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    env_list = [
        np.concatenate(
            [
                np.load(data_dir / f"{var}_train.npy"),
                np.load(data_dir / f"{var}_val.npy"),
                np.load(data_dir / f"{var}_test.npy"),
            ],
            axis=0,
        )
        for var in env_vars
    ]
    env_full = np.concatenate(env_list, axis=1).astype(np.float32)
    ais_full = np.concatenate(
        [
            np.load(data_dir / "ais_train.npy"),
            np.load(data_dir / "ais_val.npy"),
            np.load(data_dir / "ais_test.npy"),
        ],
        axis=0,
    ).astype(np.float32)
    return env_full, ais_full


def load_time_axis(data_dir: Path) -> np.ndarray:
    return np.asarray(np.load(data_dir / "ais_time_all.npy")).astype("datetime64[M]")


def format_month(value: np.datetime64) -> str:
    return str(np.datetime_as_string(value, unit="M"))


def format_range(time_axis: np.ndarray, start: int, end: int) -> str:
    return f"{format_month(time_axis[start])} to {format_month(time_axis[end - 1])}"


def rollout_predict_window(
    model: GPRFishNet,
    env_full: np.ndarray,
    ais_full: np.ndarray,
    device: torch.device,
    seq_len: int,
    horizon: int,
    start_idx: int,
    mask_np: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    hist_env = env_full[start_idx - seq_len : start_idx].copy()
    hist_ais = ais_full[start_idx - seq_len : start_idx].copy()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for step in range(horizon):
            current_input = np.concatenate([hist_env, hist_ais], axis=1)
            pred = apply_output_clamp(model(torch.from_numpy(current_input[None]).float().to(device)))
            pred_step = pred[0, 0].cpu().numpy()
            if mask_np is not None:
                pred_step[:, mask_np == 0] = 0.0
            preds.append(pred_step)
            targets.append(ais_full[start_idx + step])
            hist_env = np.concatenate([hist_env[1:], env_full[start_idx + step][None]], axis=0)
            hist_ais = np.concatenate([hist_ais[1:], pred_step[None]], axis=0)

    return np.stack(preds, axis=0).astype(np.float32), np.stack(targets, axis=0).astype(np.float32)


def save_rollout_results(
    save_dir: Path,
    preds: np.ndarray,
    targets: np.ndarray,
    mask_tensor: Optional[torch.Tensor],
    hotspot_threshold: float,
) -> Dict[str, float]:
    np.save(save_dir / f"{ROLL_OUT_PREFIX}_preds.npy", preds)
    np.save(save_dir / f"{ROLL_OUT_PREFIX}_targets.npy", targets)
    metrics = compute_metrics(preds, targets, mask_tensor, hotspot_threshold)
    save_json(metrics, save_dir / f"{ROLL_OUT_PREFIX}_metrics.json")
    return metrics


def _aggregate_numeric_rows(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    numeric_keys: List[str] = []
    for key, value in rows[0].items():
        if key == "seed":
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_keys.append(key)

    aggregated: Dict[str, float] = {}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values, ddof=0))
    return aggregated


def _aggregate_dict_list(rows: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not rows:
        return {}, {}
    keys = list(rows[0].keys())
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        means[key] = float(np.mean(values))
        stds[key] = float(np.std(values, ddof=0))
    return means, stds


def _load_mask(mask_path: Path, shape: Tuple[int, ...]) -> np.ndarray:
    if mask_path.exists():
        mask = np.load(mask_path).astype(np.float32)
    else:
        mask = np.ones(shape[-2:], dtype=np.float32)
    while mask.ndim < len(shape):
        mask = np.expand_dims(mask, axis=0)
    return np.broadcast_to(mask > 0.5, shape)


def _compute_threshold_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    mask_path: Path,
    hotspot_threshold: float,
) -> Dict[str, float]:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    valid_mask = _load_mask(mask_path, preds.shape)

    p = preds[valid_mask]
    t = targets[valid_mask]
    if p.size == 0:
        return {"CSI": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}

    pred_bin = (p >= hotspot_threshold).astype(np.uint8)
    target_bin = (t >= hotspot_threshold).astype(np.uint8)
    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))

    csi = float(tp / (tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-8))
    return {"CSI": csi, "F1": f1, "Precision": precision, "Recall": recall}


def _save_aggregated_split_outputs(
    base_save_dir: Path,
    split_name: str,
    seeds: Sequence[int],
    mask_path: Path,
    hotspot_threshold: float,
) -> None:
    seed_dirs = [Path(resolve_seed_save_dir(str(base_save_dir), seed, seeds)) for seed in seeds]
    preds_list: List[np.ndarray] = []
    targets_first: Optional[np.ndarray] = None
    inputs_first: Optional[np.ndarray] = None
    threshold_rows: List[Dict[str, float]] = []

    for run_dir in seed_dirs:
        preds_path = run_dir / f"{split_name}_preds.npy"
        targets_path = run_dir / f"{split_name}_targets.npy"
        if not (preds_path.exists() and targets_path.exists()):
            continue

        preds = np.load(preds_path).astype(np.float32)
        targets = np.load(targets_path).astype(np.float32)
        preds_list.append(preds)
        if targets_first is None:
            targets_first = targets

        inputs_path = run_dir / f"{split_name}_inputs.npy"
        if inputs_first is None and inputs_path.exists():
            inputs_first = np.load(inputs_path).astype(np.float32)

        threshold_rows.append(_compute_threshold_metrics(preds, targets, mask_path, hotspot_threshold))

    if not preds_list:
        return

    mean_preds = np.mean(np.stack(preds_list, axis=0), axis=0).astype(np.float32)
    np.save(base_save_dir / f"{split_name}_preds.npy", mean_preds)
    if targets_first is not None:
        np.save(base_save_dir / f"{split_name}_targets.npy", targets_first)
    if inputs_first is not None:
        np.save(base_save_dir / f"{split_name}_inputs.npy", inputs_first)

    threshold_mean, threshold_std = _aggregate_dict_list(threshold_rows)
    tag = threshold_tag(hotspot_threshold)
    save_json(
        {
            "aggregation": "mean_of_seed_threshold_metrics",
            "num_seeds": int(len(threshold_rows)),
            "seeds": list(seeds),
            "threshold_norm": float(hotspot_threshold),
            "metrics": threshold_mean,
            "std": threshold_std,
        },
        base_save_dir / f"{split_name}_metrics_threshold_{tag}.json",
    )


def save_additional_threshold_evaluations(
    base_save_dir: Path,
    seeds: Sequence[int],
    mask_path: Path,
    hotspot_thresholds: Sequence[float],
) -> List[float]:
    unique_thresholds: List[float] = []
    for value in hotspot_thresholds:
        if any(abs(float(value) - existing) < 1e-12 for existing in unique_thresholds):
            continue
        unique_thresholds.append(float(value))

    saved_thresholds: List[float] = []
    for hotspot_threshold in unique_thresholds:
        threshold_run_rows: List[Dict] = []
        tag = threshold_tag(hotspot_threshold)

        for seed in seeds:
            run_dir = Path(resolve_seed_save_dir(str(base_save_dir), seed, seeds))
            run_summary_path = run_dir / "run_summary.json"
            if not run_summary_path.exists():
                continue

            with run_summary_path.open("r", encoding="utf-8") as f:
                run_summary = json.load(f)

            one_step_preds_path = run_dir / f"{ONE_STEP_PREFIX}_preds.npy"
            one_step_targets_path = run_dir / f"{ONE_STEP_PREFIX}_targets.npy"
            rollout_preds_path = run_dir / f"{ROLL_OUT_PREFIX}_preds.npy"
            rollout_targets_path = run_dir / f"{ROLL_OUT_PREFIX}_targets.npy"
            if not all(
                path.exists()
                for path in (one_step_preds_path, one_step_targets_path, rollout_preds_path, rollout_targets_path)
            ):
                continue

            one_step_threshold_metrics = _compute_threshold_metrics(
                np.load(one_step_preds_path),
                np.load(one_step_targets_path),
                mask_path,
                hotspot_threshold,
            )
            rollout_threshold_metrics = _compute_threshold_metrics(
                np.load(rollout_preds_path),
                np.load(rollout_targets_path),
                mask_path,
                hotspot_threshold,
            )

            save_json(
                {
                    "threshold_norm": float(hotspot_threshold),
                    "seed": int(seed),
                    "metrics": one_step_threshold_metrics,
                },
                run_dir / f"{ONE_STEP_PREFIX}_metrics_threshold_{tag}.json",
            )
            save_json(
                {
                    "threshold_norm": float(hotspot_threshold),
                    "seed": int(seed),
                    "metrics": rollout_threshold_metrics,
                },
                run_dir / f"{ROLL_OUT_PREFIX}_metrics_threshold_{tag}.json",
            )

            threshold_run_row = dict(run_summary)
            threshold_run_row[f"{ONE_STEP_PREFIX}_CSI"] = float(one_step_threshold_metrics["CSI"])
            threshold_run_row[f"{ONE_STEP_PREFIX}_F1"] = float(one_step_threshold_metrics["F1"])
            threshold_run_row[f"{ONE_STEP_PREFIX}_Precision"] = float(one_step_threshold_metrics["Precision"])
            threshold_run_row[f"{ONE_STEP_PREFIX}_Recall"] = float(one_step_threshold_metrics["Recall"])
            threshold_run_row[f"{ROLL_OUT_PREFIX}_CSI"] = float(rollout_threshold_metrics["CSI"])
            threshold_run_row[f"{ROLL_OUT_PREFIX}_F1"] = float(rollout_threshold_metrics["F1"])
            threshold_run_row[f"{ROLL_OUT_PREFIX}_Precision"] = float(rollout_threshold_metrics["Precision"])
            threshold_run_row[f"{ROLL_OUT_PREFIX}_Recall"] = float(rollout_threshold_metrics["Recall"])
            threshold_run_row["threshold_norm"] = float(hotspot_threshold)

            save_json(threshold_run_row, run_dir / f"run_summary_threshold_{tag}.json")
            threshold_run_rows.append(threshold_run_row)

        if not threshold_run_rows:
            continue

        ordered_rows = sorted(threshold_run_rows, key=lambda row: int(row["seed"]))
        aggregated = _aggregate_numeric_rows(ordered_rows)
        write_csv(ordered_rows, base_save_dir / f"summary_runs_threshold_{tag}.csv")
        write_csv([aggregated], base_save_dir / f"summary_seed_mean_std_threshold_{tag}.csv")
        save_json(
            {
                "seeds": list(seeds),
                "threshold_norm": float(hotspot_threshold),
                "runs": ordered_rows,
                "aggregated": aggregated,
            },
            base_save_dir / f"summary_all_threshold_{tag}.json",
        )

        _save_aggregated_split_outputs(base_save_dir, ONE_STEP_PREFIX, seeds, mask_path, hotspot_threshold)
        _save_aggregated_split_outputs(base_save_dir, ROLL_OUT_PREFIX, seeds, mask_path, hotspot_threshold)
        saved_thresholds.append(float(hotspot_threshold))

    return saved_thresholds


def write_csv(rows: List[Dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finalize_window_experiment(
    base_save_dir: Path,
    seeds: Sequence[int],
    run_rows: List[Dict],
    mask_path: Path,
    hotspot_threshold: float,
    best_checkpoint_filename: str,
) -> None:
    base_save_dir.mkdir(parents=True, exist_ok=True)
    if not run_rows:
        return

    ordered_rows = sorted(run_rows, key=lambda row: int(row["seed"]))
    aggregated = _aggregate_numeric_rows(ordered_rows)
    write_csv(ordered_rows, base_save_dir / "summary_runs.csv")
    write_csv([aggregated], base_save_dir / "summary_seed_mean_std.csv")
    save_json(
        {
            "seeds": list(seeds),
            "runs": ordered_rows,
            "aggregated": aggregated,
        },
        base_save_dir / "summary_all.json",
    )

    test_rows = []
    rollout_rows = []
    metric_keys = ("MAE", "MSE", "RMSE", "R2", "SSIM", "CSI", "F1")
    for row in ordered_rows:
        test_rows.append(
            {key: float(row[f"{ONE_STEP_PREFIX}_{key}"]) for key in metric_keys if f"{ONE_STEP_PREFIX}_{key}" in row}
        )
        rollout_rows.append(
            {key: float(row[f"{ROLL_OUT_PREFIX}_{key}"]) for key in metric_keys if f"{ROLL_OUT_PREFIX}_{key}" in row}
        )

    test_mean, test_std = _aggregate_dict_list(test_rows)
    rollout_mean, rollout_std = _aggregate_dict_list(rollout_rows)
    save_json(
        {
            "aggregation": "mean_of_seed_metrics",
            "num_seeds": int(len(ordered_rows)),
            "seeds": list(seeds),
            "metrics": test_mean,
            "std": test_std,
        },
        base_save_dir / f"{ONE_STEP_PREFIX}_metrics.json",
    )
    save_json(
        {
            "aggregation": "mean_of_seed_metrics",
            "num_seeds": int(len(ordered_rows)),
            "seeds": list(seeds),
            "metrics": rollout_mean,
            "std": rollout_std,
        },
        base_save_dir / f"{ROLL_OUT_PREFIX}_metrics.json",
    )

    _save_aggregated_split_outputs(base_save_dir, ONE_STEP_PREFIX, seeds, mask_path, hotspot_threshold)
    _save_aggregated_split_outputs(base_save_dir, ROLL_OUT_PREFIX, seeds, mask_path, hotspot_threshold)

    best_row = min(ordered_rows, key=lambda row: float(row["best_val_loss"]))
    best_seed = int(best_row["seed"])
    best_seed_dir = Path(resolve_seed_save_dir(str(base_save_dir), best_seed, seeds))
    best_checkpoint_src = best_seed_dir / best_checkpoint_filename
    if best_checkpoint_src.exists():
        target_path = base_save_dir / best_checkpoint_filename
        target_path.write_bytes(best_checkpoint_src.read_bytes())

    save_json(
        {
            "selected_seed": best_seed,
            "selection_rule": "lowest_best_val_loss",
            "best_val_loss": float(best_row["best_val_loss"]),
            "best_epoch": int(best_row["best_epoch"]),
            "checkpoint_filename": best_checkpoint_filename,
        },
        base_save_dir / "selected_seed.json",
    )
    save_json(
        {
            "aggregation": "multi_seed_summary",
            "num_seeds": int(len(ordered_rows)),
            "seeds": list(seeds),
            "best_val_loss_mean": aggregated.get("best_val_loss_mean", float(best_row["best_val_loss"])),
            "best_val_loss_std": aggregated.get("best_val_loss_std", 0.0),
            "best_epoch_mean": aggregated.get("best_epoch_mean", float(best_row["best_epoch"])),
            "best_epoch_std": aggregated.get("best_epoch_std", 0.0),
            "selected_seed": best_seed,
        },
        base_save_dir / "training_history.json",
    )


def load_metric_payload(path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", payload)
    std_payload = payload.get("std", {})
    metrics_map = {key: float(value) for key, value in metrics.items()}
    std_map = {key: float(value) for key, value in std_payload.items()}
    return metrics_map, std_map


def build_window_summary_row(
    window: WindowConfig,
    time_axis: np.ndarray,
    window_dir: Path,
    hotspot_threshold: float,
    threshold_override: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    one_step_metrics, one_step_std = load_metric_payload(window_dir / f"{ONE_STEP_PREFIX}_metrics.json")
    rollout_metrics, rollout_std = load_metric_payload(window_dir / f"{ROLL_OUT_PREFIX}_metrics.json")

    if threshold_override is not None:
        tag = threshold_tag(threshold_override)
        one_step_threshold_metrics, one_step_threshold_std = load_metric_payload(
            window_dir / f"{ONE_STEP_PREFIX}_metrics_threshold_{tag}.json"
        )
        rollout_threshold_metrics, rollout_threshold_std = load_metric_payload(
            window_dir / f"{ROLL_OUT_PREFIX}_metrics_threshold_{tag}.json"
        )
    else:
        one_step_threshold_metrics = {key: one_step_metrics[key] for key in ("CSI", "F1") if key in one_step_metrics}
        rollout_threshold_metrics = {key: rollout_metrics[key] for key in ("CSI", "F1") if key in rollout_metrics}
        one_step_threshold_std = {key: one_step_std.get(key, 0.0) for key in ("CSI", "F1")}
        rollout_threshold_std = {key: rollout_std.get(key, 0.0) for key in ("CSI", "F1")}

    row: Dict[str, object] = {
        "window": window.name,
        "window_label": str(window.test_year),
        "test_year": int(window.test_year),
        "train_range": format_range(time_axis, window.train_start, window.train_end),
        "val_range": format_range(time_axis, window.val_start, window.val_end),
        "test_range": format_range(time_axis, window.test_start, window.test_end),
        "rollout_start_index": int(window.rollout_start_index),
        "rollout_horizon": int(window.rollout_horizon),
        "primary_hotspot_threshold": float(hotspot_threshold),
        "threshold_norm": float(threshold_override if threshold_override is not None else hotspot_threshold),
    }

    continuous_metric_keys = ("MAE", "MSE", "RMSE", "R2", "SSIM")
    threshold_metric_keys = ("CSI", "F1", "Precision", "Recall")
    for metric_name in continuous_metric_keys:
        row[f"{ONE_STEP_PREFIX}_{metric_name}"] = float(one_step_metrics[metric_name])
        row[f"{ONE_STEP_PREFIX}_{metric_name}_std"] = float(one_step_std.get(metric_name, 0.0))
        row[f"{ROLL_OUT_PREFIX}_{metric_name}"] = float(rollout_metrics[metric_name])
        row[f"{ROLL_OUT_PREFIX}_{metric_name}_std"] = float(rollout_std.get(metric_name, 0.0))

    for metric_name in threshold_metric_keys:
        if metric_name in one_step_threshold_metrics:
            row[f"{ONE_STEP_PREFIX}_{metric_name}"] = float(one_step_threshold_metrics[metric_name])
            row[f"{ONE_STEP_PREFIX}_{metric_name}_std"] = float(one_step_threshold_std.get(metric_name, 0.0))
        if metric_name in rollout_threshold_metrics:
            row[f"{ROLL_OUT_PREFIX}_{metric_name}"] = float(rollout_threshold_metrics[metric_name])
            row[f"{ROLL_OUT_PREFIX}_{metric_name}_std"] = float(rollout_threshold_std.get(metric_name, 0.0))
    return row


def aggregate_window_rows(rows: List[Dict[str, object]]) -> Dict[str, float]:
    aggregate: Dict[str, float] = {}
    if not rows:
        return aggregate
    numeric_keys: List[str] = []
    for key, value in rows[0].items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_keys.append(key)
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values, ddof=0))
    return aggregate


def save_root_summaries(
    save_dir: Path,
    windows: Sequence[WindowConfig],
    time_axis: np.ndarray,
    hotspot_threshold: float,
    extra_thresholds: Sequence[float],
) -> None:
    base_rows: List[Dict[str, object]] = []
    for window in windows:
        window_dir = save_dir / window.name
        if not (window_dir / f"{ONE_STEP_PREFIX}_metrics.json").exists():
            continue
        base_rows.append(build_window_summary_row(window, time_axis, window_dir, hotspot_threshold))

    base_rows = sorted(base_rows, key=lambda row: int(row["test_year"]))
    if base_rows:
        write_csv(base_rows, save_dir / "temporal_backtest_summary.csv")
        save_json(
            {
                "primary_hotspot_threshold": float(hotspot_threshold),
                "windows": [asdict(window) for window in windows],
                "rows": base_rows,
                "aggregated": aggregate_window_rows(base_rows),
            },
            save_dir / "temporal_backtest_summary.json",
        )

    for threshold in extra_thresholds:
        rows: List[Dict[str, object]] = []
        tag = threshold_tag(threshold)
        for window in windows:
            window_dir = save_dir / window.name
            one_step_threshold_path = window_dir / f"{ONE_STEP_PREFIX}_metrics_threshold_{tag}.json"
            rollout_threshold_path = window_dir / f"{ROLL_OUT_PREFIX}_metrics_threshold_{tag}.json"
            if not (one_step_threshold_path.exists() and rollout_threshold_path.exists()):
                continue
            rows.append(build_window_summary_row(window, time_axis, window_dir, hotspot_threshold, threshold_override=threshold))

        rows = sorted(rows, key=lambda row: int(row["test_year"]))
        if not rows:
            continue

        write_csv(rows, save_dir / f"temporal_backtest_summary_threshold_{tag}.csv")
        save_json(
            {
                "primary_hotspot_threshold": float(hotspot_threshold),
                "threshold_norm": float(threshold),
                "windows": [asdict(window) for window in windows],
                "rows": rows,
                "aggregated": aggregate_window_rows(rows),
            },
            save_dir / f"temporal_backtest_summary_threshold_{tag}.json",
        )


def build_window_metadata(
    window: WindowConfig,
    time_axis: np.ndarray,
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> Dict[str, object]:
    return {
        "window": asdict(window),
        "window_name": window.name,
        "test_year": int(window.test_year),
        "train_range": format_range(time_axis, window.train_start, window.train_end),
        "val_range": format_range(time_axis, window.val_start, window.val_end),
        "test_range": format_range(time_axis, window.test_start, window.test_end),
        "rollout_range": format_range(time_axis, window.test_start, window.test_end),
        "env_vars": list(args.env_vars),
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "batch_size": int(args.batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "train_loss": str(args.train_loss),
        "hotspot_threshold": float(args.hotspot_threshold),
        "extra_hotspot_thresholds": list(args.extra_hotspot_thresholds),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "model_label": MODEL_LABEL,
        "model_signature": f"{MODEL_LABEL}: STLSTM+ARP+MSSP+CAR(ContextAwareRouter)+ReLU",
        "seeds": list(seeds),
    }


def train_single_window(
    window: WindowConfig,
    env_full: np.ndarray,
    ais_full: np.ndarray,
    time_axis: np.ndarray,
    args: argparse.Namespace,
    mask_tensor: Optional[torch.Tensor],
    device: torch.device,
    seeds: Sequence[int],
) -> None:
    window_save_dir = args.save_dir / window.name
    window_save_dir.mkdir(parents=True, exist_ok=True)
    save_json(build_window_metadata(window, time_axis, args, seeds), window_save_dir / "experiment_config.json")

    train_dataset = TemporalWindowDataset(
        env_full,
        ais_full,
        split_start=window.train_start,
        split_end=window.train_end,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split_name="train",
    )
    val_dataset = TemporalWindowDataset(
        env_full,
        ais_full,
        split_start=window.val_start,
        split_end=window.val_end,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split_name="val",
    )
    test_dataset = TemporalWindowDataset(
        env_full,
        ais_full,
        split_start=window.test_start,
        split_end=window.test_end,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split_name="test",
    )

    run_rows: List[Dict] = []
    use_amp = device.type == "cuda"
    mask_np = mask_tensor.detach().cpu().numpy() if mask_tensor is not None else None

    for seed in seeds:
        set_seed(seed)
        run_save_dir = Path(resolve_seed_save_dir(str(window_save_dir), seed, seeds))
        run_save_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        model = GPRFishNet(
            in_chans=len(args.env_vars) + 1,
            hidden_dim=args.hidden_dim,
            img_size=(64, 96),
            num_layers=args.num_layers,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_epochs = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_model_path = run_save_dir / BEST_CHECKPOINT_FILENAME

        print(f"\n[{window.name}] Starting {MODEL_LABEL} backtest with seed={seed}...")
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_loader, desc=f"{window.name} | Seed {seed} | Epoch {epoch + 1}/{args.epochs} [Train]")

            for batch_index, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                targets = targets.to(device)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    preds = model(inputs)
                    if args.train_loss.lower() == "mae":
                        loss = masked_mae_loss(preds, targets, mask=mask_tensor)
                    else:
                        loss = masked_mse_loss(preds, targets, mask=mask_tensor)
                    loss = loss / args.accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if ((batch_index + 1) % args.accumulation_steps == 0) or ((batch_index + 1) == len(train_loader)):
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                real_loss = loss.item() * args.accumulation_steps
                running_loss += real_loss
                pbar.set_postfix({"loss": f"{real_loss:.6f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            avg_train_loss = running_loss / max(len(train_loader), 1)
            avg_val_loss = evaluate(model, val_loader, device, args.train_loss, mask_tensor)
            scheduler.step(avg_val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

            history["train_loss"].append(float(avg_train_loss))
            history["val_loss"].append(float(avg_val_loss))
            history["lr"].append(current_lr)

            print(
                f"{window.name} | Seed {seed} | Epoch {epoch + 1:03d} | "
                f"Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f} | LR = {current_lr:.2e}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"[{window.name}] Best model updated: {best_model_path}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= 20:
                    print(f"[{window.name}] Early stopping triggered at epoch {epoch + 1}.")
                    break

        save_json(history, run_save_dir / "training_history.json")

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)

        one_step_metrics, one_step_preds, one_step_targets = test_and_save_one_step(
            model,
            test_loader,
            device,
            run_save_dir,
            mask_tensor,
            args.hotspot_threshold,
        )
        save_example_predictions(run_save_dir, one_step_preds, one_step_targets, prefix=f"{ONE_STEP_PREFIX}_example")

        rollout_preds, rollout_targets = rollout_predict_window(
            model,
            env_full,
            ais_full,
            device,
            args.seq_len,
            window.rollout_horizon,
            window.rollout_start_index,
            mask_np,
        )
        rollout_metrics = save_rollout_results(
            run_save_dir,
            rollout_preds,
            rollout_targets,
            mask_tensor,
            args.hotspot_threshold,
        )
        save_example_predictions(run_save_dir, rollout_preds, rollout_targets, prefix=f"{ROLL_OUT_PREFIX}_example")

        run_row = build_seed_run_row(seed, best_val_loss, best_epoch, one_step_metrics, rollout_metrics)
        save_json(run_row, run_save_dir / "run_summary.json")
        run_rows.append(run_row)

        print(f"\n[{window.name}] Final one-step test metrics:")
        for key, value in one_step_metrics.items():
            print(f"  {key}: {value:.6f}")
        print(f"\n[{window.name}] Final 12-month rollout metrics:")
        for key, value in rollout_metrics.items():
            print(f"  {key}: {value:.6f}")

    finalize_window_experiment(
        base_save_dir=window_save_dir,
        seeds=seeds,
        run_rows=run_rows,
        mask_path=args.mask_path,
        hotspot_threshold=args.hotspot_threshold,
        best_checkpoint_filename=BEST_CHECKPOINT_FILENAME,
    )

    save_additional_threshold_evaluations(
        base_save_dir=window_save_dir,
        seeds=seeds,
        mask_path=args.mask_path,
        hotspot_thresholds=args.extra_hotspot_thresholds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run multi-window temporal backtests for {MODEL_LABEL}.")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory containing feature npy files.")
    parser.add_argument("--mask-path", type=str, default=str(DEFAULT_MASK_PATH), help="Shared ocean mask path.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(DEFAULT_SAVE_DIR),
        help="Root directory for backtest outputs.",
    )
    parser.add_argument(
        "--window-years",
        type=str,
        default="2022,2023,2024",
        help="Comma-separated test years to run.",
    )
    parser.add_argument(
        "--env-vars",
        type=str,
        default=",".join(DEFAULT_ENV_VARS),
        help="Comma-separated environment variables.",
    )
    parser.add_argument("--seq-len", type=int, default=12, help="Input sequence length.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction length.")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size.")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--train-loss", type=str, default="mse", help="Training loss name: mse or mae.")
    parser.add_argument("--hotspot-threshold", type=float, default=0.2755, help="Primary hotspot threshold.")
    parser.add_argument(
        "--extra-hotspot-thresholds",
        type=str,
        default="0.3175",
        help="Comma-separated extra thresholds to recompute CSI/F1 after training.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Model hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of STLSTM layers.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = Path(args.data_dir).resolve()
    args.mask_path = Path(args.mask_path).resolve()
    args.save_dir = Path(args.save_dir).resolve()
    args.env_vars = [item.strip() for item in args.env_vars.split(",") if item.strip()]
    args.extra_hotspot_thresholds = parse_float_list(args.extra_hotspot_thresholds)

    seeds = parse_int_list(args.seeds, DEFAULT_SEEDS)
    requested_years = parse_int_list(args.window_years, [2022, 2023, 2024])
    windows = [WINDOW_CONFIGS[year] for year in requested_years if year in WINDOW_CONFIGS]
    if not windows:
        raise RuntimeError(f"No valid windows selected from: {requested_years}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_tensor = None
    if args.mask_path.exists():
        mask_tensor = torch.from_numpy(np.load(args.mask_path).astype(np.float32)).to(device)

    env_full, ais_full = load_full_series(args.data_dir, args.env_vars)
    time_axis = load_time_axis(args.data_dir)
    if len(time_axis) != len(env_full):
        raise RuntimeError(
            f"Time axis length mismatch: len(time_axis)={len(time_axis)} vs len(env_full)={len(env_full)}"
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "windows": [asdict(window) for window in windows],
            "window_years": [window.test_year for window in windows],
            "env_vars": list(args.env_vars),
            "seq_len": int(args.seq_len),
            "pred_len": int(args.pred_len),
            "batch_size": int(args.batch_size),
            "accumulation_steps": int(args.accumulation_steps),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "train_loss": str(args.train_loss),
            "hotspot_threshold": float(args.hotspot_threshold),
            "extra_hotspot_thresholds": list(args.extra_hotspot_thresholds),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "seeds": list(seeds),
            "data_dir": str(args.data_dir),
            "mask_path": str(args.mask_path),
            "save_dir": str(args.save_dir),
        },
        args.save_dir / "backtest_config.json",
    )

    for window in windows:
        train_single_window(window, env_full, ais_full, time_axis, args, mask_tensor, device, seeds)

    save_root_summaries(
        save_dir=args.save_dir,
        windows=windows,
        time_axis=time_axis,
        hotspot_threshold=args.hotspot_threshold,
        extra_thresholds=args.extra_hotspot_thresholds,
    )

    print("\nTemporal backtest completed.")
    print(f"Saved root summary to: {args.save_dir / 'temporal_backtest_summary.csv'}")
    for window in windows:
        print(f"Saved window outputs to: {args.save_dir / window.name}")


if __name__ == "__main__":
    main()
