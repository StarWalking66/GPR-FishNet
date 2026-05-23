from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from process.dataset import STFishNetUltimateDataset
from utils.multi_seed_experiment import (
    DEFAULT_SEEDS,
    build_seed_run_row,
    finalize_multi_seed_experiment,
    resolve_seed_save_dir,
    save_additional_threshold_evaluations,
    threshold_tag,
)
from utils.output_clamp import apply_output_clamp


TensorInputTransform = Callable[[torch.Tensor], torch.Tensor]
RolloutInputTransform = Callable[[np.ndarray], np.ndarray]


@dataclass
class ExperimentConfig:
    data_dir: str
    mask_path: str
    seq_len: int = 12
    pred_len: int = 1
    rollout_2024_horizon: int = 12
    rollout_start_index: int = 144

    batch_size: int = 2
    accumulation_steps: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    train_loss: str = "mse"
    hotspot_threshold: float = 0.2755

    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 8
    min_lr: float = 1e-6
    early_stop_patience: int = 20
    max_grad_norm: float = 5.0

    hidden_dim: int = 64
    img_size: Tuple[int, int] = (64, 96)
    num_layers: int = 2


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj: Dict[str, Any], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    ssim = float(((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))

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
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    mask_tensor: Optional[torch.Tensor] = None,
    input_batch_transform: Optional[TensorInputTransform] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if input_batch_transform is not None:
                inputs = input_batch_transform(inputs)
            targets = targets.to(device)
            preds = apply_output_clamp(model(inputs))
            if loss_name.lower() == "mae":
                loss = masked_mae_loss(preds, targets, mask_tensor)
            else:
                loss = masked_mse_loss(preds, targets, mask_tensor)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def test_and_save_one_step(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    mask_tensor: Optional[torch.Tensor],
    hotspot_threshold: float,
    file_prefix: str,
    input_batch_transform: Optional[TensorInputTransform] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_inputs: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"[{file_prefix.upper()}] Inference"):
            inputs = inputs.to(device)
            if input_batch_transform is not None:
                inputs = input_batch_transform(inputs)
            targets = targets.to(device)
            preds = apply_output_clamp(model(inputs))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    inputs_np = np.concatenate(all_inputs, axis=0)

    np.save(os.path.join(save_dir, f"{file_prefix}_preds.npy"), preds_np)
    np.save(os.path.join(save_dir, f"{file_prefix}_targets.npy"), targets_np)
    np.save(os.path.join(save_dir, f"{file_prefix}_inputs.npy"), inputs_np)

    metrics = compute_metrics(preds_np, targets_np, mask_tensor, hotspot_threshold)
    save_json(metrics, os.path.join(save_dir, f"{file_prefix}_metrics.json"))
    return metrics, preds_np, targets_np


def save_example_predictions(save_dir: str, preds: np.ndarray, targets: np.ndarray, num_examples: int = 5, prefix: str = "example") -> None:
    num_examples = min(num_examples, preds.shape[0])
    example_dir = os.path.join(save_dir, f"{prefix}_samples")
    os.makedirs(example_dir, exist_ok=True)
    for i in range(num_examples):
        np.save(os.path.join(example_dir, f"{prefix}_{i}_pred.npy"), preds[i])
        np.save(os.path.join(example_dir, f"{prefix}_{i}_target.npy"), targets[i])


def load_full_series(data_dir: str, env_vars: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    env_list = [
        np.concatenate(
            [
                np.load(os.path.join(data_dir, f"{var}_train.npy")),
                np.load(os.path.join(data_dir, f"{var}_val.npy")),
                np.load(os.path.join(data_dir, f"{var}_test.npy")),
            ],
            axis=0,
        )
        for var in env_vars
    ]
    env_full = np.concatenate(env_list, axis=1).astype(np.float32)
    ais_full = np.concatenate(
        [
            np.load(os.path.join(data_dir, "ais_train.npy")),
            np.load(os.path.join(data_dir, "ais_val.npy")),
            np.load(os.path.join(data_dir, "ais_test.npy")),
        ],
        axis=0,
    ).astype(np.float32)
    return env_full, ais_full


def rollout_predict_2024(
    model: nn.Module,
    data_dir: str,
    env_vars: Sequence[str],
    device: torch.device,
    seq_len: int,
    horizon: int,
    start_idx: int,
    mask_np: Optional[np.ndarray],
    rollout_input_transform: Optional[RolloutInputTransform] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    env_full, ais_full = load_full_series(data_dir, env_vars)
    hist_env = env_full[start_idx - seq_len: start_idx].copy()
    hist_ais = ais_full[start_idx - seq_len: start_idx].copy()

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for step in range(horizon):
            current_input = np.concatenate([hist_env, hist_ais], axis=1)
            if rollout_input_transform is not None:
                current_input = rollout_input_transform(current_input)
            current_input_t = torch.from_numpy(current_input[None]).float().to(device)
            pred = apply_output_clamp(model(current_input_t))
            pred_step = pred[0, 0].cpu().numpy()
            if mask_np is not None:
                pred_step[:, mask_np == 0] = 0.0

            preds.append(pred_step)
            targets.append(ais_full[start_idx + step])

            hist_env = np.concatenate([hist_env[1:], env_full[start_idx + step][None]], axis=0)
            hist_ais = np.concatenate([hist_ais[1:], pred_step[None]], axis=0)

    return np.stack(preds, axis=0).astype(np.float32), np.stack(targets, axis=0).astype(np.float32)


def save_rollout_results(
    save_dir: str,
    preds: np.ndarray,
    targets: np.ndarray,
    mask_tensor: Optional[torch.Tensor],
    hotspot_threshold: float,
    file_prefix: str,
) -> Dict[str, float]:
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{file_prefix}_preds.npy"), preds)
    np.save(os.path.join(save_dir, f"{file_prefix}_targets.npy"), targets)
    metrics = compute_metrics(preds, targets, mask_tensor, hotspot_threshold)
    save_json(metrics, os.path.join(save_dir, f"{file_prefix}_metrics.json"))
    return metrics


def collect_summary_row(experiment_name: str, save_dir: str, extra_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    test_metrics = load_json(os.path.join(save_dir, "test_one_step_metrics.json"))
    rollout_metrics = load_json(os.path.join(save_dir, "rollout_2024_metrics.json"))
    train_metrics = load_json(os.path.join(save_dir, "training_history.json"))
    selected_seed_payload = load_json(os.path.join(save_dir, "selected_seed.json"))

    row: Dict[str, Any] = {
        "experiment": experiment_name,
        "save_dir": save_dir,
        "selected_seed": int(selected_seed_payload.get("selected_seed", -1)),
        "best_val_loss_mean": float(train_metrics.get("best_val_loss_mean", np.nan)),
        "best_val_loss_std": float(train_metrics.get("best_val_loss_std", np.nan)),
    }

    for metric_name, metric_value in test_metrics.get("metrics", {}).items():
        row[f"test_one_step_{metric_name}"] = float(metric_value)
    for metric_name, metric_value in test_metrics.get("std", {}).items():
        row[f"test_one_step_{metric_name}_std"] = float(metric_value)

    for metric_name, metric_value in rollout_metrics.get("metrics", {}).items():
        row[f"rollout_2024_{metric_name}"] = float(metric_value)
    for metric_name, metric_value in rollout_metrics.get("std", {}).items():
        row[f"rollout_2024_{metric_name}_std"] = float(metric_value)

    if extra_fields:
        row.update(extra_fields)
    return row


def collect_summary_row_for_threshold(
    experiment_name: str,
    save_dir: str,
    hotspot_threshold: float,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_test_metrics = load_json(os.path.join(save_dir, "test_one_step_metrics.json"))
    base_rollout_metrics = load_json(os.path.join(save_dir, "rollout_2024_metrics.json"))
    train_metrics = load_json(os.path.join(save_dir, "training_history.json"))
    selected_seed_payload = load_json(os.path.join(save_dir, "selected_seed.json"))

    tag = threshold_tag(hotspot_threshold)
    threshold_test_metrics = load_json(os.path.join(save_dir, f"test_one_step_metrics_threshold_{tag}.json"))
    threshold_rollout_metrics = load_json(os.path.join(save_dir, f"rollout_2024_metrics_threshold_{tag}.json"))

    row: Dict[str, Any] = {
        "experiment": experiment_name,
        "save_dir": save_dir,
        "selected_seed": int(selected_seed_payload.get("selected_seed", -1)),
        "best_val_loss_mean": float(train_metrics.get("best_val_loss_mean", np.nan)),
        "best_val_loss_std": float(train_metrics.get("best_val_loss_std", np.nan)),
        "hotspot_threshold": float(hotspot_threshold),
    }

    def _attach_split_metrics(
        split_prefix: str,
        base_payload: Dict[str, Any],
        threshold_payload: Dict[str, Any],
    ) -> None:
        for metric_name, metric_value in base_payload.get("metrics", {}).items():
            if metric_name in {"CSI", "F1", "Precision", "Recall"}:
                continue
            row[f"{split_prefix}_{metric_name}"] = float(metric_value)
        for metric_name, metric_value in base_payload.get("std", {}).items():
            if metric_name in {"CSI", "F1", "Precision", "Recall"}:
                continue
            row[f"{split_prefix}_{metric_name}_std"] = float(metric_value)

        for metric_name, metric_value in threshold_payload.get("metrics", {}).items():
            row[f"{split_prefix}_{metric_name}"] = float(metric_value)
        for metric_name, metric_value in threshold_payload.get("std", {}).items():
            row[f"{split_prefix}_{metric_name}_std"] = float(metric_value)

    _attach_split_metrics("test_one_step", base_test_metrics, threshold_test_metrics)
    _attach_split_metrics("rollout_2024", base_rollout_metrics, threshold_rollout_metrics)

    if extra_fields:
        row.update(extra_fields)
    return row


def save_rows(rows: List[Dict[str, Any]], csv_path: str, json_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)

    if not rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    all_keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def run_multi_seed_experiment(
    experiment_name: str,
    save_dir: str,
    env_vars: Sequence[str],
    model_builder: Callable[[int, ExperimentConfig], nn.Module],
    config: ExperimentConfig,
    best_checkpoint_filename: str,
    metadata: Optional[Dict[str, Any]] = None,
    input_batch_transform: Optional[TensorInputTransform] = None,
    rollout_input_transform: Optional[RolloutInputTransform] = None,
    extra_hotspot_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    env_vars = list(env_vars)

    config_payload = asdict(config)
    config_payload["env_vars"] = env_vars
    config_payload["device"] = str(device)
    if metadata is not None:
        config_payload["metadata"] = metadata
    save_json(config_payload, os.path.join(save_dir, "experiment_config.json"))

    mask_tensor = torch.from_numpy(np.load(config.mask_path).astype(np.float32)).to(device) if os.path.exists(config.mask_path) else None

    run_rows: List[Dict[str, Any]] = []
    for seed in config.seeds:
        set_seed(seed)
        scaler = torch.amp.GradScaler(enabled=use_amp)
        run_save_dir = resolve_seed_save_dir(save_dir, seed, config.seeds)
        os.makedirs(run_save_dir, exist_ok=True)

        train_dataset = STFishNetUltimateDataset(config.data_dir, env_vars, "train", config.seq_len, config.pred_len)
        val_dataset = STFishNetUltimateDataset(config.data_dir, env_vars, "val", config.seq_len, config.pred_len)
        test_dataset = STFishNetUltimateDataset(config.data_dir, env_vars, "test", config.seq_len, config.pred_len)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)

        model = model_builder(len(env_vars) + 1, config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.min_lr,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_epochs = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_model_path = os.path.join(run_save_dir, best_checkpoint_filename)

        print(f"Starting {experiment_name} | seed={seed} | env_vars={env_vars}")

        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_loader, desc=f"{experiment_name} | seed {seed} | epoch {epoch + 1}/{config.epochs}")

            for i, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                if input_batch_transform is not None:
                    inputs = input_batch_transform(inputs)
                targets = targets.to(device)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    preds = model(inputs)
                    if config.train_loss.lower() == "mae":
                        loss = masked_mae_loss(preds, targets, mask=mask_tensor)
                    else:
                        loss = masked_mse_loss(preds, targets, mask=mask_tensor)
                    loss = loss / config.accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_accumulation_boundary = ((i + 1) % config.accumulation_steps == 0) or ((i + 1) == len(train_loader))
                if is_accumulation_boundary:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                real_loss = loss.item() * config.accumulation_steps
                running_loss += real_loss
                pbar.set_postfix({"loss": f"{real_loss:.6f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            avg_train_loss = running_loss / max(len(train_loader), 1)
            avg_val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                loss_name=config.train_loss,
                mask_tensor=mask_tensor,
                input_batch_transform=input_batch_transform,
            )
            scheduler.step(avg_val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

            history["train_loss"].append(float(avg_train_loss))
            history["val_loss"].append(float(avg_val_loss))
            history["lr"].append(current_lr)

            print(
                f"{experiment_name} | seed {seed} | epoch {epoch + 1:03d} "
                f"| train={avg_train_loss:.6f} | val={avg_val_loss:.6f} | lr={current_lr:.2e}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated: {best_model_path}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= config.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        save_json(history, os.path.join(run_save_dir, "training_history.json"))

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)

        one_step_metrics, one_step_preds, one_step_targets = test_and_save_one_step(
            model=model,
            loader=test_loader,
            device=device,
            save_dir=run_save_dir,
            mask_tensor=mask_tensor,
            hotspot_threshold=config.hotspot_threshold,
            file_prefix="test_one_step",
            input_batch_transform=input_batch_transform,
        )
        save_example_predictions(run_save_dir, one_step_preds, one_step_targets, prefix="test_one_step_example")

        mask_np = mask_tensor.detach().cpu().numpy() if mask_tensor is not None else None
        rollout_preds, rollout_targets = rollout_predict_2024(
            model=model,
            data_dir=config.data_dir,
            env_vars=env_vars,
            device=device,
            seq_len=config.seq_len,
            horizon=config.rollout_2024_horizon,
            start_idx=config.rollout_start_index,
            mask_np=mask_np,
            rollout_input_transform=rollout_input_transform,
        )
        rollout_metrics = save_rollout_results(
            save_dir=run_save_dir,
            preds=rollout_preds,
            targets=rollout_targets,
            mask_tensor=mask_tensor,
            hotspot_threshold=config.hotspot_threshold,
            file_prefix="rollout_2024",
        )
        save_example_predictions(run_save_dir, rollout_preds, rollout_targets, prefix="rollout_2024_example")

        run_row = build_seed_run_row(seed, best_val_loss, best_epoch, one_step_metrics, rollout_metrics)
        save_json(run_row, os.path.join(run_save_dir, "run_summary.json"))
        run_rows.append(run_row)

        print(f"{experiment_name} | seed {seed} | one-step metrics: {one_step_metrics}")
        print(f"{experiment_name} | seed {seed} | rollout metrics: {rollout_metrics}")

    finalize_multi_seed_experiment(
        base_save_dir=save_dir,
        seeds=config.seeds,
        run_rows=run_rows,
        mask_path=config.mask_path,
        hotspot_threshold=config.hotspot_threshold,
        best_checkpoint_filename=best_checkpoint_filename,
    )
    if extra_hotspot_thresholds:
        save_additional_threshold_evaluations(
            base_save_dir=save_dir,
            seeds=config.seeds,
            mask_path=config.mask_path,
            hotspot_thresholds=extra_hotspot_thresholds,
        )
    return collect_summary_row(experiment_name, save_dir, extra_fields=metadata)
