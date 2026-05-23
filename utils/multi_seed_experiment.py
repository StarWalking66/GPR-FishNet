from __future__ import annotations

import csv
import json
import os
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46]
SUMMARY_METRIC_KEYS: Tuple[str, ...] = ("MAE", "MSE", "RMSE", "R2", "SSIM", "CSI", "F1")


def save_json(obj: Dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def write_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_seed_save_dir(base_save_dir: str, seed: int, seeds: Sequence[int]) -> str:
    if len(seeds) <= 1:
        return base_save_dir
    return os.path.join(base_save_dir, f"seed_{seed}")


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def threshold_tag(hotspot_threshold: float) -> str:
    return f"{hotspot_threshold:.10f}".rstrip("0").rstrip(".").replace(".", "p")


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
    row.update(flatten_metrics("test_one_step", one_step_metrics))
    row.update(flatten_metrics("rollout_2024", rollout_metrics))
    return row


def _aggregate_numeric_rows(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}

    metric_columns = []
    for key, value in rows[0].items():
        if key == "seed":
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            metric_columns.append(key)

    aggregated: Dict[str, float] = {}
    for key in metric_columns:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values, ddof=0))
    return aggregated


def _load_mask(mask_path: str, shape: Tuple[int, ...]) -> np.ndarray:
    if os.path.exists(mask_path):
        mask = np.load(mask_path).astype(np.float32)
    else:
        mask = np.ones(shape[-2:], dtype=np.float32)
    while mask.ndim < len(shape):
        mask = np.expand_dims(mask, axis=0)
    return np.broadcast_to(mask > 0.5, shape)


def _compute_threshold_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    mask_path: str,
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


def _save_aggregated_split_outputs(
    base_save_dir: str,
    split_name: str,
    seeds: Sequence[int],
    mask_path: str,
    hotspot_threshold: float,
) -> None:
    seed_dirs = [resolve_seed_save_dir(base_save_dir, seed, seeds) for seed in seeds]
    preds_list = []
    targets_first = None
    inputs_first = None
    threshold_rows = []

    for run_dir in seed_dirs:
        preds_path = os.path.join(run_dir, f"{split_name}_preds.npy")
        targets_path = os.path.join(run_dir, f"{split_name}_targets.npy")
        if not (os.path.exists(preds_path) and os.path.exists(targets_path)):
            continue

        preds = np.load(preds_path)
        targets = np.load(targets_path)
        preds_list.append(preds.astype(np.float32))

        if targets_first is None:
            targets_first = targets.astype(np.float32)
        inputs_path = os.path.join(run_dir, f"{split_name}_inputs.npy")
        if inputs_first is None and os.path.exists(inputs_path):
            inputs_first = np.load(inputs_path).astype(np.float32)

        threshold_rows.append(_compute_threshold_metrics(preds, targets, mask_path, hotspot_threshold))

    if not preds_list:
        return

    mean_preds = np.mean(np.stack(preds_list, axis=0), axis=0).astype(np.float32)
    np.save(os.path.join(base_save_dir, f"{split_name}_preds.npy"), mean_preds)
    if targets_first is not None:
        np.save(os.path.join(base_save_dir, f"{split_name}_targets.npy"), targets_first)
    if inputs_first is not None:
        np.save(os.path.join(base_save_dir, f"{split_name}_inputs.npy"), inputs_first)

    threshold_mean, threshold_std = _aggregate_dict_list(threshold_rows)
    threshold_payload = {
        "aggregation": "mean_of_seed_threshold_metrics",
        "num_seeds": int(len(threshold_rows)),
        "seeds": list(seeds),
        "threshold_norm": float(hotspot_threshold),
        "metrics": threshold_mean,
        "std": threshold_std,
    }
    tag = threshold_tag(hotspot_threshold)
    save_json(threshold_payload, os.path.join(base_save_dir, f"{split_name}_metrics_threshold_{tag}.json"))


def save_additional_threshold_evaluations(
    base_save_dir: str,
    seeds: Sequence[int],
    mask_path: str,
    hotspot_thresholds: Sequence[float],
) -> List[float]:
    thresholds: List[float] = []
    for value in hotspot_thresholds:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(parsed):
            continue
        if any(abs(parsed - existing) < 1e-12 for existing in thresholds):
            continue
        thresholds.append(parsed)

    if not thresholds:
        return []

    for hotspot_threshold in thresholds:
        tag = threshold_tag(hotspot_threshold)
        threshold_run_rows: List[Dict] = []

        for seed in seeds:
            run_dir = resolve_seed_save_dir(base_save_dir, seed, seeds)
            run_summary_path = os.path.join(run_dir, "run_summary.json")
            if not os.path.exists(run_summary_path):
                continue

            with open(run_summary_path, "r", encoding="utf-8") as f:
                run_summary = json.load(f)
            test_preds_path = os.path.join(run_dir, "test_one_step_preds.npy")
            test_targets_path = os.path.join(run_dir, "test_one_step_targets.npy")
            rollout_preds_path = os.path.join(run_dir, "rollout_2024_preds.npy")
            rollout_targets_path = os.path.join(run_dir, "rollout_2024_targets.npy")
            if not all(
                os.path.exists(path)
                for path in (test_preds_path, test_targets_path, rollout_preds_path, rollout_targets_path)
            ):
                continue

            test_threshold_metrics = _compute_threshold_metrics(
                np.load(test_preds_path),
                np.load(test_targets_path),
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
                    "metrics": test_threshold_metrics,
                },
                os.path.join(run_dir, f"test_one_step_metrics_threshold_{tag}.json"),
            )
            save_json(
                {
                    "threshold_norm": float(hotspot_threshold),
                    "seed": int(seed),
                    "metrics": rollout_threshold_metrics,
                },
                os.path.join(run_dir, f"rollout_2024_metrics_threshold_{tag}.json"),
            )

            threshold_run_row = dict(run_summary)
            threshold_run_row["test_one_step_CSI"] = float(test_threshold_metrics["CSI"])
            threshold_run_row["test_one_step_F1"] = float(test_threshold_metrics["F1"])
            threshold_run_row["test_one_step_Precision"] = float(test_threshold_metrics["Precision"])
            threshold_run_row["test_one_step_Recall"] = float(test_threshold_metrics["Recall"])
            threshold_run_row["rollout_2024_CSI"] = float(rollout_threshold_metrics["CSI"])
            threshold_run_row["rollout_2024_F1"] = float(rollout_threshold_metrics["F1"])
            threshold_run_row["rollout_2024_Precision"] = float(rollout_threshold_metrics["Precision"])
            threshold_run_row["rollout_2024_Recall"] = float(rollout_threshold_metrics["Recall"])
            threshold_run_row["threshold_norm"] = float(hotspot_threshold)
            save_json(
                threshold_run_row,
                os.path.join(run_dir, f"run_summary_threshold_{tag}.json"),
            )
            threshold_run_rows.append(threshold_run_row)

        if not threshold_run_rows:
            continue

        ordered_rows = sorted(threshold_run_rows, key=lambda row: int(row["seed"]))
        aggregated = _aggregate_numeric_rows(ordered_rows)
        write_csv(ordered_rows, os.path.join(base_save_dir, f"summary_runs_threshold_{tag}.csv"))
        write_csv([aggregated], os.path.join(base_save_dir, f"summary_seed_mean_std_threshold_{tag}.csv"))
        save_json(
            {
                "seeds": list(seeds),
                "threshold_norm": float(hotspot_threshold),
                "runs": ordered_rows,
                "aggregated": aggregated,
            },
            os.path.join(base_save_dir, f"summary_all_threshold_{tag}.json"),
        )

        _save_aggregated_split_outputs(base_save_dir, "test_one_step", seeds, mask_path, hotspot_threshold)
        _save_aggregated_split_outputs(base_save_dir, "rollout_2024", seeds, mask_path, hotspot_threshold)

    return thresholds


def finalize_multi_seed_experiment(
    base_save_dir: str,
    seeds: Sequence[int],
    run_rows: List[Dict],
    mask_path: str,
    hotspot_threshold: float,
    best_checkpoint_filename: str,
) -> None:
    os.makedirs(base_save_dir, exist_ok=True)

    if not run_rows:
        return

    ordered_rows = sorted(run_rows, key=lambda row: int(row["seed"]))
    aggregated = _aggregate_numeric_rows(ordered_rows)

    write_csv(ordered_rows, os.path.join(base_save_dir, "summary_runs.csv"))
    write_csv([aggregated], os.path.join(base_save_dir, "summary_seed_mean_std.csv"))
    save_json(
        {
            "seeds": list(seeds),
            "runs": ordered_rows,
            "aggregated": aggregated,
        },
        os.path.join(base_save_dir, "summary_all.json"),
    )

    test_rows = []
    rollout_rows = []
    for row in ordered_rows:
        test_rows.append({key: float(row[f"test_one_step_{key}"]) for key in SUMMARY_METRIC_KEYS if f"test_one_step_{key}" in row})
        rollout_rows.append({key: float(row[f"rollout_2024_{key}"]) for key in SUMMARY_METRIC_KEYS if f"rollout_2024_{key}" in row})

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
        os.path.join(base_save_dir, "test_one_step_metrics.json"),
    )
    save_json(
        {
            "aggregation": "mean_of_seed_metrics",
            "num_seeds": int(len(ordered_rows)),
            "seeds": list(seeds),
            "metrics": rollout_mean,
            "std": rollout_std,
        },
        os.path.join(base_save_dir, "rollout_2024_metrics.json"),
    )

    _save_aggregated_split_outputs(base_save_dir, "test_one_step", seeds, mask_path, hotspot_threshold)
    _save_aggregated_split_outputs(base_save_dir, "rollout_2024", seeds, mask_path, hotspot_threshold)

    best_row = min(ordered_rows, key=lambda row: float(row["best_val_loss"]))
    best_seed = int(best_row["seed"])
    best_seed_dir = resolve_seed_save_dir(base_save_dir, best_seed, seeds)
    best_checkpoint_src = os.path.join(best_seed_dir, best_checkpoint_filename)
    if os.path.exists(best_checkpoint_src):
        shutil.copy2(best_checkpoint_src, os.path.join(base_save_dir, best_checkpoint_filename))

    save_json(
        {
            "selected_seed": best_seed,
            "selection_rule": "lowest_best_val_loss",
            "best_val_loss": float(best_row["best_val_loss"]),
            "best_epoch": int(best_row["best_epoch"]),
            "checkpoint_filename": best_checkpoint_filename,
        },
        os.path.join(base_save_dir, "selected_seed.json"),
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
        os.path.join(base_save_dir, "training_history.json"),
    )
