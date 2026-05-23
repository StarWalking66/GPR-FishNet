import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np


MODEL_LABELS = {
    "checkpoints_predrnn_baseline": "PredRNN",
    "checkpoints_xgboost_baseline": "XGBoost",
    "checkpoints_predrnn_v2_baseline": "PredRNN_v2",
    "checkpoints_unet_baseline": "U-Net",
    "checkpoints_convlstm_baseline": "ConvLSTM",
    "checkpoints_swinlstm_baseline": "SwinLSTM",
}

LOWER_IS_BETTER = {"MAE", "MSE", "RMSE"}
HIGHER_IS_BETTER = {"R2", "SSIM", "CSI", "F1"}
RANK_METRICS = ["MAE", "MSE", "RMSE", "R2", "SSIM", "CSI", "F1"]
SPLITS = ["test_one_step", "rollout_2024"]


def get_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, os.pardir))


def format_threshold_tag(threshold: float) -> str:
    return f"{threshold:.4f}".replace(".", "p")


def load_train_99p(data_dir: str) -> float:
    params_path = os.path.join(data_dir, "ais_norm_params.npy")
    if not os.path.exists(params_path):
        return 1.0
    params = np.load(params_path, allow_pickle=True).item()
    return float(params.get("train_99p", 1.0))


def load_mask(mask_path: str) -> np.ndarray:
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    return np.load(mask_path).astype(bool)


def normalize_spatiotemporal_array(array_path: str, mask_shape: Tuple[int, int]) -> np.ndarray:
    if not os.path.exists(array_path):
        raise FileNotFoundError(f"Array file not found: {array_path}")
    array = np.squeeze(np.load(array_path))
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected [T, H, W] after squeeze for {array_path}, got {array.shape}")
    if array.shape[-2:] != mask_shape:
        raise ValueError(f"Spatial shape mismatch for {array_path}: {array.shape[-2:]} vs {mask_shape}")
    return array.astype(np.float64)


def compute_metrics(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray, hotspot_threshold: float) -> Dict[str, float]:
    if preds.shape != targets.shape:
        raise ValueError(f"Pred/target shape mismatch: {preds.shape} vs {targets.shape}")

    pred_values = preds[:, mask]
    target_values = targets[:, mask]

    valid = np.isfinite(pred_values) & np.isfinite(target_values)
    p = pred_values[valid].astype(np.float64)
    t = target_values[valid].astype(np.float64)

    if p.size == 0:
        return {metric_name: 0.0 for metric_name in RANK_METRICS}

    diff = p - t
    mae = float(np.abs(diff).mean())
    mse = float((diff ** 2).mean())
    rmse = float(np.sqrt(mse))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2) + 1e-8
    r2 = float(1.0 - ss_res / ss_tot)

    mu_x = np.mean(p)
    mu_y = np.mean(t)
    var_x = np.var(p)
    var_y = np.var(t)
    cov_xy = np.mean((p - mu_x) * (t - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
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

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "SSIM": ssim,
        "CSI": csi,
        "F1": f1,
    }


def list_model_dirs(model_outcomes_dir: str) -> List[str]:
    results = []
    for folder_name in MODEL_LABELS:
        folder_path = os.path.join(model_outcomes_dir, folder_name)
        if os.path.isdir(folder_path):
            results.append(folder_path)
    return results


def save_threshold_metrics(
    model_dir: str,
    split_name: str,
    threshold: float,
    threshold_hours: float,
    metrics: Dict[str, float],
) -> str:
    tag = format_threshold_tag(threshold)
    output_path = os.path.join(model_dir, f"{split_name}_metrics_threshold_{tag}.json")
    payload = {
        "threshold_norm": float(threshold),
        "threshold_physical_hours_per_day": float(threshold_hours),
        "split": split_name,
        "metrics": metrics,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    return output_path


def overwrite_base_metrics(model_dir: str, split_name: str, metrics: Dict[str, float]) -> str:
    output_path = os.path.join(model_dir, f"{split_name}_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    return output_path


def compute_average_ranks(rows: List[Dict[str, object]], split_name: str) -> None:
    for metric_name in RANK_METRICS:
        candidates = []
        metric_key = f"{split_name}_{metric_name}"
        for row in rows:
            value = row.get(metric_key)
            if value is not None:
                candidates.append((row, float(value)))

        reverse = metric_name in HIGHER_IS_BETTER
        sorted_candidates = sorted(candidates, key=lambda item: item[1], reverse=reverse)
        for rank, (row, _) in enumerate(sorted_candidates, start=1):
            row[f"{split_name}_{metric_name}_rank"] = float(rank)

    for row in rows:
        rank_values = [float(row[f"{split_name}_{metric_name}_rank"]) for metric_name in RANK_METRICS]
        row[f"{split_name}_avg_rank"] = float(sum(rank_values) / len(rank_values))


def write_summary_csv(rows: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "model_label",
        "model_dir",
        "threshold_norm",
        "threshold_physical_hours_per_day",
    ]
    for split_name in SPLITS:
        for metric_name in RANK_METRICS:
            fieldnames.append(f"{split_name}_{metric_name}")
        for metric_name in RANK_METRICS:
            fieldnames.append(f"{split_name}_{metric_name}_rank")
        fieldnames.append(f"{split_name}_avg_rank")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(rows: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)


def main() -> None:
    project_root = get_project_root()
    default_data_dir = os.path.join(project_root, "data", "ST_FishNet_Features")
    default_model_outcomes_dir = os.path.join(project_root, "model_outcomes")
    default_mask_path = os.path.join(default_data_dir, "all_vars_train_mask_intersection.npy")
    default_summary_dir = os.path.join(default_model_outcomes_dir, "threshold_recomputed_metrics")

    parser = argparse.ArgumentParser(
        description="Recompute hotspot-aware metrics for all saved model outputs without retraining."
    )
    parser.add_argument("--threshold", type=float, default=0.3175, help="Normalized hotspot threshold.")
    parser.add_argument("--data-dir", default=default_data_dir, help="Data directory containing ais_norm_params.npy.")
    parser.add_argument("--mask-path", default=default_mask_path, help="Ocean-valid mask used during evaluation.")
    parser.add_argument("--model-outcomes-dir", default=default_model_outcomes_dir, help="Root directory containing model checkpoint folders.")
    parser.add_argument("--summary-dir", default=default_summary_dir, help="Directory to save recomputed summary tables.")
    parser.add_argument("--overwrite-base-json", action="store_true", help="Overwrite existing split metrics json files in each model folder.")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"Threshold must be within [0, 1], got {args.threshold}")

    mask = load_mask(args.mask_path)
    train_99p = load_train_99p(args.data_dir)
    threshold_hours = float(args.threshold * train_99p)
    model_dirs = list_model_dirs(args.model_outcomes_dir)

    if not model_dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {args.model_outcomes_dir}")

    summary_rows: List[Dict[str, object]] = []
    saved_json_paths: List[str] = []

    for model_dir in model_dirs:
        folder_name = os.path.basename(model_dir)
        model_label = MODEL_LABELS.get(folder_name, folder_name)
        row: Dict[str, object] = {
            "model_label": model_label,
            "model_dir": model_dir,
            "threshold_norm": float(args.threshold),
            "threshold_physical_hours_per_day": threshold_hours,
        }

        print(f"[INFO] Recomputing metrics for {model_label} ...")
        for split_name in SPLITS:
            preds_path = os.path.join(model_dir, f"{split_name}_preds.npy")
            targets_path = os.path.join(model_dir, f"{split_name}_targets.npy")
            preds = normalize_spatiotemporal_array(preds_path, mask.shape)
            targets = normalize_spatiotemporal_array(targets_path, mask.shape)
            metrics = compute_metrics(preds, targets, mask, args.threshold)

            threshold_metrics_path = save_threshold_metrics(
                model_dir=model_dir,
                split_name=split_name,
                threshold=args.threshold,
                threshold_hours=threshold_hours,
                metrics=metrics,
            )
            saved_json_paths.append(threshold_metrics_path)

            if args.overwrite_base_json:
                overwrite_base_metrics(model_dir, split_name, metrics)

            for metric_name, metric_value in metrics.items():
                row[f"{split_name}_{metric_name}"] = float(metric_value)

        summary_rows.append(row)

    for split_name in SPLITS:
        compute_average_ranks(summary_rows, split_name)

    summary_rows.sort(key=lambda row: float(row["rollout_2024_avg_rank"]))

    threshold_tag = format_threshold_tag(args.threshold)
    summary_csv_path = os.path.join(args.summary_dir, f"summary_metrics_threshold_{threshold_tag}.csv")
    summary_json_path = os.path.join(args.summary_dir, f"summary_metrics_threshold_{threshold_tag}.json")
    write_summary_csv(summary_rows, summary_csv_path)
    write_summary_json(summary_rows, summary_json_path)

    print(f"[INFO] train_99p = {train_99p:.6f} hours/day")
    print(f"[INFO] Threshold = {args.threshold:.4f} norm ({threshold_hours:.4f} hours/day)")
    print("[INFO] Rollout average-rank order:")
    for index, row in enumerate(summary_rows, start=1):
        print(
            f"  {index}. {row['model_label']} | "
            f"rollout_avg_rank={float(row['rollout_2024_avg_rank']):.3f} | "
            f"rollout_CSI={float(row['rollout_2024_CSI']):.6f} | "
            f"rollout_F1={float(row['rollout_2024_F1']):.6f}"
        )
    print(f"[OK] Summary CSV saved to: {summary_csv_path}")
    print(f"[OK] Summary JSON saved to: {summary_json_path}")
    print(f"[OK] Threshold-specific per-model JSON files saved: {len(saved_json_paths)}")
    if args.overwrite_base_json:
        print("[WARN] Existing base metrics json files were overwritten with the new threshold results.")


if __name__ == "__main__":
    main()
