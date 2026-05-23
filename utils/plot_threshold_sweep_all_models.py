import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.generate_model_ranking_xlsx import (  # noqa: E402
    MASK_PATH,
    MODEL_OUTCOMES_DIR,
    THRESHOLD,
    load_checkpoint_rows,
    threshold_tag,
)


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["axes.linewidth"] = 1.2
rcParams["xtick.major.width"] = 1.2
rcParams["ytick.major.width"] = 1.2
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["axes.labelweight"] = "bold"


OUTPUT_DIR = MODEL_OUTCOMES_DIR / "threshold_sweeps"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ST_FishNet_Features"
METRICS = ("CSI", "F1", "Precision", "Recall")
SPLITS: Sequence[Tuple[str, str]] = (
    ("test_one_step", "One-step Forecasting"),
    ("rollout_2024", "2024 Rollout Forecasting"),
)
MODEL_ORDER = [
    "checkpoints_gpr_fishnet_final",
    "checkpoints_predrnn_baseline",
    "checkpoints_pfgnet_final",
    "checkpoints_convlstm_baseline",
    "checkpoints_predrnn_v2_baseline",
    "checkpoints_exprecast_final",
    "checkpoints_timekan_final",
    "checkpoints_seacast_baseline",
    "checkpoints_swinlstm_baseline",
]
LABEL_OVERRIDES = {
    "checkpoints_gpr_fishnet_final": "GPR-FishNet (ours)",
}
MODEL_STYLES = {
    "GPR-FishNet (ours)": {"color": "#B91C1C", "lw": 2.8, "ls": "-", "zorder": 10},
    "PredRNN": {"color": "#2563EB", "lw": 2.0, "ls": "-", "zorder": 9},
    "PFGNet": {"color": "#0F766E", "lw": 2.0, "ls": "-", "zorder": 8},
    "ConvLSTM": {"color": "#7C3AED", "lw": 1.9, "ls": "-", "zorder": 7},
    "PredRNN-V2": {"color": "#F97316", "lw": 1.9, "ls": "-", "zorder": 6},
    "ExPreCast": {"color": "#16A34A", "lw": 1.9, "ls": "-", "zorder": 5},
    "TimeKAN": {"color": "#D97706", "lw": 1.9, "ls": "-", "zorder": 4},
    "SeaCast": {"color": "#0891B2", "lw": 1.9, "ls": "-", "zorder": 3},
    "SwinLSTM": {"color": "#6B7280", "lw": 1.9, "ls": "-", "zorder": 2},
}


def load_train_99p(data_dir: Path) -> float:
    params_path = data_dir / "ais_norm_params.npy"
    if not params_path.exists():
        return 1.0
    params = np.load(params_path, allow_pickle=True).item()
    return float(params.get("train_99p", 1.0))


def normalize_spatiotemporal_array(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected [T, H, W] after squeeze, got {array.shape}")
    return array.astype(np.float64)


def expand_mask(mask_2d: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    mask = mask_2d.astype(bool)
    while mask.ndim < len(shape):
        mask = np.expand_dims(mask, axis=0)
    return np.broadcast_to(mask, shape)


def extract_valid_values(preds_path: Path, targets_path: Path, mask_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    preds = normalize_spatiotemporal_array(np.load(preds_path))
    targets = normalize_spatiotemporal_array(np.load(targets_path))
    if preds.shape != targets.shape:
        raise ValueError(f"Pred/target shape mismatch: {preds.shape} vs {targets.shape}")

    valid_mask = expand_mask(mask_2d, preds.shape)
    p = preds[valid_mask]
    t = targets[valid_mask]
    valid = np.isfinite(p) & np.isfinite(t)
    return p[valid], t[valid]


def compute_binary_metrics(pred_values: np.ndarray, target_values: np.ndarray, threshold: float) -> Dict[str, float]:
    pred_bin = pred_values >= threshold
    target_bin = target_values >= threshold
    tp = np.sum(pred_bin & target_bin)
    fp = np.sum(pred_bin & ~target_bin)
    fn = np.sum(~pred_bin & target_bin)
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-8))
    csi = float(tp / (tp + fp + fn + 1e-8))
    return {
        "CSI": csi,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
    }


def aggregate_metric_dicts(rows: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for metric_name in METRICS:
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
        means[metric_name] = float(np.mean(values))
        stds[metric_name] = float(np.std(values, ddof=0))
    return means, stds


def sort_model_rows(rows: List[Dict]) -> List[Dict]:
    order_map = {model_dir: idx for idx, model_dir in enumerate(MODEL_ORDER)}
    return sorted(rows, key=lambda row: (order_map.get(row["model_dir"], 999), row["model_label"]))


def collect_available_models(model_outcomes_dir: Path, mask_path: Path, focus_threshold: float) -> List[Dict]:
    rows, _ = load_checkpoint_rows(
        model_outcomes_dir=model_outcomes_dir,
        mask_path=mask_path,
        threshold=focus_threshold,
        return_skipped=True,
    )
    filtered_rows = []
    for row in rows:
        if row["model_dir"] not in MODEL_ORDER:
            continue
        row = dict(row)
        if row["model_dir"] in LABEL_OVERRIDES:
            row["model_label"] = LABEL_OVERRIDES[row["model_dir"]]
        filtered_rows.append(row)
    return sort_model_rows(filtered_rows)


def collect_threshold_curve_for_split(
    checkpoint_dir: Path,
    split_name: str,
    mask_2d: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[List[Dict[str, float]], str]:
    seed_dirs = sorted(path for path in checkpoint_dir.glob("seed_*") if path.is_dir())
    per_seed_values: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for seed_dir in seed_dirs:
        preds_path = seed_dir / f"{split_name}_preds.npy"
        targets_path = seed_dir / f"{split_name}_targets.npy"
        if not (preds_path.exists() and targets_path.exists()):
            continue
        pred_values, target_values = extract_valid_values(preds_path, targets_path, mask_2d)
        per_seed_values.append((seed_dir.name, pred_values, target_values))

    source: str
    if per_seed_values:
        source = f"seed-level mean±std from {len(per_seed_values)} seed dirs"
    else:
        preds_path = checkpoint_dir / f"{split_name}_preds.npy"
        targets_path = checkpoint_dir / f"{split_name}_targets.npy"
        if not (preds_path.exists() and targets_path.exists()):
            raise FileNotFoundError(
                f"Missing predictions for {checkpoint_dir.name}/{split_name}: "
                f"{preds_path.name}, {targets_path.name}"
            )
        pred_values, target_values = extract_valid_values(preds_path, targets_path, mask_2d)
        per_seed_values = [("root", pred_values, target_values)]
        source = "root aggregated preds/targets"

    curve_rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        metric_rows = []
        for _, pred_values, target_values in per_seed_values:
            metric_rows.append(compute_binary_metrics(pred_values, target_values, float(threshold)))
        metric_means, metric_stds = aggregate_metric_dicts(metric_rows)
        row: Dict[str, float] = {
            "threshold_norm": float(threshold),
        }
        for metric_name in METRICS:
            row[f"{metric_name}_mean"] = metric_means[metric_name]
            row[f"{metric_name}_std"] = metric_stds[metric_name]
        curve_rows.append(row)
    return curve_rows, source


def write_curve_csv(output_path: Path, curve_payloads: Sequence[Dict], train_99p: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_dir",
        "model_label",
        "split_name",
        "split_label",
        "threshold_norm",
        "threshold_hours_per_day",
        "CSI_mean",
        "CSI_std",
        "F1_mean",
        "F1_std",
        "Precision_mean",
        "Precision_std",
        "Recall_mean",
        "Recall_std",
        "metric_source",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for payload in curve_payloads:
            for row in payload["curve_rows"]:
                writer.writerow(
                    {
                        "model_dir": payload["model_dir"],
                        "model_label": payload["model_label"],
                        "split_name": payload["split_name"],
                        "split_label": payload["split_label"],
                        "threshold_norm": row["threshold_norm"],
                        "threshold_hours_per_day": row["threshold_norm"] * train_99p,
                        "CSI_mean": row["CSI_mean"],
                        "CSI_std": row["CSI_std"],
                        "F1_mean": row["F1_mean"],
                        "F1_std": row["F1_std"],
                        "Precision_mean": row["Precision_mean"],
                        "Precision_std": row["Precision_std"],
                        "Recall_mean": row["Recall_mean"],
                        "Recall_std": row["Recall_std"],
                        "metric_source": payload["metric_source"],
                    }
                )


def write_best_thresholds_json(output_path: Path, curve_payloads: Sequence[Dict], train_99p: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for payload in curve_payloads:
        split_bucket = result.setdefault(payload["split_name"], {})
        model_bucket: Dict[str, Dict[str, float]] = {
            "_meta": {
                "model_dir": payload["model_dir"],
                "source": payload["metric_source"],
            }
        }
        curve_rows = payload["curve_rows"]
        for metric_name in METRICS:
            best_row = max(curve_rows, key=lambda row: row[f"{metric_name}_mean"])
            model_bucket[metric_name] = {
                "threshold_norm": float(best_row["threshold_norm"]),
                "threshold_hours_per_day": float(best_row["threshold_norm"] * train_99p),
                "score_mean": float(best_row[f"{metric_name}_mean"]),
                "score_std": float(best_row[f"{metric_name}_std"]),
            }
        split_bucket[payload["model_label"]] = model_bucket

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def figure_height_for_model_count(model_count: int) -> float:
    return max(9.6, 8.5 + model_count * 0.20)


def plot_split_figure(
    curve_payloads: Sequence[Dict],
    split_name: str,
    split_label: str,
    output_prefix: Path,
    focus_threshold: float,
    train_99p: float,
) -> None:
    split_rows = [payload for payload in curve_payloads if payload["split_name"] == split_name]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, figure_height_for_model_count(len(split_rows))), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.24, hspace=0.34, wspace=0.24)
    axes = axes.ravel()
    panel_labels = ("(a)", "(b)", "(c)", "(d)")

    for idx, metric_name in enumerate(METRICS):
        ax = axes[idx]
        for payload in split_rows:
            style = MODEL_STYLES.get(payload["model_label"], {"color": "#111827", "lw": 1.8, "ls": "-", "zorder": 1})
            x = [row["threshold_norm"] for row in payload["curve_rows"]]
            y = [row[f"{metric_name}_mean"] for row in payload["curve_rows"]]
            ax.plot(
                x,
                y,
                label=payload["model_label"],
                color=style["color"],
                linewidth=style["lw"],
                linestyle=style["ls"],
                zorder=style["zorder"],
            )

        ax.axvline(focus_threshold, color="#374151", linestyle=":", linewidth=1.5, alpha=0.9)
        if metric_name == "CSI":
            ax.text(
                focus_threshold + 0.006,
                0.04,
                f"Selected threshold = {focus_threshold:.4f}",
                rotation=90,
                fontsize=9.5,
                color="#374151",
                va="bottom",
            )

        ax.set_xlim(
            float(split_rows[0]["curve_rows"][0]["threshold_norm"]),
            float(split_rows[0]["curve_rows"][-1]["threshold_norm"]),
        )
        ax.set_ylim(0.0, 1.02)
        ax.set_title(metric_name, fontsize=15, fontweight="bold", pad=10)
        ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.6)
        ax.text(-0.12, 1.05, panel_labels[idx], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

        if idx % 2 == 0:
            ax.set_ylabel("Metric Score", fontsize=12)
        if idx >= 2:
            ax.set_xlabel("Normalized Threshold", fontsize=12)

        if idx < 2:
            secax = ax.secondary_xaxis(
                "top",
                functions=(lambda x: x * train_99p, lambda x: x / max(train_99p, 1e-8)),
            )
            secax.set_xlabel("Fishing Intensity (hours/day)", fontsize=10.5, labelpad=9)
            secax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.085),
        fontsize=10.5,
        columnspacing=1.8,
        handlelength=2.3,
    )
    fig.suptitle(
        f"Threshold Sensitivity Analysis for {split_label} (9 Models)",
        fontsize=19,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.08,
        0.012,
        "Curves show seed-level mean scores when seed_* predictions are available; otherwise root aggregated predictions are used. "
        f"The vertical dotted line marks the selected threshold = {focus_threshold:.4f}.",
        fontsize=10,
        color="#374151",
        ha="left",
        va="bottom",
        wrap=True,
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_archive_tag(focus_threshold: float, model_count: int) -> str:
    return f"threshold_{threshold_tag(focus_threshold)}_{model_count}models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def copy_latest(archive_path: Path, latest_path: Path) -> None:
    if archive_path == latest_path:
        return
    shutil.copy2(archive_path, latest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run threshold sensitivity analysis for the 9 models included in the ranking figure."
    )
    parser.add_argument(
        "--model-outcomes-dir",
        type=str,
        default=str(MODEL_OUTCOMES_DIR),
        help="Root directory containing model checkpoint result folders.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        default=str(MASK_PATH),
        help="Shared ocean mask used to evaluate hotspot metrics.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Data directory containing ais_norm_params.npy for physical-unit conversion.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory used to store threshold sweep figures and tables.",
    )
    parser.add_argument(
        "--focus-threshold",
        type=float,
        default=THRESHOLD,
        help="Selected threshold highlighted in the plots.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.10,
        help="Minimum normalized threshold to scan.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.60,
        help="Maximum normalized threshold to scan.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=101,
        help="Number of threshold points scanned between min and max.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_outcomes_dir = Path(args.model_outcomes_dir)
    mask_path = Path(args.mask_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not model_outcomes_dir.is_absolute():
        model_outcomes_dir = (PROJECT_ROOT / model_outcomes_dir).resolve()
    if not mask_path.is_absolute():
        mask_path = (PROJECT_ROOT / mask_path).resolve()
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    if args.num_points < 2:
        raise ValueError("--num-points must be at least 2")
    if not 0.0 <= args.threshold_min < args.threshold_max <= 1.0:
        raise ValueError("Threshold range must satisfy 0 <= min < max <= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_2d = np.load(mask_path).astype(bool)
    train_99p = load_train_99p(data_dir)
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.num_points, dtype=np.float64)
    thresholds = np.unique(np.sort(np.append(thresholds, np.float64(args.focus_threshold))))

    model_rows = collect_available_models(model_outcomes_dir, mask_path, args.focus_threshold)
    if len(model_rows) != 9:
        raise RuntimeError(f"Expected 9 ranking models, found {len(model_rows)}: {[row['model_dir'] for row in model_rows]}")

    curve_payloads: List[Dict] = []
    for row in model_rows:
        checkpoint_dir = model_outcomes_dir / row["model_dir"]
        for split_name, split_label in SPLITS:
            curve_rows, metric_source = collect_threshold_curve_for_split(
                checkpoint_dir=checkpoint_dir,
                split_name=split_name,
                mask_2d=mask_2d,
                thresholds=thresholds,
            )
            curve_payloads.append(
                {
                    "model_dir": row["model_dir"],
                    "model_label": row["model_label"],
                    "split_name": split_name,
                    "split_label": split_label,
                    "metric_source": metric_source,
                    "curve_rows": curve_rows,
                }
            )

    archive_tag = build_archive_tag(args.focus_threshold, len(model_rows))
    csv_archive = output_dir / f"SCI_Threshold_Sweep_All_Models_{archive_tag}.csv"
    csv_latest = output_dir / f"SCI_Threshold_Sweep_All_Models_latest_threshold_{threshold_tag(args.focus_threshold)}.csv"
    json_archive = output_dir / f"SCI_Threshold_Sweep_Best_Thresholds_{archive_tag}.json"
    json_latest = output_dir / f"SCI_Threshold_Sweep_Best_Thresholds_latest_threshold_{threshold_tag(args.focus_threshold)}.json"

    write_curve_csv(csv_archive, curve_payloads, train_99p)
    write_best_thresholds_json(json_archive, curve_payloads, train_99p)
    copy_latest(csv_archive, csv_latest)
    copy_latest(json_archive, json_latest)

    split_name_to_prefix = {
        "test_one_step": output_dir / f"SCI_Threshold_Sweep_OneStep_All_Models_{archive_tag}",
        "rollout_2024": output_dir / f"SCI_Threshold_Sweep_Rollout2024_All_Models_{archive_tag}",
    }
    split_name_to_latest = {
        "test_one_step": output_dir / f"SCI_Threshold_Sweep_OneStep_All_Models_latest_threshold_{threshold_tag(args.focus_threshold)}",
        "rollout_2024": output_dir / f"SCI_Threshold_Sweep_Rollout2024_All_Models_latest_threshold_{threshold_tag(args.focus_threshold)}",
    }

    for split_name, split_label in SPLITS:
        archive_prefix = split_name_to_prefix[split_name]
        latest_prefix = split_name_to_latest[split_name]
        plot_split_figure(
            curve_payloads=curve_payloads,
            split_name=split_name,
            split_label=split_label,
            output_prefix=archive_prefix,
            focus_threshold=args.focus_threshold,
            train_99p=train_99p,
        )
        for suffix in (".png", ".pdf"):
            copy_latest(archive_prefix.with_suffix(suffix), latest_prefix.with_suffix(suffix))

    print(f"Included models: {len(model_rows)}")
    for row in model_rows:
        print(f"  - {row['model_label']} ({row['model_dir']})")
    print(f"Saved sweep CSV to: {csv_archive}")
    print(f"Updated latest CSV to: {csv_latest}")
    print(f"Saved best-threshold JSON to: {json_archive}")
    print(f"Updated latest JSON to: {json_latest}")
    for split_name, _ in SPLITS:
        archive_prefix = split_name_to_prefix[split_name]
        latest_prefix = split_name_to_latest[split_name]
        print(f"Saved figure archive to: {archive_prefix.with_suffix('.png')}")
        print(f"Saved figure archive to: {archive_prefix.with_suffix('.pdf')}")
        print(f"Updated latest figure to: {latest_prefix.with_suffix('.png')}")
        print(f"Updated latest figure to: {latest_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
