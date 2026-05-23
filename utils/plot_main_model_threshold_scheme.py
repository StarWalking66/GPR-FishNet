from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
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


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ST_FishNet_Features"
DEFAULT_MASK_PATH = DEFAULT_DATA_DIR / "all_vars_train_mask_intersection.npy"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_outcomes" / "checkpoints_gpr_fishnet_final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model_outcomes" / "threshold_scheme_main_model"
METRICS = ("CSI", "F1", "Precision", "Recall")
CSI_COLOR = "#8B0000"
F1_COLOR = "#00008B"
TOP_RATIO_COLOR = "#0F766E"
OBSERVED_COLOR = "#6B7280"
PREDICTED_COLOR = "#0F766E"
BAR_EDGE_COLOR = "#111827"
BAR_VALUE_COLOR = "#374151"
THRESHOLD_LINE_STYLES = {
    "fixed_reference": {"color": "#4B5563", "linestyle": ":", "linewidth": 1.7},
    "main_threshold": {"color": "#0072B2", "linestyle": "--", "linewidth": 2.1},
    "otsu_threshold": {"color": "#CC79A7", "linestyle": "-.", "linewidth": 1.7},
}
TOP_RATIO_LINESTYLES = [
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
    (0, (5, 2, 1, 2)),
]
PNG_DPI = 600
MIN_FULL_WIDTH_PNG_PX = 3740


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["axes.linewidth"] = 1.2
rcParams["xtick.major.width"] = 1.2
rcParams["ytick.major.width"] = 1.2
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["axes.labelweight"] = "bold"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


@dataclass(frozen=True)
class ThresholdScheme:
    name: str
    label: str
    kind: str
    threshold_norm: float
    validation_target_ratio: float


def threshold_tag(threshold: float) -> str:
    return f"{threshold:.10f}".rstrip("0").rstrip(".").replace("-", "neg_").replace(".", "p")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_train_99p(data_dir: Path) -> float:
    params_path = data_dir / "ais_norm_params.npy"
    if not params_path.exists():
        return 1.0
    params = np.load(params_path, allow_pickle=True).item()
    return float(params.get("train_99p", 1.0))


def load_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    return np.load(mask_path).astype(bool)


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


def extract_valid_values_from_arrays(
    preds: np.ndarray,
    targets: np.ndarray,
    mask_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    preds_3d = normalize_spatiotemporal_array(preds)
    targets_3d = normalize_spatiotemporal_array(targets)
    if preds_3d.shape != targets_3d.shape:
        raise ValueError(f"Pred/target shape mismatch: {preds_3d.shape} vs {targets_3d.shape}")
    valid_mask = expand_mask(mask_2d, preds_3d.shape)
    pred_values = preds_3d[valid_mask]
    target_values = targets_3d[valid_mask]
    valid = np.isfinite(pred_values) & np.isfinite(target_values)
    return pred_values[valid], target_values[valid]


def extract_valid_values_from_paths(
    preds_path: Path,
    targets_path: Path,
    mask_2d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return extract_valid_values_from_arrays(np.load(preds_path), np.load(targets_path), mask_2d)


def load_validation_values(data_dir: Path, mask_2d: np.ndarray) -> np.ndarray:
    val_path = data_dir / "ais_val.npy"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation AIS file not found: {val_path}")
    values = normalize_spatiotemporal_array(np.load(val_path))
    valid_mask = expand_mask(mask_2d, values.shape)
    valid_values = values[valid_mask]
    valid_values = valid_values[np.isfinite(valid_values)]
    return valid_values.astype(np.float64)


def compute_otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weights = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    omega = np.cumsum(weights)
    mu = np.cumsum(weights * centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    between = np.zeros_like(centers)
    valid = denom > 0
    between[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denom[valid]
    if between.size > 1:
        between[-1] = 0.0
    return float(centers[int(np.argmax(between))])


def threshold_from_top_ratio(values: np.ndarray, ratio: float) -> float:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"Top ratio must be in (0, 1), got {ratio}")
    return float(np.quantile(values, 1.0 - ratio))


def compute_binary_metrics(pred_values: np.ndarray, target_values: np.ndarray, threshold: float) -> Dict[str, float]:
    pred_bin = pred_values >= threshold
    target_bin = target_values >= threshold
    tp = float(np.sum(pred_bin & target_bin))
    fp = float(np.sum(pred_bin & ~target_bin))
    fn = float(np.sum(~pred_bin & target_bin))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    csi = tp / (tp + fp + fn + 1e-8)
    return {
        "CSI": float(csi),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "pred_hotspot_ratio": float(np.mean(pred_bin)),
        "target_hotspot_ratio": float(np.mean(target_bin)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aggregate_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name in (*METRICS, "pred_hotspot_ratio", "target_hotspot_ratio"):
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values, ddof=0))
    return result


def collect_model_runs(model_dir: Path, split_name: str, mask_2d: np.ndarray) -> Tuple[List[Dict], str]:
    runs: List[Dict] = []
    seed_dirs = sorted(path for path in model_dir.glob("seed_*") if path.is_dir())
    for seed_dir in seed_dirs:
        preds_path = seed_dir / f"{split_name}_preds.npy"
        targets_path = seed_dir / f"{split_name}_targets.npy"
        if not (preds_path.exists() and targets_path.exists()):
            continue
        pred_values, target_values = extract_valid_values_from_paths(preds_path, targets_path, mask_2d)
        runs.append(
            {
                "run_name": seed_dir.name,
                "pred_values": pred_values,
                "target_values": target_values,
                "preds_path": str(preds_path),
                "targets_path": str(targets_path),
            }
        )

    if runs:
        return runs, f"seed-level mean/std from {len(runs)} random seeds"

    preds_path = model_dir / f"{split_name}_preds.npy"
    targets_path = model_dir / f"{split_name}_targets.npy"
    if not (preds_path.exists() and targets_path.exists()):
        raise FileNotFoundError(f"Missing {split_name}_preds.npy or {split_name}_targets.npy under {model_dir}")
    pred_values, target_values = extract_valid_values_from_paths(preds_path, targets_path, mask_2d)
    return [
        {
            "run_name": "root",
            "pred_values": pred_values,
            "target_values": target_values,
            "preds_path": str(preds_path),
            "targets_path": str(targets_path),
        }
    ], "root aggregated predictions"


def build_threshold_schemes(
    val_values: np.ndarray,
    reference_threshold: float,
    main_threshold: float,
    otsu_threshold: float,
    top_ratios: Sequence[float],
) -> List[ThresholdScheme]:
    schemes = [
        ThresholdScheme(
            name="reference_0p2755",
            label=f"Reference\n{reference_threshold:.4f}",
            kind="fixed_reference",
            threshold_norm=float(reference_threshold),
            validation_target_ratio=float(np.mean(val_values >= reference_threshold)),
        ),
        ThresholdScheme(
            name="main_top20_0p3175",
            label=f"Main / top 20%\n{main_threshold:.4f}",
            kind="main_threshold",
            threshold_norm=float(main_threshold),
            validation_target_ratio=float(np.mean(val_values >= main_threshold)),
        ),
        ThresholdScheme(
            name="otsu_0p3965",
            label=f"Otsu\n{otsu_threshold:.4f}",
            kind="otsu_threshold",
            threshold_norm=float(otsu_threshold),
            validation_target_ratio=float(np.mean(val_values >= otsu_threshold)),
        ),
    ]
    for ratio in top_ratios:
        if abs(ratio - 0.20) < 1e-9:
            continue
        threshold = threshold_from_top_ratio(val_values, ratio)
        schemes.append(
            ThresholdScheme(
                name=f"top_{int(round(ratio * 100)):02d}pct",
                label=f"Top {int(round(ratio * 100))}%\n{threshold:.4f}",
                kind="top_ratio_sensitivity",
                threshold_norm=float(threshold),
                validation_target_ratio=float(np.mean(val_values >= threshold)),
            )
        )
    schemes.sort(key=lambda item: (item.threshold_norm, item.kind != "main_threshold", item.label))
    return schemes


def scan_thresholds(
    runs: Sequence[Dict],
    thresholds: np.ndarray,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        run_metric_rows = [
            compute_binary_metrics(run["pred_values"], run["target_values"], float(threshold))
            for run in runs
        ]
        aggregate = aggregate_rows(run_metric_rows)
        rows.append({"threshold_norm": float(threshold), **aggregate})
    return rows


def evaluate_schemes(
    runs: Sequence[Dict],
    schemes: Sequence[ThresholdScheme],
    train_99p: float,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for scheme in schemes:
        run_metric_rows = [
            compute_binary_metrics(run["pred_values"], run["target_values"], scheme.threshold_norm)
            for run in runs
        ]
        aggregate = aggregate_rows(run_metric_rows)
        rows.append(
            {
                "scheme_name": scheme.name,
                "scheme_label": scheme.label.replace("\n", " "),
                "scheme_kind": scheme.kind,
                "threshold_norm": scheme.threshold_norm,
                "threshold_hours_per_day": scheme.threshold_norm * train_99p,
                "validation_target_hotspot_ratio": scheme.validation_target_ratio,
                **aggregate,
            }
        )
    return rows


def write_scan_csv(path: Path, rows: Sequence[Dict[str, float]], train_99p: float, split_name: str) -> None:
    fieldnames = [
        "split_name",
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
        "pred_hotspot_ratio_mean",
        "pred_hotspot_ratio_std",
        "target_hotspot_ratio_mean",
        "target_hotspot_ratio_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split_name": split_name,
                    "threshold_hours_per_day": row["threshold_norm"] * train_99p,
                    **{key: row[key] for key in fieldnames if key in row},
                }
            )


def write_scheme_csv(path: Path, rows: Sequence[Dict[str, float | str]]) -> None:
    fieldnames = [
        "scheme_name",
        "scheme_label",
        "scheme_kind",
        "threshold_norm",
        "threshold_hours_per_day",
        "validation_target_hotspot_ratio",
        "CSI_mean",
        "CSI_std",
        "F1_mean",
        "F1_std",
        "Precision_mean",
        "Precision_std",
        "Recall_mean",
        "Recall_std",
        "pred_hotspot_ratio_mean",
        "pred_hotspot_ratio_std",
        "target_hotspot_ratio_mean",
        "target_hotspot_ratio_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def copy_latest(archive_path: Path, latest_path: Path) -> None:
    if archive_path == latest_path:
        return
    shutil.copy2(archive_path, latest_path)


def format_scheme_short_label(scheme: ThresholdScheme) -> str:
    threshold = scheme.threshold_norm
    if scheme.kind == "fixed_reference":
        return f"ref. ({threshold:.2f})"
    if scheme.kind == "main_threshold":
        return f"top 20% ({threshold:.2f})"
    if scheme.kind == "otsu_threshold":
        return f"Otsu ({threshold:.2f})"
    if scheme.kind == "top_ratio_sensitivity":
        parts = scheme.label.replace("\n", " ").split()
        percent = parts[1] if len(parts) > 1 else "ratio"
        return f"top {percent} ({threshold:.2f})"
    return f"{scheme.name} ({threshold:.2f})"


def add_threshold_lines(
    ax: plt.Axes,
    schemes: Sequence[ThresholdScheme],
    include_top_sensitivity: bool = False,
) -> None:
    labels_seen = set()
    top_ratio_index = 0
    for scheme in schemes:
        if scheme.kind == "top_ratio_sensitivity" and not include_top_sensitivity:
            continue
        if scheme.kind == "top_ratio_sensitivity":
            style = {
                "color": TOP_RATIO_COLOR,
                "linestyle": TOP_RATIO_LINESTYLES[top_ratio_index % len(TOP_RATIO_LINESTYLES)],
                "linewidth": 1.45,
            }
            top_ratio_index += 1
        else:
            style = THRESHOLD_LINE_STYLES.get(
                scheme.kind,
                {"color": TOP_RATIO_COLOR, "linestyle": (0, (2, 2)), "linewidth": 1.35},
            )
        label = format_scheme_short_label(scheme)
        if label in labels_seen:
            label = None
        else:
            labels_seen.add(label)
        ax.axvline(scheme.threshold_norm, alpha=0.9, label=label, zorder=3, **style)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.082,
        1.006,
        label,
        transform=ax.transAxes,
        fontsize=15.0,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="bottom",
        zorder=20,
    )


def annotate_bar_values(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    errors: np.ndarray,
    y_offset: float,
    fmt: str = "{:.2f}",
) -> None:
    for bar, value, error in zip(bars, values, errors):
        if not np.isfinite(value):
            continue
        error_value = float(error) if np.isfinite(error) else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + error_value + y_offset,
            fmt.format(float(value)),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.4,
            color=BAR_VALUE_COLOR,
            clip_on=False,
        )


def tight_png_width_px(fig: plt.Figure, dpi: int) -> int:
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    return int(round(bbox.width * dpi))


def plot_figure(
    output_prefix: Path,
    val_values: np.ndarray,
    scan_rows: Sequence[Dict[str, float]],
    scheme_rows: Sequence[Dict[str, float | str]],
    schemes: Sequence[ThresholdScheme],
    train_99p: float,
    scan_min: float,
    scan_max: float,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.13, hspace=0.34, wspace=0.28)

    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for ax, label in zip(axes.ravel(), panel_labels):
        add_panel_label(ax, label)

    # Panel (a): validation-derived hotspot area ratio.
    ax = axes[0, 0]
    area_thresholds = np.linspace(0.0, 1.0, 401)
    sorted_values = np.sort(val_values)
    counts = sorted_values.size - np.searchsorted(sorted_values, area_thresholds, side="left")
    area_ratios = counts / max(sorted_values.size, 1)
    ax.plot(area_thresholds, area_ratios, color="#C44E52", linewidth=2.4)
    for scheme in schemes:
        if scheme.kind == "top_ratio_sensitivity":
            ax.scatter(
                scheme.threshold_norm,
                scheme.validation_target_ratio,
                s=36,
                color=TOP_RATIO_COLOR,
                zorder=4,
            )
            label_offset = (-10, 14) if "30%" in scheme.label else (10, 14)
            ha = "right" if "30%" in scheme.label else "left"
            ax.annotate(
                scheme.label.split("\n")[0],
                xy=(scheme.threshold_norm, scheme.validation_target_ratio),
                xytext=label_offset,
                textcoords="offset points",
                fontsize=8.5,
                color=TOP_RATIO_COLOR,
                ha=ha,
                va="bottom",
            )
    add_threshold_lines(ax, schemes)
    ax.set_xlabel("Normalized threshold", fontsize=9.8, labelpad=0)
    ax.xaxis.set_label_coords(0.5, -0.055)
    ax.set_ylabel("Validation hotspot-area ratio", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.6)
    secax = ax.secondary_xaxis("top", functions=(lambda x: x * train_99p, lambda x: x / max(train_99p, 1e-8)))
    secax.set_xlabel("Fishing intensity (h d$^{-1}$)", fontsize=9.5, labelpad=7)
    secax.tick_params(labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel (b): threshold metric sweep from main-model outputs.
    ax = axes[0, 1]
    x = np.asarray([row["threshold_norm"] for row in scan_rows], dtype=np.float64)
    for metric_name, color, width in (("CSI", CSI_COLOR, 2.5), ("F1", F1_COLOR, 2.5)):
        y = np.asarray([row[f"{metric_name}_mean"] for row in scan_rows], dtype=np.float64)
        ax.plot(x, y, color=color, linewidth=width, label=metric_name)
    add_threshold_lines(ax, schemes, include_top_sensitivity=True)
    ax.set_xlabel("Normalized threshold", fontsize=9.8, labelpad=0)
    ax.xaxis.set_label_coords(0.5, -0.055)
    ax.set_ylabel("Metric score", fontsize=11)
    top_line_max = max(
        (scheme.threshold_norm for scheme in schemes if scheme.kind == "top_ratio_sensitivity"),
        default=scan_max,
    )
    ax.set_xlim(scan_min, max(scan_max, top_line_max + 0.02))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.6)
    secax = ax.secondary_xaxis("top", functions=(lambda x: x * train_99p, lambda x: x / max(train_99p, 1e-8)))
    secax.set_xlabel("Fishing intensity (h d$^{-1}$)", fontsize=9.5, labelpad=7)
    secax.tick_params(labelsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    kept = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label not in seen:
            kept.append((handle, label))
            seen.add(label)
    ax.legend(
        [item[0] for item in kept],
        [item[1] for item in kept],
        frameon=False,
        fontsize=7.8,
        loc="upper right",
        ncol=2,
        columnspacing=1.2,
        handlelength=2.8,
    )

    # Panel (c): discrete scheme metrics.
    ax = axes[1, 0]
    labels = [
        format_scheme_short_label(
            ThresholdScheme(
                name=str(row["scheme_name"]),
                label=str(row["scheme_label"]),
                kind=str(row["scheme_kind"]),
                threshold_norm=float(row["threshold_norm"]),
                validation_target_ratio=float(row["validation_target_hotspot_ratio"]),
            )
        )
        for row in scheme_rows
    ]
    xpos = np.arange(len(scheme_rows), dtype=np.float64)
    width = 0.34
    csi = np.asarray([float(row["CSI_mean"]) for row in scheme_rows], dtype=np.float64)
    f1 = np.asarray([float(row["F1_mean"]) for row in scheme_rows], dtype=np.float64)
    csi_std = np.asarray([float(row["CSI_std"]) for row in scheme_rows], dtype=np.float64)
    f1_std = np.asarray([float(row["F1_std"]) for row in scheme_rows], dtype=np.float64)
    csi_bars = ax.bar(
        xpos - width / 2,
        csi,
        width=width,
        color=CSI_COLOR,
        alpha=0.88,
        label="CSI",
        yerr=csi_std,
        capsize=3,
        hatch="///",
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.65,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    f1_bars = ax.bar(
        xpos + width / 2,
        f1,
        width=width,
        color=F1_COLOR,
        alpha=0.88,
        label="F1",
        yerr=f1_std,
        capsize=3,
        hatch="\\\\\\",
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.65,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    annotate_bar_values(ax, csi_bars, csi, csi_std, y_offset=0.014)
    annotate_bar_values(ax, f1_bars, f1, f1_std, y_offset=0.014)
    ax.set_ylabel("Metric score", fontsize=11)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8.3, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # Panel (d): hotspot area ratio under each scheme.
    ax = axes[1, 1]
    target_ratio = np.asarray([float(row["target_hotspot_ratio_mean"]) for row in scheme_rows], dtype=np.float64)
    pred_ratio = np.asarray([float(row["pred_hotspot_ratio_mean"]) for row in scheme_rows], dtype=np.float64)
    target_std = np.asarray([float(row["target_hotspot_ratio_std"]) for row in scheme_rows], dtype=np.float64)
    pred_std = np.asarray([float(row["pred_hotspot_ratio_std"]) for row in scheme_rows], dtype=np.float64)
    observed_bars = ax.bar(
        xpos - width / 2,
        target_ratio,
        width=width,
        color=OBSERVED_COLOR,
        alpha=0.78,
        label="Observed hotspot ratio",
        yerr=target_std,
        capsize=3,
        hatch="...",
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.65,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    predicted_bars = ax.bar(
        xpos + width / 2,
        pred_ratio,
        width=width,
        color=PREDICTED_COLOR,
        alpha=0.86,
        label="Predicted hotspot ratio",
        yerr=pred_std,
        capsize=3,
        hatch="xx",
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.65,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
    )
    annotate_bar_values(ax, observed_bars, target_ratio, target_std, y_offset=0.010)
    annotate_bar_values(ax, predicted_bars, pred_ratio, pred_std, y_offset=0.010)
    ax.set_ylabel("Area ratio", fontsize=11)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8.3, rotation=28, ha="right", rotation_mode="anchor")
    ratio_top = max(
        0.45,
        float(np.max(target_ratio + target_std)) + 0.07,
        float(np.max(pred_ratio + pred_std)) + 0.07,
    )
    ax.set_ylim(0.0, min(1.0, ratio_top))
    ax.grid(axis="y", linestyle="--", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.6, loc="upper right")

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    png_width_px = tight_png_width_px(fig, PNG_DPI)
    if png_width_px < MIN_FULL_WIDTH_PNG_PX:
        raise ValueError(
            f"Tight PNG export would be {png_width_px}px wide; "
            f"expected at least {MIN_FULL_WIDTH_PNG_PX}px for full-width output."
        )
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_caption(
    path: Path,
    split_label: str,
    metric_source: str,
    train_99p: float,
) -> None:
    caption = (
        f"Threshold-scheme analysis for GPR-FishNet ({split_label}). "
        "(a) Validation hotspot-area ratio as a function of normalized threshold. "
        "(b) Critical success index (CSI) and F1 score as functions of normalized threshold; "
        "the top 30%, reference, top 20%, Otsu and top 10% thresholds are marked by vertical lines. "
        "(c) CSI and F1 across the five discrete threshold schemes. "
        "(d) Observed and predicted hotspot-area ratios across the same schemes. "
        "Threshold definitions use validation-set observed fishing-effort values, whereas metric curves "
        "and bars use GPR-FishNet outputs. Upper x-axes are aligned using "
        f"c99 = {train_99p:.1f} h d\u207b\u00b9, so a normalized threshold of 1.0 corresponds to c99. "
        f"Metrics are reported as {metric_source.replace('mean/std', 'mean and standard deviation')}."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create threshold-scheme figures from GPR-FishNet prediction outputs."
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--mask-path", type=str, default=str(DEFAULT_MASK_PATH))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--split",
        type=str,
        default="rollout_2024",
        choices=("rollout_2024", "test_one_step"),
        help="Prediction split used for metric curves and scheme comparisons.",
    )
    parser.add_argument("--reference-threshold", type=float, default=0.2755)
    parser.add_argument("--main-threshold", type=float, default=0.3175)
    parser.add_argument(
        "--otsu-threshold",
        type=float,
        default=None,
        help="Optional fixed Otsu threshold. If omitted, it is computed from validation AIS values.",
    )
    parser.add_argument("--top-ratios", type=str, default="0.10,0.20,0.30")
    parser.add_argument("--scan-min", type=float, default=0.10)
    parser.add_argument("--scan-max", type=float, default=0.60)
    parser.add_argument("--scan-points", type=int, default=101)
    parser.add_argument("--otsu-bins", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    mask_path = resolve_path(args.mask_path)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)

    if not 0.0 <= args.scan_min < args.scan_max <= 1.0:
        raise ValueError("--scan-min and --scan-max must satisfy 0 <= min < max <= 1")
    if args.scan_points < 2:
        raise ValueError("--scan-points must be at least 2")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_2d = load_mask(mask_path)
    train_99p = load_train_99p(data_dir)
    val_values = load_validation_values(data_dir, mask_2d)
    top_ratios = [float(item.strip()) for item in args.top_ratios.split(",") if item.strip()]
    otsu_threshold = float(args.otsu_threshold) if args.otsu_threshold is not None else compute_otsu_threshold(
        val_values,
        bins=args.otsu_bins,
    )
    schemes = build_threshold_schemes(
        val_values=val_values,
        reference_threshold=args.reference_threshold,
        main_threshold=args.main_threshold,
        otsu_threshold=otsu_threshold,
        top_ratios=top_ratios,
    )

    runs, metric_source = collect_model_runs(model_dir, args.split, mask_2d)
    scan_thresholds_array = np.linspace(args.scan_min, args.scan_max, args.scan_points, dtype=np.float64)
    scan_thresholds_array = np.unique(np.sort(np.append(scan_thresholds_array, [args.main_threshold])))
    scan_rows = scan_thresholds(runs, scan_thresholds_array)
    scheme_rows = evaluate_schemes(runs, schemes, train_99p)

    split_label = "2024 Rollout Forecasting" if args.split == "rollout_2024" else "One-step Forecasting"
    archive_tag = f"{args.split}_threshold_{threshold_tag(args.main_threshold)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_prefix = output_dir / f"GPR_FishNet_Threshold_Scheme_{archive_tag}"
    latest_prefix = output_dir / f"GPR_FishNet_Threshold_Scheme_latest_{args.split}_threshold_{threshold_tag(args.main_threshold)}"

    scan_csv_archive = output_dir / f"GPR_FishNet_Threshold_Scheme_Scan_{archive_tag}.csv"
    scan_csv_latest = output_dir / f"GPR_FishNet_Threshold_Scheme_Scan_latest_{args.split}_threshold_{threshold_tag(args.main_threshold)}.csv"
    scheme_csv_archive = output_dir / f"GPR_FishNet_Threshold_Scheme_Discrete_{archive_tag}.csv"
    scheme_csv_latest = output_dir / f"GPR_FishNet_Threshold_Scheme_Discrete_latest_{args.split}_threshold_{threshold_tag(args.main_threshold)}.csv"
    summary_json_archive = output_dir / f"GPR_FishNet_Threshold_Scheme_Summary_{archive_tag}.json"
    summary_json_latest = output_dir / f"GPR_FishNet_Threshold_Scheme_Summary_latest_{args.split}_threshold_{threshold_tag(args.main_threshold)}.json"
    caption_archive = output_dir / f"GPR_FishNet_Threshold_Scheme_Caption_{archive_tag}.txt"
    caption_latest = output_dir / f"GPR_FishNet_Threshold_Scheme_Caption_latest_{args.split}_threshold_{threshold_tag(args.main_threshold)}.txt"

    write_scan_csv(scan_csv_archive, scan_rows, train_99p, args.split)
    write_scheme_csv(scheme_csv_archive, scheme_rows)
    plot_figure(
        output_prefix=archive_prefix,
        val_values=val_values,
        scan_rows=scan_rows,
        scheme_rows=scheme_rows,
        schemes=schemes,
        train_99p=train_99p,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
    )
    write_caption(caption_archive, split_label, metric_source, train_99p)

    summary = {
        "split": args.split,
        "split_label": split_label,
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "mask_path": str(mask_path),
        "train_99p_hours_per_day": train_99p,
        "metric_source": metric_source,
        "num_runs": len(runs),
        "thresholds": [
            {
                "name": scheme.name,
                "label": scheme.label.replace("\n", " "),
                "kind": scheme.kind,
                "threshold_norm": scheme.threshold_norm,
                "threshold_hours_per_day": scheme.threshold_norm * train_99p,
                "validation_target_hotspot_ratio": scheme.validation_target_ratio,
            }
            for scheme in schemes
        ],
        "scheme_metrics": scheme_rows,
        "outputs": {
            "figure_png": str(archive_prefix.with_suffix(".png")),
            "figure_pdf": str(archive_prefix.with_suffix(".pdf")),
            "scan_csv": str(scan_csv_archive),
            "scheme_csv": str(scheme_csv_archive),
            "caption": str(caption_archive),
        },
    }
    with summary_json_archive.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    for suffix in (".png", ".pdf"):
        copy_latest(archive_prefix.with_suffix(suffix), latest_prefix.with_suffix(suffix))
    copy_latest(scan_csv_archive, scan_csv_latest)
    copy_latest(scheme_csv_archive, scheme_csv_latest)
    copy_latest(summary_json_archive, summary_json_latest)
    copy_latest(caption_archive, caption_latest)

    print(f"Metric source: {metric_source}")
    print(f"Train 99p: {train_99p:.6f} h d^-1")
    for scheme in schemes:
        print(
            f"{scheme.label.replace(chr(10), ' ')}: "
            f"threshold={scheme.threshold_norm:.4f}, "
            f"h d^-1={scheme.threshold_norm * train_99p:.2f}, "
            f"val hotspot ratio={scheme.validation_target_ratio:.4f}"
        )
    print(f"Saved figure: {archive_prefix.with_suffix('.png')}")
    print(f"Updated latest figure: {latest_prefix.with_suffix('.png')}")
    print(f"Saved discrete CSV: {scheme_csv_archive}")
    print(f"Saved scan CSV: {scan_csv_archive}")
    print(f"Saved summary JSON: {summary_json_archive}")
    print(f"Saved caption: {caption_archive}")


if __name__ == "__main__":
    main()
