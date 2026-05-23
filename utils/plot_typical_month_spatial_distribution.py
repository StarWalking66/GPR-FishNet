import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.generate_model_ranking_xlsx import (  # noqa: E402
    CHECKPOINT_GLOB,
    CONTINUOUS_METRICS,
    MASK_PATH,
    MODEL_OUTCOMES_DIR,
    THRESHOLD,
    add_average_ranks,
    assign_ranks,
    build_artifact_tag,
    build_threshold_metric_specs,
    infer_model_label,
    load_checkpoint_rows,
    threshold_metric_key,
    threshold_tag,
)


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


OUTPUT_DIR = PROJECT_ROOT / "model_outcomes" / "typical_month_spatial_distribution"
DEFAULT_OURS_MODEL_DIR = "checkpoints_gpr_fishnet_final"
DEFAULT_LABEL_OVERRIDES = {
    DEFAULT_OURS_MODEL_DIR: "GPR-FishNet (ours)",
}
DEFAULT_EFFORT_CMAP = "cividis"
ALLOWED_EFFORT_CMAPS = {"cividis", "viridis", "turbo"}
LEGACY_EFFORT_CMAPS = {"jet", "rainbow"}
ERROR_CENTER_COLOR = "#ffffff"
MIN_EXPORT_DPI = 600
MIN_EXPORT_WIDTH_PX = 3740
OURS_LABEL_COLOR = "#be123c"
DEFAULT_LABEL_COLOR = "#111827"
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def load_selected_seed(checkpoint_dir: Path) -> int:
    selected_seed_path = checkpoint_dir / "selected_seed.json"
    if not selected_seed_path.exists():
        raise FileNotFoundError(f"selected_seed.json not found in {checkpoint_dir}")

    payload = json.loads(selected_seed_path.read_text(encoding="utf-8"))
    if "selected_seed" not in payload:
        raise KeyError(f"selected_seed.json in {checkpoint_dir} does not contain 'selected_seed'")
    return int(payload["selected_seed"])


def resolve_prediction_artifact_path(
    checkpoint_dir: Path,
    artifact_name: str,
    prediction_source: str,
    fixed_seed: Optional[int],
) -> Path:
    if prediction_source == "mean":
        return checkpoint_dir / artifact_name
    if prediction_source == "selected_seed":
        seed_value = load_selected_seed(checkpoint_dir)
        return checkpoint_dir / f"seed_{seed_value}" / artifact_name
    if prediction_source == "fixed_seed":
        if fixed_seed is None:
            raise ValueError("fixed_seed must be provided when prediction_source='fixed_seed'")
        return checkpoint_dir / f"seed_{fixed_seed}" / artifact_name
    raise ValueError(f"Unsupported prediction_source: {prediction_source}")


def prediction_source_tag(prediction_source: str, fixed_seed: Optional[int]) -> str:
    if prediction_source == "mean":
        return "predsrc_mean"
    if prediction_source == "selected_seed":
        return "predsrc_selected_seed"
    if prediction_source == "fixed_seed":
        if fixed_seed is None:
            raise ValueError("fixed_seed must be provided when prediction_source='fixed_seed'")
        return f"predsrc_seed_{fixed_seed}"
    raise ValueError(f"Unsupported prediction_source: {prediction_source}")


def resolve_path(path_text: str, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def get_train_99p(data_dir: Path) -> float:
    params_path = data_dir / "ais_norm_params.npy"
    if params_path.exists():
        params = np.load(params_path, allow_pickle=True).item()
        return float(params.get("train_99p", 1.0))
    return 1.0


def normalize_spatiotemporal_array(array_path: Path) -> np.ndarray:
    array = np.squeeze(np.load(array_path))
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected [T, H, W] after squeeze for {array_path}, got {array.shape}")
    return array.astype(np.float32)


def load_mask(mask_path: Path, expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None

    mask = np.squeeze(np.load(mask_path)).astype(bool)
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match spatial shape {expected_shape}")
    return mask


def post_process_predictions(
    pred: np.ndarray,
    train_99p: float,
    noise_threshold: float,
) -> np.ndarray:
    physical_pred = np.asarray(pred, dtype=np.float32) * float(train_99p)
    physical_pred[physical_pred < noise_threshold] = 0.0
    return physical_pred


def load_prediction_series(
    model_outcomes_dir: Path,
    model_dir: str,
    prediction_source: str,
    fixed_seed: Optional[int],
    train_99p: float,
    noise_threshold: float,
) -> Tuple[np.ndarray, Path]:
    checkpoint_dir = model_outcomes_dir / model_dir
    pred_path = resolve_prediction_artifact_path(
        checkpoint_dir=checkpoint_dir,
        artifact_name="rollout_2024_preds.npy",
        prediction_source=prediction_source,
        fixed_seed=fixed_seed,
    )
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction artifact not found: {pred_path}")

    preds = post_process_predictions(
        normalize_spatiotemporal_array(pred_path),
        train_99p=train_99p,
        noise_threshold=noise_threshold,
    )
    return preds, pred_path


def prediction_artifact_exists(
    model_outcomes_dir: Path,
    model_dir: str,
    prediction_source: str,
    fixed_seed: Optional[int],
) -> bool:
    try:
        path = resolve_prediction_artifact_path(
            checkpoint_dir=model_outcomes_dir / model_dir,
            artifact_name="rollout_2024_preds.npy",
            prediction_source=prediction_source,
            fixed_seed=fixed_seed,
        )
    except Exception:
        return False
    return path.exists()


def prepare_ranked_rows(
    model_outcomes_dir: Path,
    mask_path: Path,
    checkpoint_glob: str,
    threshold: float,
    label_overrides: Dict[str, str],
) -> Tuple[List[Dict], List[Dict[str, str]]]:
    rows, skipped_rows = load_checkpoint_rows(
        model_outcomes_dir=model_outcomes_dir,
        mask_path=mask_path,
        threshold=threshold,
        checkpoint_glob=checkpoint_glob,
        label_overrides=label_overrides,
        return_skipped=True,
    )
    if not rows:
        raise RuntimeError(
            f"No complete checkpoint results were found under {model_outcomes_dir} with pattern {checkpoint_glob}"
        )

    threshold_metrics = build_threshold_metric_specs(threshold)
    assign_ranks(rows, CONTINUOUS_METRICS)
    assign_ranks(rows, threshold_metrics)
    add_average_ranks(rows, CONTINUOUS_METRICS, threshold_metrics)

    rollout_csi_rank_key = f"{threshold_metric_key('rollout_2024', 'CSI', threshold)}_rank"
    test_csi_rank_key = f"{threshold_metric_key('test_one_step', 'CSI', threshold)}_rank"
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            row["overall_avg_rank"],
            row[rollout_csi_rank_key],
            row[test_csi_rank_key],
        ),
    )
    for idx, row in enumerate(ranked_rows, start=1):
        row["overall_rank"] = idx
    return ranked_rows, skipped_rows


def find_row_by_model_dir(ranked_rows: Sequence[Dict], model_dir: str) -> Optional[Dict]:
    for row in ranked_rows:
        if row["model_dir"] == model_dir:
            return row
    return None


def resolve_model_label(
    ranked_rows: Sequence[Dict],
    model_dir: str,
    explicit_label: str,
    label_overrides: Dict[str, str],
) -> str:
    if explicit_label:
        return explicit_label

    row = find_row_by_model_dir(ranked_rows, model_dir)
    if row is not None:
        return str(row["model_label"])
    return infer_model_label(model_dir, label_overrides=label_overrides)


def select_baseline_row(
    ranked_rows: Sequence[Dict],
    model_outcomes_dir: Path,
    ours_model_dir: str,
    baseline_model_dir: str,
    prediction_source: str,
    fixed_seed: Optional[int],
) -> Dict:
    if baseline_model_dir:
        row = find_row_by_model_dir(ranked_rows, baseline_model_dir)
        if row is not None:
            return row
        return {
            "model_dir": baseline_model_dir,
            "model_label": infer_model_label(baseline_model_dir),
            "overall_rank": None,
        }

    for row in ranked_rows:
        if row["model_dir"] == ours_model_dir:
            continue
        if prediction_artifact_exists(
            model_outcomes_dir=model_outcomes_dir,
            model_dir=row["model_dir"],
            prediction_source=prediction_source,
            fixed_seed=fixed_seed,
        ):
            return row

    raise RuntimeError("No baseline checkpoint with a compatible rollout_2024 prediction artifact was found.")


def find_rollout_targets_path(
    model_outcomes_dir: Path,
    ranked_rows: Sequence[Dict],
    preferred_model_dir: str,
) -> Path:
    preferred_path = model_outcomes_dir / preferred_model_dir / "rollout_2024_targets.npy"
    if preferred_path.exists():
        return preferred_path

    for row in ranked_rows:
        target_path = model_outcomes_dir / row["model_dir"] / "rollout_2024_targets.npy"
        if target_path.exists():
            return target_path
    raise FileNotFoundError("No rollout_2024_targets.npy file found in the discovered checkpoint directories.")


def masked_spatial_mean(series: np.ndarray, mask_2d: Optional[np.ndarray]) -> np.ndarray:
    if mask_2d is None:
        return np.asarray(series, dtype=np.float64).reshape(series.shape[0], -1).mean(axis=1)

    valid = mask_2d.astype(bool)
    if not np.any(valid):
        raise ValueError("Mask contains no valid cells.")
    return np.asarray([float(np.mean(frame[valid])) for frame in series], dtype=np.float64)


def parse_month_numbers(months_text: str, horizon: int) -> List[int]:
    if not months_text.strip():
        raise ValueError("--months must be provided when --month-strategy explicit is used.")

    result: List[int] = []
    for item in months_text.split(","):
        text = item.strip()
        if not text:
            continue
        month_number = int(text)
        if month_number < 1 or month_number > horizon:
            raise ValueError(f"Month number {month_number} is outside the valid range 1-{horizon}.")
        idx = month_number - 1
        if idx not in result:
            result.append(idx)

    if not result:
        raise ValueError("No valid month numbers were parsed from --months.")
    return result


def select_typical_months(
    targets_2024: np.ndarray,
    mask_2d: Optional[np.ndarray],
    strategy: str,
    explicit_months: str,
    month_order: str,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    horizon = int(targets_2024.shape[0])
    monthly_mean = masked_spatial_mean(targets_2024, mask_2d)

    if strategy == "explicit":
        selected = [
            {"idx": idx, "role": "Selected"}
            for idx in parse_month_numbers(explicit_months, horizon)
        ]
    elif strategy == "seasonal":
        seasonal_indices = [idx for idx in (0, 3, 6, 9) if idx < horizon]
        selected = [{"idx": idx, "role": "Seasonal"} for idx in seasonal_indices]
    elif strategy == "auto":
        low_idx = int(np.argmin(monthly_mean))
        high_idx = int(np.argmax(monthly_mean))
        selected = []
        used = set()

        def add_candidate(role: str, idx: int) -> bool:
            if idx in used:
                return False
            selected.append({"idx": idx, "role": role})
            used.add(idx)
            return True

        add_candidate("Low activity", low_idx)
        add_candidate("High activity", high_idx)

        if horizon >= 2:
            deltas = np.diff(monthly_mean)
            for idx in np.argsort(deltas)[::-1] + 1:
                if add_candidate("Fast increase", int(idx)):
                    break
            for idx in np.argsort(deltas) + 1:
                if add_candidate("Fast decrease", int(idx)):
                    break

        filler_indices = [idx for idx in (0, 3, 6, 9) if idx < horizon]
        high_to_low = list(np.argsort(monthly_mean)[::-1])
        low_to_high = list(np.argsort(monthly_mean))
        for idx in filler_indices + high_to_low + low_to_high:
            if len(selected) >= 4:
                break
            int_idx = int(idx)
            if int_idx in used:
                continue
            selected.append({"idx": int_idx, "role": "Supplemental"})
            used.add(int_idx)
            if len(selected) >= 4:
                break
    else:
        raise ValueError(f"Unsupported month selection strategy: {strategy}")

    if not selected:
        raise RuntimeError("No typical months were selected.")

    if month_order == "chronological":
        selected = sorted(selected, key=lambda item: int(item["idx"]))
    elif month_order != "role":
        raise ValueError(f"Unsupported month order: {month_order}")

    return selected, monthly_mean


def month_label(month_idx: int) -> str:
    if 0 <= month_idx < len(MONTH_NAMES):
        return f"{MONTH_NAMES[month_idx]} 2024"
    return f"Month {month_idx + 1:02d}"


def format_month_title(month_item: Dict[str, object], monthly_mean: np.ndarray) -> str:
    idx = int(month_item["idx"])
    role = str(month_item["role"])
    mean_value = float(monthly_mean[idx])
    return f"{role}\n{month_label(idx)} | mean={mean_value:.2f}"


def build_geo_ticks(data_dir: Path, height: int, width: int) -> Tuple[List[int], List[str], List[int], List[str]]:
    lon_path = data_dir / "target_lons.npy"
    lat_path = data_dir / "target_lats.npy"

    if lon_path.exists():
        lons = np.asarray(np.load(lon_path), dtype=np.float64)
        x_labels = [f"{value:g}E" for value in (lons[0], lons[len(lons) // 2], lons[-1])]
    else:
        x_labels = ["West", "Center", "East"]

    if lat_path.exists():
        lats = np.asarray(np.load(lat_path), dtype=np.float64)
        y_labels = [f"{value:g}N" for value in (lats[0], lats[len(lats) // 2], lats[-1])]
    else:
        y_labels = ["South", "Center", "North"]

    x_ticks = [max(0, int(round(width * 0.08))), width // 2, min(width - 1, int(round(width * 0.92)))]
    y_ticks = [max(0, int(round(height * 0.10))), height // 2, min(height - 1, int(round(height * 0.90)))]
    return x_ticks, x_labels, y_ticks, y_labels


def apply_plot_mask(frame: np.ndarray, mask_2d: Optional[np.ndarray]) -> np.ndarray:
    plot_frame = np.asarray(frame, dtype=np.float32).copy()
    if mask_2d is not None:
        plot_frame[~mask_2d.astype(bool)] = np.nan
    return plot_frame


def resolve_vmax(values: np.ndarray, explicit_vmax: float, percentile: float, fallback: float = 0.1) -> float:
    if explicit_vmax > 0:
        return float(explicit_vmax)

    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return fallback

    vmax = float(np.percentile(finite_values, percentile))
    if vmax <= 0:
        vmax = float(np.max(finite_values))
    if vmax <= 0:
        vmax = fallback
    return vmax


def resolve_effort_cmap_name(cmap_name: str) -> str:
    normalized = cmap_name.strip().lower()
    if normalized in LEGACY_EFFORT_CMAPS:
        print(
            f"Replacing legacy prediction colormap '{cmap_name}' with '{DEFAULT_EFFORT_CMAP}' "
            "for publication-safe heatmaps."
        )
        return DEFAULT_EFFORT_CMAP
    if normalized not in ALLOWED_EFFORT_CMAPS:
        allowed_text = ", ".join(sorted(ALLOWED_EFFORT_CMAPS))
        raise ValueError(f"--cmap must be one of: {allowed_text}")
    return normalized


def build_white_center_diverging_cmap(cmap_name: str) -> mcolors.ListedColormap:
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(0.0, 1.0, 257))
    colors[len(colors) // 2] = mcolors.to_rgba(ERROR_CENTER_COLOR)
    cmap = mcolors.ListedColormap(colors, name=f"{cmap_name}_white_center")
    cmap.set_bad("#F2F2F2")
    return cmap


def render_figure(
    visual_rows: Sequence[Dict[str, object]],
    month_items: Sequence[Dict[str, object]],
    monthly_mean: np.ndarray,
    data_dir: Path,
    output_prefix: Path,
    mask_2d: Optional[np.ndarray],
    apply_mask: bool,
    effort_cmap_name: str,
    error_cmap_name: str,
    vmax: float,
    vmax_percentile: float,
    error_vmax: float,
    error_vmax_percentile: float,
    dpi: int,
) -> None:
    if not visual_rows:
        raise RuntimeError("No visual rows are available for plotting.")
    if not month_items:
        raise RuntimeError("No selected months are available for plotting.")

    selected_indices = [int(item["idx"]) for item in month_items]
    num_rows = len(visual_rows)
    num_cols = len(selected_indices)
    first_series = np.asarray(visual_rows[0]["data"])
    _, height, width = first_series.shape

    fig_w = max(11.0, num_cols * 3.2)
    fig_h = max(8.0, num_rows * 2.25)
    bitmap_width_px = int(round(fig_w * dpi))
    if dpi < MIN_EXPORT_DPI:
        raise ValueError(f"--dpi must be at least {MIN_EXPORT_DPI} for publication bitmap export")
    if bitmap_width_px < MIN_EXPORT_WIDTH_PX:
        raise ValueError(
            f"Bitmap export would be {bitmap_width_px}px wide; expected at least {MIN_EXPORT_WIDTH_PX}px"
        )
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        num_rows,
        num_cols,
        figure=fig,
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.91,
        wspace=0.20,
        hspace=0.34,
    )

    plot_mask = mask_2d if apply_mask else None
    effort_arrays = []
    error_arrays = []
    for row in visual_rows:
        row_kind = str(row.get("kind", "effort"))
        row_data = np.asarray(row["data"], dtype=np.float32)
        selected_data = np.asarray([apply_plot_mask(row_data[idx], plot_mask) for idx in selected_indices])
        if row_kind == "error":
            error_arrays.append(selected_data)
        else:
            effort_arrays.append(selected_data)

    effort_stack = np.concatenate(effort_arrays, axis=0) if effort_arrays else np.zeros((1, height, width))
    vmax_effort = resolve_vmax(effort_stack, vmax, vmax_percentile)
    effort_norm = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax_effort)
    effort_cmap_name = resolve_effort_cmap_name(effort_cmap_name)
    effort_cmap = plt.get_cmap(effort_cmap_name).copy()
    effort_cmap.set_bad("#F2F2F2")

    if error_arrays:
        error_stack = np.concatenate(error_arrays, axis=0)
        abs_error = np.abs(error_stack)
        vmax_error = resolve_vmax(abs_error, error_vmax, error_vmax_percentile)
    else:
        vmax_error = 0.1
    error_norm = mcolors.TwoSlopeNorm(vmin=-vmax_error, vcenter=0.0, vmax=vmax_error)
    error_cmap = build_white_center_diverging_cmap(error_cmap_name)

    x_ticks, x_labels, y_ticks, y_labels = build_geo_ticks(data_dir, height, width)
    effort_im = None
    error_im = None

    for row_idx, row in enumerate(visual_rows):
        row_kind = str(row.get("kind", "effort"))
        row_data = np.asarray(row["data"], dtype=np.float32)
        row_label = str(row["label"])
        is_ours_row = bool(row.get("is_ours", False))

        for col_idx, month_idx in enumerate(selected_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            frame = apply_plot_mask(row_data[month_idx], plot_mask)

            if row_kind == "error":
                error_im = ax.imshow(frame, cmap=error_cmap, norm=error_norm, origin="lower")
            else:
                effort_im = ax.imshow(frame, cmap=effort_cmap, norm=effort_norm, origin="lower")

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            if row_idx == num_rows - 1:
                ax.set_xticklabels(x_labels, fontsize=10)
                ax.tick_params(axis="x", direction="out", length=3, width=0.8, colors="black")
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", direction="in", length=2, width=0.5, colors="#999999")

            if col_idx == 0:
                ax.set_yticklabels(y_labels, fontsize=10)
                ax.tick_params(axis="y", direction="out", length=3, width=0.8, colors="black")
                ax.set_ylabel(
                    row_label,
                    fontsize=13,
                    fontweight="bold",
                    labelpad=13,
                    color=OURS_LABEL_COLOR if is_ours_row else DEFAULT_LABEL_COLOR,
                )
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", direction="in", length=2, width=0.5, colors="#999999")

            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.8)

            if row_idx == 0:
                ax.set_title(
                    format_month_title(month_items[col_idx], monthly_mean),
                    fontsize=12,
                    fontweight="bold",
                    pad=10,
                    linespacing=1.25,
                )

    if effort_im is None:
        raise RuntimeError("Failed to render any effort image tiles.")

    effort_cbar_ax = fig.add_axes([0.905, 0.28, 0.015, 0.60])
    effort_cbar = fig.colorbar(effort_im, cax=effort_cbar_ax)
    effort_cbar.set_label("AIS fishing effort (h/day)", fontsize=12, fontweight="bold", labelpad=12)
    effort_cbar.ax.tick_params(labelsize=10)

    if error_im is not None:
        error_cbar_ax = fig.add_axes([0.905, 0.08, 0.015, 0.14])
        error_cbar = fig.colorbar(error_im, cax=error_cbar_ax)
        error_cbar.set_ticks([-vmax_error, 0.0, vmax_error])
        error_cbar.set_label("Prediction error (h/day; 0 = white)", fontsize=11, fontweight="bold", labelpad=10)
        error_cbar.ax.tick_params(labelsize=9)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    print(f"Saved PNG to: {png_path}")
    print(f"Saved PDF to: {pdf_path}")


def resolve_output_prefixes(
    output_dir: Path,
    explicit_output_prefix: str,
    threshold: float,
    month_strategy: str,
    prediction_source_name: str,
) -> Tuple[Path, Path]:
    if explicit_output_prefix:
        output_prefix = Path(explicit_output_prefix)
        if not output_prefix.is_absolute():
            output_prefix = (PROJECT_ROOT / output_prefix).resolve()
        return output_prefix, output_prefix

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_prefix = output_dir / (
        "Typical_Month_Spatial_Distribution_"
        f"{build_artifact_tag(threshold, 2, timestamp)}_{month_strategy}_{prediction_source_name}"
    )
    latest_prefix = output_dir / (
        "Typical_Month_Spatial_Distribution_"
        f"latest_threshold_{threshold_tag(threshold)}_{month_strategy}_{prediction_source_name}"
    )
    return archive_prefix, latest_prefix


def save_metadata(
    metadata_path: Path,
    month_items: Sequence[Dict[str, object]],
    monthly_mean: np.ndarray,
    targets_path: Path,
    ours_pred_path: Path,
    baseline_pred_path: Path,
    ours_model_dir: str,
    baseline_model_dir: str,
    train_99p: float,
    noise_threshold: float,
    prediction_source: str,
    fixed_seed: Optional[int],
    effort_cmap_name: str,
    error_cmap_name: str,
    dpi: int,
    num_visual_rows: int,
) -> None:
    selected_months = []
    for item in month_items:
        idx = int(item["idx"])
        selected_months.append(
            {
                "month_index_zero_based": idx,
                "month_number": idx + 1,
                "month_label": month_label(idx),
                "role": str(item["role"]),
                "ground_truth_mean_hours_per_day": float(monthly_mean[idx]),
            }
        )

    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_months": selected_months,
        "monthly_ground_truth_mean_hours_per_day": [float(value) for value in monthly_mean],
        "export": {
            "bitmap_dpi": int(dpi),
            "figure_width_in": max(11.0, len(month_items) * 3.2),
            "figure_height_in": max(8.0, num_visual_rows * 2.25),
            "bitmap_width_px": int(round(max(11.0, len(month_items) * 3.2) * dpi)),
            "minimum_bitmap_dpi": MIN_EXPORT_DPI,
            "minimum_bitmap_width_px": MIN_EXPORT_WIDTH_PX,
            "pdf": "PDF is also exported; heatmap panels are embedded at the configured DPI.",
        },
        "colormaps": {
            "effort": effort_cmap_name,
            "effort_allowed": sorted(ALLOWED_EFFORT_CMAPS),
            "error": error_cmap_name,
            "error_zero_midpoint_color": ERROR_CENTER_COLOR,
        },
        "colorbar_units": {
            "effort": "AIS fishing effort (h/day)",
            "error": "Prediction error (h/day; 0 = white)",
        },
        "targets_path": str(targets_path),
        "ours_model_dir": ours_model_dir,
        "ours_prediction_path": str(ours_pred_path),
        "baseline_model_dir": baseline_model_dir,
        "baseline_prediction_path": str(baseline_pred_path),
        "train_99p": float(train_99p),
        "noise_threshold_hours_per_day": float(noise_threshold),
        "prediction_source": prediction_source,
        "fixed_seed": fixed_seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    print(f"Saved metadata to: {metadata_path}")


def update_latest_outputs(archive_prefix: Path, latest_prefix: Path) -> None:
    if archive_prefix == latest_prefix:
        return
    for suffix in (".png", ".pdf", ".json"):
        src = archive_prefix.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, latest_prefix.with_suffix(suffix))


def print_scan_report(
    ranked_rows: Sequence[Dict],
    metric_skipped_rows: Sequence[Dict[str, str]],
) -> None:
    print(f"Ranked checkpoint dirs: {len(ranked_rows)}")
    print(f"Skipped during metrics scan: {len(metric_skipped_rows)}")
    for item in metric_skipped_rows:
        print(f"  - {item['model_dir']}: {item['reason']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a compact typical-month spatial distribution figure using rollout_2024 predictions."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "ST_FishNet_Features"),
        help="Directory containing AIS normalization metadata and optional lat/lon arrays.",
    )
    parser.add_argument(
        "--model-outcomes-dir",
        type=str,
        default=str(MODEL_OUTCOMES_DIR),
        help="Directory containing checkpoint result folders.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        default=str(MASK_PATH),
        help="Shared ocean-mask path used for model ranking and optional visual masking.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory used when --output-prefix is not provided.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional explicit output prefix without extension.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Threshold used for ranking the automatically discovered checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=CHECKPOINT_GLOB,
        help="Glob pattern used to discover checkpoint directories.",
    )
    parser.add_argument(
        "--ours-model-dir",
        type=str,
        default=DEFAULT_OURS_MODEL_DIR,
        help="Checkpoint directory name for the proposed/main model.",
    )
    parser.add_argument(
        "--baseline-model-dir",
        type=str,
        default="",
        help="Optional checkpoint directory name for the baseline row. If omitted, the best-ranked non-ours model is used.",
    )
    parser.add_argument(
        "--ours-label",
        type=str,
        default="",
        help="Optional display label for the proposed/main model.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="",
        help="Optional display label for the baseline model.",
    )
    parser.add_argument(
        "--month-strategy",
        type=str,
        default="auto",
        choices=["auto", "seasonal", "explicit"],
        help=(
            "auto selects low, high, fastest-increase, and fastest-decrease months from 2024 ground truth; "
            "seasonal uses months 1,4,7,10; explicit uses --months."
        ),
    )
    parser.add_argument(
        "--months",
        type=str,
        default="",
        help="Comma-separated 1-based month numbers used when --month-strategy explicit is selected.",
    )
    parser.add_argument(
        "--month-order",
        type=str,
        default="role",
        choices=["role", "chronological"],
        help="Column ordering for selected months.",
    )
    parser.add_argument(
        "--prediction-source",
        type=str,
        default="mean",
        choices=["mean", "selected_seed", "fixed_seed"],
        help=(
            "Prediction artifact source used for every model uniformly. "
            "'mean' uses aggregated root predictions, 'selected_seed' uses each model's selected_seed.json, "
            "and 'fixed_seed' uses the same seed_N subdirectory for every model."
        ),
    )
    parser.add_argument(
        "--fixed-seed",
        type=int,
        default=None,
        help="Seed value used when --prediction-source fixed_seed is selected.",
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=0.05,
        help="Physical-hours threshold used to suppress tiny prediction noise after denormalization.",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Include the corresponding 2023 history months from ais_val.npy as an extra top row.",
    )
    parser.add_argument(
        "--no-plot-mask",
        action="store_true",
        help="Do not mask invalid land cells in the rendered figure.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_EFFORT_CMAP,
        help="Colormap used for ground-truth and prediction effort maps: cividis, viridis, or turbo.",
    )
    parser.add_argument(
        "--error-cmap",
        type=str,
        default="RdBu_r",
        help="Diverging colormap used for signed error maps; the zero midpoint is forced to white.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.0,
        help="Explicit effort colorbar max in hours/day. If <=0, percentile scaling is used.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.5,
        help="Percentile used for effort colorbar max when --vmax is not provided.",
    )
    parser.add_argument(
        "--error-vmax",
        type=float,
        default=0.0,
        help="Explicit absolute signed-error colorbar max. If <=0, percentile scaling is used.",
    )
    parser.add_argument(
        "--error-vmax-percentile",
        type=float,
        default=99.0,
        help="Percentile used for signed-error colorbar max when --error-vmax is not provided.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=MIN_EXPORT_DPI,
        help="Output DPI for PNG/PDF export; must be at least 600.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any checkpoint is skipped during the metrics scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cmap = resolve_effort_cmap_name(args.cmap)
    if args.dpi < MIN_EXPORT_DPI:
        raise ValueError(f"--dpi must be at least {MIN_EXPORT_DPI} for publication bitmap export")

    if args.prediction_source == "fixed_seed" and args.fixed_seed is None:
        raise ValueError("--fixed-seed must be provided when --prediction-source fixed_seed is used.")
    if args.prediction_source != "fixed_seed" and args.fixed_seed is not None:
        print("Ignoring --fixed-seed because --prediction-source is not fixed_seed.")

    data_dir = resolve_path(args.data_dir)
    model_outcomes_dir = resolve_path(args.model_outcomes_dir)
    mask_path = resolve_path(args.mask_path)
    output_dir = resolve_path(args.output_dir)

    label_overrides = dict(DEFAULT_LABEL_OVERRIDES)
    if args.ours_label:
        label_overrides[args.ours_model_dir] = args.ours_label
    if args.baseline_model_dir and args.baseline_label:
        label_overrides[args.baseline_model_dir] = args.baseline_label

    ranked_rows, metric_skipped_rows = prepare_ranked_rows(
        model_outcomes_dir=model_outcomes_dir,
        mask_path=mask_path,
        checkpoint_glob=args.checkpoint_glob,
        threshold=args.threshold,
        label_overrides=label_overrides,
    )
    if args.strict and metric_skipped_rows:
        print_scan_report(ranked_rows, metric_skipped_rows)
        raise RuntimeError("Strict mode enabled and some checkpoints were skipped during metrics scan.")

    train_99p = get_train_99p(data_dir)
    targets_path = find_rollout_targets_path(
        model_outcomes_dir=model_outcomes_dir,
        ranked_rows=ranked_rows,
        preferred_model_dir=args.ours_model_dir,
    )
    targets_2024 = normalize_spatiotemporal_array(targets_path) * train_99p
    horizon, height, width = targets_2024.shape
    mask_2d = load_mask(mask_path, (height, width))

    month_items, monthly_mean = select_typical_months(
        targets_2024=targets_2024,
        mask_2d=mask_2d,
        strategy=args.month_strategy,
        explicit_months=args.months,
        month_order=args.month_order,
    )

    ours_preds, ours_pred_path = load_prediction_series(
        model_outcomes_dir=model_outcomes_dir,
        model_dir=args.ours_model_dir,
        prediction_source=args.prediction_source,
        fixed_seed=args.fixed_seed,
        train_99p=train_99p,
        noise_threshold=args.noise_threshold,
    )
    if ours_preds.shape != targets_2024.shape:
        raise ValueError(f"Ours prediction shape {ours_preds.shape} does not match targets {targets_2024.shape}")

    baseline_row = select_baseline_row(
        ranked_rows=ranked_rows,
        model_outcomes_dir=model_outcomes_dir,
        ours_model_dir=args.ours_model_dir,
        baseline_model_dir=args.baseline_model_dir,
        prediction_source=args.prediction_source,
        fixed_seed=args.fixed_seed,
    )
    baseline_model_dir = str(baseline_row["model_dir"])
    baseline_preds, baseline_pred_path = load_prediction_series(
        model_outcomes_dir=model_outcomes_dir,
        model_dir=baseline_model_dir,
        prediction_source=args.prediction_source,
        fixed_seed=args.fixed_seed,
        train_99p=train_99p,
        noise_threshold=args.noise_threshold,
    )
    if baseline_preds.shape != targets_2024.shape:
        raise ValueError(
            f"Baseline prediction shape {baseline_preds.shape} does not match targets {targets_2024.shape}"
        )

    ours_label = resolve_model_label(
        ranked_rows=ranked_rows,
        model_dir=args.ours_model_dir,
        explicit_label=args.ours_label,
        label_overrides=label_overrides,
    )
    baseline_label = args.baseline_label or str(baseline_row["model_label"])

    visual_rows: List[Dict[str, object]] = []
    if args.include_history:
        ais_val_path = data_dir / "ais_val.npy"
        if not ais_val_path.exists():
            raise FileNotFoundError(f"History source not found: {ais_val_path}")
        history_full = normalize_spatiotemporal_array(ais_val_path) * train_99p
        if history_full.shape[0] < horizon:
            raise ValueError(
                f"History series is shorter than the rollout horizon: {history_full.shape[0]} < {horizon}"
            )
        visual_rows.append({"label": "History (2023)", "data": history_full[-horizon:], "kind": "effort"})

    visual_rows.extend(
        [
            {"label": "G.T. (2024)", "data": targets_2024, "kind": "effort"},
            {"label": ours_label, "data": ours_preds, "kind": "effort", "is_ours": True},
            {"label": baseline_label, "data": baseline_preds, "kind": "effort"},
            {"label": "Ours - G.T.", "data": ours_preds - targets_2024, "kind": "error"},
        ]
    )

    archive_prefix, latest_prefix = resolve_output_prefixes(
        output_dir=output_dir,
        explicit_output_prefix=args.output_prefix,
        threshold=args.threshold,
        month_strategy=args.month_strategy,
        prediction_source_name=prediction_source_tag(args.prediction_source, args.fixed_seed),
    )
    archive_prefix.parent.mkdir(parents=True, exist_ok=True)
    latest_prefix.parent.mkdir(parents=True, exist_ok=True)

    render_figure(
        visual_rows=visual_rows,
        month_items=month_items,
        monthly_mean=monthly_mean,
        data_dir=data_dir,
        output_prefix=archive_prefix,
        mask_2d=mask_2d,
        apply_mask=not args.no_plot_mask,
        effort_cmap_name=args.cmap,
        error_cmap_name=args.error_cmap,
        vmax=args.vmax,
        vmax_percentile=args.vmax_percentile,
        error_vmax=args.error_vmax,
        error_vmax_percentile=args.error_vmax_percentile,
        dpi=args.dpi,
    )
    save_metadata(
        metadata_path=archive_prefix.with_suffix(".json"),
        month_items=month_items,
        monthly_mean=monthly_mean,
        targets_path=targets_path,
        ours_pred_path=ours_pred_path,
        baseline_pred_path=baseline_pred_path,
        ours_model_dir=args.ours_model_dir,
        baseline_model_dir=baseline_model_dir,
        train_99p=train_99p,
        noise_threshold=args.noise_threshold,
        prediction_source=args.prediction_source,
        fixed_seed=args.fixed_seed,
        effort_cmap_name=args.cmap,
        error_cmap_name=args.error_cmap,
        dpi=args.dpi,
        num_visual_rows=len(visual_rows),
    )
    update_latest_outputs(archive_prefix, latest_prefix)

    print("Selected typical months:")
    for item in month_items:
        idx = int(item["idx"])
        print(f"  - {str(item['role'])}: {month_label(idx)} (mean={monthly_mean[idx]:.4f} hours/day)")
    print(f"Ours: {args.ours_model_dir} -> {ours_pred_path}")
    print(f"Baseline: {baseline_model_dir} -> {baseline_pred_path}")
    if latest_prefix != archive_prefix:
        print(f"Updated latest PNG to: {latest_prefix.with_suffix('.png')}")
        print(f"Updated latest PDF to: {latest_prefix.with_suffix('.pdf')}")
        print(f"Updated latest metadata to: {latest_prefix.with_suffix('.json')}")
    print_scan_report(ranked_rows, metric_skipped_rows)


if __name__ == "__main__":
    main()
