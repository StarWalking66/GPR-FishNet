from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, rcParams
from matplotlib.lines import Line2D

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_ablation_common import (
    DEFAULT_ABLATION_OUT_DIR,
    canonical_metric_list,
    ensure_rows_by_key,
    get_metric_std,
    get_metric_value,
    load_csv_rows,
    load_significance_map,
    metric_direction,
    pretty_metric_name,
    pretty_split_name,
    save_figure,
    set_publication_style,
    significance_marker_and_color,
    signed_gain,
    variance_sum_std,
)


DEFAULT_ENV_DIR = DEFAULT_ABLATION_OUT_DIR / "env_factor_ablation"
DEFAULT_SUMMARY_CSV = DEFAULT_ENV_DIR / "env_factor_leave_one_out_summary.csv"
DEFAULT_SIGNIFICANCE_JSON = DEFAULT_ENV_DIR / "env_factor_significance_vs_full_factors.json"
DEFAULT_OUTPUT_DIR = DEFAULT_ABLATION_OUT_DIR / "figures"
DEFAULT_PRIMARY_HOTSPOT_THRESHOLD = 0.2755
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
DOWN_ARROW = "\u2193"
UP_ARROW = "\u2191"
PNG_DPI = 600
MIN_FULL_WIDTH_PNG_PX = 3740
GAIN_POSITIVE_COLOR = "#0072B2"
GAIN_NEGATIVE_COLOR = "#E69F00"
GAIN_BAR_EDGE_COLOR = "#111827"
GAIN_LABEL_COLOR = "#111827"
SIGNIFICANCE_MARKER_COLOR = "#374151"

FACTOR_LABELS = {
    "full_factors": "Full factors",
    "thetao": "w/o thetao",
    "chl": "w/o chl",
    "uo": "w/o uo",
    "vo": "w/o vo",
    "so": "w/o so",
    "zos": "w/o zos",
    "o2": "w/o o2",
}
HIGHLIGHT_EXPERIMENTS = {"full_factors"}

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures for environmental-factor ablation.")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="Optional explicit summary CSV path. If omitted, the script auto-resolves from --threshold.",
    )
    parser.add_argument(
        "--significance-json",
        type=str,
        default="",
        help="Optional explicit significance JSON path. If omitted, the script auto-resolves from --threshold.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for figure exports.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional explicit output prefix without extension. If omitted, archive and latest outputs are written.",
    )
    parser.add_argument("--split", type=str, default="rollout_2024", choices=["test_one_step", "rollout_2024"], help="Split used in gain-bar figure.")
    parser.add_argument("--metrics", type=str, default="CSI,F1,RMSE,R2", help="Comma-separated metric names for gain bars.")
    parser.add_argument(
        "--heatmap-metrics",
        type=str,
        default="MAE,RMSE,R2,CSI,F1",
        help="Comma-separated metric names for heatmap columns.",
    )
    parser.add_argument(
        "--heatmap-splits",
        type=str,
        default="test_one_step,rollout_2024",
        help="Comma-separated splits used in heatmap.",
    )
    parser.add_argument(
        "--table-metrics",
        type=str,
        default="RMSE,R2,CSI,F1",
        help="Comma-separated metric names for the manuscript-style preview table.",
    )
    parser.add_argument(
        "--factor-order",
        type=str,
        default="thetao,chl,uo,vo,so,zos,o2",
        help="Comma-separated environmental-factor order. Missing factors are appended automatically.",
    )
    parser.add_argument(
        "--hotspot-threshold-label",
        type=float,
        default=None,
        help="Optional threshold label to append to CSI/F1 table columns, e.g. 0.3175.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold selector. If set to a non-primary threshold, the script auto-loads threshold-specific summary/significance files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for heatmap/table PNG export. Gain-bar PNG is fixed at 600 dpi with vector PDF.",
    )
    return parser.parse_args()


def threshold_tag(hotspot_threshold: float) -> str:
    return f"{hotspot_threshold:.10f}".rstrip("0").rstrip(".").replace(".", "p")


def resolve_input_paths(
    summary_csv_raw: str,
    significance_json_raw: str,
    requested_threshold: Optional[float],
) -> Tuple[Path, Path]:
    if summary_csv_raw:
        summary_csv = Path(summary_csv_raw)
    else:
        if requested_threshold is None or abs(float(requested_threshold) - DEFAULT_PRIMARY_HOTSPOT_THRESHOLD) < 1e-12:
            summary_csv = DEFAULT_SUMMARY_CSV
        else:
            summary_csv = DEFAULT_ENV_DIR / f"env_factor_leave_one_out_summary_threshold_{threshold_tag(float(requested_threshold))}.csv"

    if significance_json_raw:
        significance_json = Path(significance_json_raw)
    else:
        if requested_threshold is None or abs(float(requested_threshold) - DEFAULT_PRIMARY_HOTSPOT_THRESHOLD) < 1e-12:
            significance_json = DEFAULT_SIGNIFICANCE_JSON
        else:
            significance_json = DEFAULT_ENV_DIR / f"env_factor_significance_vs_full_factors_threshold_{threshold_tag(float(requested_threshold))}.json"

    return summary_csv, significance_json


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_artifact_tag(hotspot_threshold: float, factor_count: int, timestamp: str) -> str:
    return f"threshold_{threshold_tag(hotspot_threshold)}_{factor_count}factors_{timestamp}"


def resolve_output_prefixes(
    output_dir: Path,
    explicit_output_prefix: str,
    figure_label: str,
    hotspot_threshold: float,
    factor_count: int,
) -> Tuple[Path, Path]:
    suffix = f"_{figure_label}"
    if explicit_output_prefix:
        output_prefix = Path(explicit_output_prefix)
        if not output_prefix.is_absolute():
            output_prefix = (PROJECT_ROOT / output_prefix).resolve()
        prefixed = output_prefix.with_name(f"{output_prefix.name}{suffix}")
        return prefixed, prefixed

    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    archive_prefix = output_dir / f"{figure_label}_{build_artifact_tag(hotspot_threshold, factor_count, timestamp)}"
    latest_prefix = output_dir / f"{figure_label}_latest_threshold_{threshold_tag(hotspot_threshold)}"
    return archive_prefix, latest_prefix


def update_latest_outputs(archive_prefix: Path, latest_prefix: Path) -> None:
    if archive_prefix == latest_prefix:
        return
    for suffix in (".png", ".pdf", ".csv"):
        archive_file = archive_prefix.with_suffix(suffix)
        if archive_file.exists():
            shutil.copy2(archive_file, latest_prefix.with_suffix(suffix))


def tight_png_width_px(fig: plt.Figure, dpi: int) -> int:
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    return int(round(bbox.width * dpi))


def save_gain_bar_figure(fig: plt.Figure, output_prefix: Path) -> Tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
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
    return png_path, pdf_path


def _build_drop_rows(rows: Sequence[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    rows_by_exp = ensure_rows_by_key(list(rows), "experiment")
    full_row = rows_by_exp.get("full_factors")
    if full_row is None:
        raise RuntimeError("The summary file does not contain 'full_factors', cannot build leave-one-out reference.")

    drop_rows: Dict[str, Dict] = {}
    for row in rows:
        dropped = str(row.get("dropped_factor", "")).strip()
        if not dropped or dropped.lower() == "none":
            continue
        drop_rows[dropped] = row
    if not drop_rows:
        raise RuntimeError("No drop_* rows were found in environment-factor ablation summary.")
    return {"full": full_row}, drop_rows


def _ordered_factors(drop_rows: Dict[str, Dict], preferred_order: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for factor in preferred_order:
        if factor in drop_rows and factor not in ordered:
            ordered.append(factor)
    for factor in drop_rows.keys():
        if factor not in ordered:
            ordered.append(factor)
    return ordered


def resolve_hotspot_threshold_label(
    full_row: Dict,
    drop_rows: Dict[str, Dict],
    explicit_threshold: Optional[float],
    requested_threshold: Optional[float],
) -> Optional[float]:
    if explicit_threshold is not None:
        return float(explicit_threshold)

    detected_values: List[float] = []
    for row in [full_row, *drop_rows.values()]:
        value = row.get("hotspot_threshold")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and not any(abs(parsed - existing) < 1e-12 for existing in detected_values):
            detected_values.append(parsed)

    if len(detected_values) == 1:
        return detected_values[0]
    if requested_threshold is not None:
        return float(requested_threshold)
    return None


def pretty_metric_display(metric_name: str, hotspot_threshold: Optional[float]) -> str:
    label = pretty_metric_name(metric_name)
    if hotspot_threshold is not None and metric_name.upper() in {"CSI", "F1"}:
        return f"{label}@{hotspot_threshold:.4f}"
    return label


def make_gain_bar_figure(
    full_row: Dict,
    drop_rows: Dict[str, Dict],
    factors: List[str],
    split: str,
    metrics: List[str],
    sig_map: Dict[Tuple[str, str, str], Dict],
    hotspot_threshold: Optional[float],
) -> plt.Figure:
    num_panels = len(metrics)
    ncols = 2
    nrows = int(np.ceil(num_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.6 * ncols, 4.8 * nrows), squeeze=False)

    x = np.arange(len(factors), dtype=np.float64)
    factor_labels = [f"-{name}" for name in factors]

    for panel_idx, metric in enumerate(metrics):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        gains: List[float] = []
        gain_stds: List[float] = []
        colors: List[str] = []

        full_val = get_metric_value(full_row, split, metric)
        full_std = get_metric_std(full_row, split, metric)

        for factor in factors:
            row = drop_rows[factor]
            cand_val = get_metric_value(row, split, metric)
            cand_std = get_metric_std(row, split, metric)
            gain = signed_gain(cand_val, full_val, metric)
            gains.append(gain)
            gain_stds.append(variance_sum_std(cand_std, full_std))
            colors.append(GAIN_POSITIVE_COLOR if gain >= 0 else GAIN_NEGATIVE_COLOR)

        gain_values = np.asarray(gains, dtype=np.float64)
        gain_errors = np.nan_to_num(np.asarray(gain_stds, dtype=np.float64), nan=0.0)
        bars = ax.bar(
            x,
            gain_values,
            yerr=gain_errors,
            capsize=3.5,
            color=colors,
            alpha=0.88,
            edgecolor=GAIN_BAR_EDGE_COLOR,
            linewidth=0.75,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
            zorder=3,
        )

        finite_bounds = np.concatenate([gain_values - gain_errors, gain_values + gain_errors])
        finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
        if finite_bounds.size:
            y_min = float(np.min(finite_bounds))
            y_max = float(np.max(finite_bounds))
            y_min = min(y_min, 0.0)
            y_max = max(y_max, 0.0)
            spread = max(1e-8, y_max - y_min)
        else:
            y_min, y_max, spread = -0.01, 0.01, 0.02
        y_offset = 0.045 * spread
        label_pad = 0.24 * spread
        for i, bar in enumerate(bars):
            factor = factors[i]
            exp_name = str(drop_rows[factor].get("experiment", f"drop_{factor}"))
            marker, _ = significance_marker_and_color(sig_map.get((exp_name, split, metric)))
            value = float(gain_values[i])
            err = float(gain_errors[i]) if np.isfinite(gain_errors[i]) else 0.0
            y_text = value + err + y_offset if value >= 0 else value - err - y_offset
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{value:+.3f}{marker}",
                color=GAIN_LABEL_COLOR,
                fontsize=8.7,
                fontweight="bold",
                ha="center",
                va=va,
                rotation=0,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.35),
                clip_on=True,
                zorder=5,
            )

        ax.axhline(0.0, color="#374151", linewidth=1.0, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(factor_labels, rotation=24, ha="right", rotation_mode="anchor", fontsize=11.2)
        ax.set_title(pretty_metric_display(metric, hotspot_threshold), fontsize=14.8, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        ax.tick_params(axis="y", labelsize=11.0)
        ax.tick_params(axis="x", labelsize=11.2)
        ax.set_ylim(y_min - label_pad, y_max + label_pad)

    for panel_idx in range(num_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].axis("off")

    legend_handles = [
        Line2D([0], [0], color=GAIN_POSITIVE_COLOR, linewidth=7, label="Positive signed gain (+)"),
        Line2D([0], [0], color=GAIN_NEGATIVE_COLOR, linewidth=7, label="Negative signed gain (-)"),
        Line2D([0], [0], color=SIGNIFICANCE_MARKER_COLOR, marker="*", linestyle="None", markersize=9, label="Significant vs full"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11.2,
        bbox_to_anchor=(0.5, 0.006),
        handlelength=1.8,
        columnspacing=1.4,
    )
    fig.suptitle(
        f"Env-Factor Ablation ({pretty_split_name(split)}): Signed Gain over Full Factors",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.95])
    return fig


def make_gain_heatmap_figure(
    full_row: Dict,
    drop_rows: Dict[str, Dict],
    factors: List[str],
    heatmap_splits: List[str],
    heatmap_metrics: List[str],
    sig_map: Dict[Tuple[str, str, str], Dict],
    hotspot_threshold: Optional[float],
) -> plt.Figure:
    n_rows = len(factors)
    n_cols = len(heatmap_splits) * len(heatmap_metrics)
    matrix = np.zeros((n_rows, n_cols), dtype=np.float64)
    texts: List[List[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    col_labels: List[str] = []

    for split in heatmap_splits:
        for metric in heatmap_metrics:
            col_labels.append(f"{pretty_split_name(split)}\n{pretty_metric_display(metric, hotspot_threshold)}")

    for row_idx, factor in enumerate(factors):
        row = drop_rows[factor]
        exp_name = str(row.get("experiment", f"drop_{factor}"))

        col_idx = 0
        for split in heatmap_splits:
            for metric in heatmap_metrics:
                cand = get_metric_value(row, split, metric)
                ref = get_metric_value(full_row, split, metric)
                gain = signed_gain(cand, ref, metric)
                matrix[row_idx, col_idx] = gain

                marker, _ = significance_marker_and_color(sig_map.get((exp_name, split, metric)))
                texts[row_idx][col_idx] = f"{gain:+.3f}{marker}"
                col_idx += 1

    max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(np.nanmax(np.abs(matrix))) else 1.0
    max_abs = max(max_abs, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(max(11.0, n_cols * 1.35), max(4.5, n_rows * 0.85 + 2.1)))
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"-{factor}" for factor in factors], fontsize=11)
    ax.set_title("Env-Factor Leave-One-Out: Signed Gain Heatmap", fontsize=16, fontweight="bold", pad=10)

    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.45 * max_abs else "#111827"
            ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Signed gain over Full-Factors", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout(rect=[0.0, 0.015, 1.0, 0.95])
    return fig


def build_table_columns(hotspot_threshold: Optional[float], table_metrics: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    one_step_columns: List[Tuple[str, str]] = []
    rollout_columns: List[Tuple[str, str]] = []
    for metric in table_metrics:
        display_name = pretty_metric_display(metric, hotspot_threshold)
        arrow = DOWN_ARROW if metric.upper() in {"MAE", "MSE", "RMSE"} else UP_ARROW
        one_step_columns.append((f"test_one_step_{metric}", f"{display_name} {arrow}"))
        rollout_columns.append((f"rollout_2024_{metric}", f"{display_name} {arrow}"))
    return one_step_columns, rollout_columns


def prepare_table_rows(full_row: Dict, drop_rows: Dict[str, Dict], factors: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    full_display_row = dict(full_row)
    full_display_row["row_label"] = FACTOR_LABELS["full_factors"]
    full_display_row["display_order"] = 0
    rows.append(full_display_row)

    for index, factor in enumerate(factors, start=1):
        display_row = dict(drop_rows[factor])
        display_row["row_label"] = FACTOR_LABELS.get(factor, f"w/o {factor}")
        display_row["display_order"] = index
        rows.append(display_row)

    rows.sort(key=lambda row: int(row.get("display_order", 999)))
    return rows


def format_table_value(metric_key: str, value: float, row: Dict) -> str:
    std_key = f"{metric_key}_std"
    std_value = row.get(std_key)
    if isinstance(std_value, (int, float, np.integer, np.floating)):
        return f"{value:.4f}\u00b1{float(std_value):.4f}"
    return f"{value:.4f}"


def metric_name_from_key(metric_key: str) -> str:
    for split_name in ("test_one_step", "rollout_2024"):
        prefix = f"{split_name}_"
        if metric_key.startswith(prefix):
            return metric_key.replace(prefix, "", 1)
    return metric_key


def table_cell_ranks(rows: Sequence[Dict], metric_columns: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
    rank_map: Dict[Tuple[str, str], int] = {}
    for metric_key, _ in metric_columns:
        values: List[Tuple[Dict, float]] = []
        for row in rows:
            try:
                values.append((row, float(row[metric_key])))
            except (KeyError, TypeError, ValueError):
                continue
        if not values:
            continue

        metric_name = metric_name_from_key(metric_key)
        reverse = metric_direction(metric_name) != "lower"
        sorted_values = sorted(values, key=lambda item: item[1], reverse=reverse)
        previous_value: Optional[float] = None
        current_rank = 0
        for position, (row, value) in enumerate(sorted_values, start=1):
            if previous_value is None or abs(value - previous_value) > max(1e-12, abs(previous_value) * 1e-9):
                current_rank = position
                previous_value = value
            rank_map[(str(row.get("experiment", "")), metric_key)] = current_rank
    return rank_map


def table_figure_height(num_rows: int) -> float:
    return max(9.8, 6.1 + num_rows * 0.44)


def draw_table_panel(
    ax,
    rows: Sequence[Dict],
    panel_title: str,
    metric_columns: Sequence[Tuple[str, str]],
    panel_tag: str,
) -> None:
    ax.set_axis_off()
    n_rows = len(rows)
    col_defs = [("row_label", "Setting", 3.10)] + [(metric_key, metric_label, 1.88) for metric_key, metric_label in metric_columns]

    total_width = sum(width for _, _, width in col_defs)
    header_h = 1.08
    row_h = 0.92
    title_gap = 0.72
    bottom_gap = 0.18
    total_height = title_gap + header_h + n_rows * row_h + bottom_gap

    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    header_y = total_height - title_gap - header_h
    ax.text(
        total_width / 2,
        total_height - 0.10,
        f"{panel_tag} {panel_title}",
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="top",
    )

    x0 = 0.0
    for _, label, width in col_defs:
        rect = patches.Rectangle(
            (x0, header_y),
            width,
            header_h,
            facecolor="#243447",
            edgecolor="white",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + width / 2,
            header_y + header_h / 2,
            label,
            fontsize=10.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        x0 += width

    border_color = "#d8dde6"
    stripe_fill = "#fafbfc"
    normal_fill = "#ffffff"
    highlight_row_fill = "#fff5f5"
    metric_name_set = {name for name, _ in metric_columns}
    cell_ranks = table_cell_ranks(rows, metric_columns)
    for row_idx, row in enumerate(rows):
        y = header_y - (row_idx + 1) * row_h
        base_fill = stripe_fill if row_idx % 2 == 0 else normal_fill
        if str(row.get("experiment", "")) in HIGHLIGHT_EXPERIMENTS:
            base_fill = highlight_row_fill
        x = 0.0
        for col_idx, (metric_key, _, width) in enumerate(col_defs):
            text_color = "#111827"
            fontweight = "normal"
            if metric_key == "row_label" and str(row.get("experiment", "")) in HIGHLIGHT_EXPERIMENTS:
                text_color = "#b91c1c"
                fontweight = "bold"
            elif metric_key in metric_name_set:
                metric_rank = cell_ranks.get((str(row.get("experiment", "")), metric_key))
                if metric_rank == 1:
                    text_color = "#b91c1c"
                    fontweight = "bold"
                elif metric_rank == 2:
                    text_color = "#2563eb"
                    fontweight = "bold"

            rect = patches.Rectangle(
                (x, y),
                width,
                row_h,
                facecolor=base_fill,
                edgecolor=border_color,
                linewidth=0.7,
            )
            ax.add_patch(rect)
            value = row[metric_key]
            text = value if metric_key == "row_label" else format_table_value(metric_key, value, row)
            ha = "left" if metric_key == "row_label" else "center"
            x_text = x + 0.12 if metric_key == "row_label" else x + width / 2
            text_fontsize = 9.4 if metric_key in metric_name_set else 10.0
            ax.text(
                x_text,
                y + row_h / 2,
                text,
                fontsize=text_fontsize,
                color=text_color,
                fontweight=fontweight,
                ha=ha,
                va="center",
            )

            if col_idx == 0 and str(row.get("experiment", "")) in HIGHLIGHT_EXPERIMENTS:
                ax.add_line(
                    plt.Line2D(
                        [x + 0.02, x + 0.02],
                        [y + 0.06, y + row_h - 0.06],
                        color="#b91c1c",
                        linewidth=2.0,
                    )
                )
            x += width

    ax.add_patch(
        patches.Rectangle(
            (0.0, header_y - n_rows * row_h),
            total_width,
            header_h + n_rows * row_h,
            fill=False,
            edgecolor="#8d96a6",
            linewidth=0.9,
        )
    )


def plot_env_factor_table_figure(
    rows: List[Dict],
    output_prefix: Path,
    threshold: float,
    table_metrics: Sequence[str],
    dpi: int = 300,
) -> Tuple[Path, Path]:
    fig = plt.figure(figsize=(14.8, table_figure_height(len(rows))), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.12, top=0.895, bottom=0.055, left=0.06, right=0.94)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    one_step_columns, rollout_columns = build_table_columns(threshold, table_metrics)
    draw_table_panel(ax1, rows, "One-step Forecasting Performance Ranking", one_step_columns, "(a)")
    draw_table_panel(ax2, rows, "2024 Rollout Forecasting Performance Ranking", rollout_columns, "(b)")

    fig.suptitle(
        f"Environmental-factor ablation performance under a fixed hotspot threshold ({len(rows)} settings)",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    return save_figure(fig, output_prefix, dpi=dpi)


def export_env_factor_table_csv(
    rows: Sequence[Dict],
    output_prefix: Path,
    threshold: float,
    table_metrics: Sequence[str],
) -> Path:
    csv_path = output_prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Setting"]
    for split in ("test_one_step", "rollout_2024"):
        split_label = pretty_split_name(split)
        for metric in table_metrics:
            headers.append(f"{split_label} {pretty_metric_display(metric, threshold)}")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            output_row = [row["row_label"]]
            for split in ("test_one_step", "rollout_2024"):
                for metric in table_metrics:
                    metric_key = f"{split}_{metric}"
                    output_row.append(format_table_value(metric_key, float(row[metric_key]), row))
            writer.writerow(output_row)

    return csv_path


def main() -> None:
    args = parse_args()
    set_publication_style()

    summary_csv, significance_json = resolve_input_paths(args.summary_csv, args.significance_json, args.threshold)
    output_dir = resolve_output_path(Path(args.output_dir))
    if not summary_csv.is_absolute():
        summary_csv = resolve_output_path(summary_csv)
    if not significance_json.is_absolute():
        significance_json = resolve_output_path(significance_json)

    metrics = canonical_metric_list(args.metrics)
    heatmap_metrics = canonical_metric_list(args.heatmap_metrics)
    heatmap_splits = canonical_metric_list(args.heatmap_splits)
    table_metrics = canonical_metric_list(args.table_metrics)
    preferred_order = canonical_metric_list(args.factor_order)

    rows = load_csv_rows(summary_csv)
    full_container, drop_rows = _build_drop_rows(rows)
    full_row = full_container["full"]
    factors = _ordered_factors(drop_rows, preferred_order)
    sig_map = load_significance_map(significance_json)
    hotspot_threshold = resolve_hotspot_threshold_label(full_row, drop_rows, args.hotspot_threshold_label, args.threshold)
    output_threshold = (
        float(hotspot_threshold)
        if hotspot_threshold is not None
        else float(args.threshold if args.threshold is not None else DEFAULT_PRIMARY_HOTSPOT_THRESHOLD)
    )
    table_rows = prepare_table_rows(full_row, drop_rows, factors)

    fig_bars = make_gain_bar_figure(
        full_row=full_row,
        drop_rows=drop_rows,
        factors=factors,
        split=args.split,
        metrics=metrics,
        sig_map=sig_map,
        hotspot_threshold=hotspot_threshold,
    )
    bars_archive_prefix, bars_latest_prefix = resolve_output_prefixes(
        output_dir,
        args.output_prefix,
        "SCI_Env_Factor_Ablation_Gain_Bars",
        output_threshold,
        len(factors),
    )
    bars_png, bars_pdf = save_gain_bar_figure(fig_bars, bars_archive_prefix)
    update_latest_outputs(bars_archive_prefix, bars_latest_prefix)

    fig_heatmap = make_gain_heatmap_figure(
        full_row=full_row,
        drop_rows=drop_rows,
        factors=factors,
        heatmap_splits=heatmap_splits,
        heatmap_metrics=heatmap_metrics,
        sig_map=sig_map,
        hotspot_threshold=hotspot_threshold,
    )
    heat_archive_prefix, heat_latest_prefix = resolve_output_prefixes(
        output_dir,
        args.output_prefix,
        "SCI_Env_Factor_Ablation_Gain_Heatmap",
        output_threshold,
        len(factors),
    )
    heat_png, heat_pdf = save_figure(fig_heatmap, heat_archive_prefix, dpi=args.dpi)
    update_latest_outputs(heat_archive_prefix, heat_latest_prefix)

    table_archive_prefix, table_latest_prefix = resolve_output_prefixes(
        output_dir,
        args.output_prefix,
        "SCI_Env_Factor_Ablation_Performance_Table",
        output_threshold,
        len(table_rows),
    )
    table_png, table_pdf = plot_env_factor_table_figure(
        table_rows,
        table_archive_prefix,
        threshold=output_threshold,
        table_metrics=table_metrics,
        dpi=args.dpi,
    )
    table_csv = export_env_factor_table_csv(
        table_rows,
        table_archive_prefix,
        threshold=output_threshold,
        table_metrics=table_metrics,
    )
    update_latest_outputs(table_archive_prefix, table_latest_prefix)

    print(f"Loaded summary: {summary_csv}")
    print(f"Loaded significance: {significance_json}")
    print(f"Saved gain-bar archive to: {bars_png}")
    print(f"Saved gain-bar archive to: {bars_pdf}")
    if bars_latest_prefix != bars_archive_prefix:
        print(f"Updated gain-bar latest to: {bars_latest_prefix.with_suffix('.png')}")
        print(f"Updated gain-bar latest to: {bars_latest_prefix.with_suffix('.pdf')}")
    print(f"Saved heatmap archive to: {heat_png}")
    print(f"Saved heatmap archive to: {heat_pdf}")
    if heat_latest_prefix != heat_archive_prefix:
        print(f"Updated heatmap latest to: {heat_latest_prefix.with_suffix('.png')}")
        print(f"Updated heatmap latest to: {heat_latest_prefix.with_suffix('.pdf')}")
    print(f"Saved performance-table archive to: {table_png}")
    print(f"Saved performance-table archive to: {table_pdf}")
    print(f"Saved editable table CSV to: {table_csv}")
    if table_latest_prefix != table_archive_prefix:
        print(f"Updated performance-table latest to: {table_latest_prefix.with_suffix('.png')}")
        print(f"Updated performance-table latest to: {table_latest_prefix.with_suffix('.pdf')}")
        print(f"Updated performance-table CSV latest to: {table_latest_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
