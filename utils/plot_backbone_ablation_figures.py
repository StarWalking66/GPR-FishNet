from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, rcParams

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
)


DEFAULT_BACKBONE_DIR = DEFAULT_ABLATION_OUT_DIR / "backbone_ablation"
DEFAULT_SUMMARY_CSV = DEFAULT_BACKBONE_DIR / "backbone_ablation_summary.csv"
DEFAULT_SIGNIFICANCE_JSON = DEFAULT_BACKBONE_DIR / "backbone_significance_vs_full.json"
DEFAULT_OUTPUT_DIR = DEFAULT_ABLATION_OUT_DIR / "figures"
DEFAULT_PRIMARY_HOTSPOT_THRESHOLD = 0.2755
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["axes.linewidth"] = 0.8

DEFAULT_VARIANT_ORDER = [
    "baseline",
    "plus_arp",
    "plus_mssp",
    "plus_arp_mssp",
    "full",
]

VARIANT_LABELS = {
    "baseline": "STLSTM",
    "plus_arp": "STLSTM+ARP",
    "plus_mssp": "STLSTM+MSSP",
    "plus_arp_mssp": "STLSTM+ARP+MSSP",
    "full": "STLSTM+ARP+MSSP+CAR",
}

EXCLUDED_VARIANTS = {"no_relu"}

SPLITS = ["test_one_step", "rollout_2024"]
SPLIT_COLORS = {
    "test_one_step": "#2563eb",
    "rollout_2024": "#ea580c",
}
DOWN_ARROW = "\u2193"
UP_ARROW = "\u2191"
HIGHLIGHT_VARIANTS = {"full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures for backbone ablation.")
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
        help="Optional explicit output prefix without extension. If omitted, the script writes both archive and latest files.",
    )
    parser.add_argument("--metrics", type=str, default="RMSE,R2,CSI,F1", help="Comma-separated metric names.")
    parser.add_argument(
        "--variant-order",
        type=str,
        default=",".join(DEFAULT_VARIANT_ORDER),
        help="Comma-separated variant order in the figure.",
    )
    parser.add_argument(
        "--hotspot-threshold-label",
        type=float,
        default=None,
        help="Optional threshold label to append to CSI/F1 panels, e.g. 0.3175.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold selector. If set to a non-primary threshold, the script auto-loads threshold-specific summary/significance files.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG/PDF export.")
    return parser.parse_args()


def reorder_variants(available: Sequence[str], requested: Sequence[str]) -> List[str]:
    requested_unique: List[str] = []
    for name in requested:
        if name and name not in requested_unique:
            requested_unique.append(name)

    if not requested_unique:
        return list(available)
    return [name for name in requested_unique if name in available]


def _bar_error_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return arr


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
            summary_csv = DEFAULT_BACKBONE_DIR / f"backbone_ablation_summary_threshold_{threshold_tag(float(requested_threshold))}.csv"

    if significance_json_raw:
        significance_json = Path(significance_json_raw)
    else:
        if requested_threshold is None or abs(float(requested_threshold) - DEFAULT_PRIMARY_HOTSPOT_THRESHOLD) < 1e-12:
            significance_json = DEFAULT_SIGNIFICANCE_JSON
        else:
            significance_json = DEFAULT_BACKBONE_DIR / f"backbone_significance_vs_full_threshold_{threshold_tag(float(requested_threshold))}.json"

    return summary_csv, significance_json


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_artifact_tag(hotspot_threshold: float, variant_count: int, timestamp: str) -> str:
    return f"threshold_{threshold_tag(hotspot_threshold)}_{variant_count}variants_{timestamp}"


def resolve_output_prefixes(
    output_dir: Path,
    explicit_output_prefix: str,
    figure_label: str,
    hotspot_threshold: float,
    variant_count: int,
) -> Tuple[Path, Path]:
    suffix = f"_{figure_label}"
    if explicit_output_prefix:
        output_prefix = Path(explicit_output_prefix)
        if not output_prefix.is_absolute():
            output_prefix = (PROJECT_ROOT / output_prefix).resolve()
        prefixed = output_prefix.with_name(f"{output_prefix.name}{suffix}")
        return prefixed, prefixed

    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    archive_prefix = output_dir / f"{figure_label}_{build_artifact_tag(hotspot_threshold, variant_count, timestamp)}"
    latest_prefix = output_dir / f"{figure_label}_latest_threshold_{threshold_tag(hotspot_threshold)}"
    return archive_prefix, latest_prefix


def update_latest_outputs(archive_prefix: Path, latest_prefix: Path) -> None:
    if archive_prefix == latest_prefix:
        return
    for suffix in (".png", ".pdf"):
        shutil.copy2(archive_prefix.with_suffix(suffix), latest_prefix.with_suffix(suffix))


def resolve_hotspot_threshold_label(
    rows_by_variant: Dict[str, Dict],
    explicit_threshold: Optional[float],
    requested_threshold: Optional[float],
) -> Optional[float]:
    if explicit_threshold is not None:
        return float(explicit_threshold)

    detected_values = []
    for row in rows_by_variant.values():
        value = row.get("hotspot_threshold")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            detected_values.append(parsed)

    unique_values = []
    for value in detected_values:
        if not any(abs(value - existing) < 1e-12 for existing in unique_values):
            unique_values.append(value)

    if len(unique_values) == 1:
        return unique_values[0]
    if requested_threshold is not None:
        return float(requested_threshold)
    return None


def pretty_metric_display(metric_name: str, hotspot_threshold: Optional[float]) -> str:
    label = pretty_metric_name(metric_name)
    if hotspot_threshold is not None and metric_name.upper() in {"CSI", "F1"}:
        return f"{label}@{hotspot_threshold:.4f}"
    return label


def build_ranking_table_columns(hotspot_threshold: Optional[float]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    one_step_columns = [
        ("test_one_step_MAE", f"MAE {DOWN_ARROW}"),
        ("test_one_step_MSE", f"MSE {DOWN_ARROW}"),
        ("test_one_step_RMSE", f"RMSE {DOWN_ARROW}"),
        ("test_one_step_R2", f"R\u00b2 {UP_ARROW}"),
        ("test_one_step_SSIM", f"SSIM {UP_ARROW}"),
        ("test_one_step_CSI", f"{pretty_metric_display('CSI', hotspot_threshold)} {UP_ARROW}"),
        ("test_one_step_F1", f"{pretty_metric_display('F1', hotspot_threshold)} {UP_ARROW}"),
    ]
    rollout_columns = [
        ("rollout_2024_MAE", f"MAE {DOWN_ARROW}"),
        ("rollout_2024_MSE", f"MSE {DOWN_ARROW}"),
        ("rollout_2024_RMSE", f"RMSE {DOWN_ARROW}"),
        ("rollout_2024_R2", f"R\u00b2 {UP_ARROW}"),
        ("rollout_2024_SSIM", f"SSIM {UP_ARROW}"),
        ("rollout_2024_CSI", f"{pretty_metric_display('CSI', hotspot_threshold)} {UP_ARROW}"),
        ("rollout_2024_F1", f"{pretty_metric_display('F1', hotspot_threshold)} {UP_ARROW}"),
    ]
    return one_step_columns, rollout_columns


def prepare_ranking_table_rows(rows_by_variant: Dict[str, Dict], variants: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    for variant in variants:
        base_row = dict(rows_by_variant[variant])
        base_row["variant"] = variant
        base_row["model_label"] = VARIANT_LABELS.get(variant, variant)
        rows.append(base_row)

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            1 if str(row.get("variant", "")).strip().lower() == "full" else 0,
            len(str(row.get("model_label", ""))),
            str(row.get("model_label", "")),
        ),
    )
    return ordered_rows


def format_table_value(metric_key: str, value: float, row: Dict) -> str:
    std_key = f"{metric_key}_std"
    std_value = row.get(std_key)
    if isinstance(std_value, (int, float, np.integer, np.floating)):
        return f"{value:.4f}\u00b1{float(std_value):.4f}"
    return f"{value:.4f}"


def metric_name_from_key(metric_key: str) -> str:
    for split_name in SPLITS:
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
            rank_map[(str(row.get("variant", "")), metric_key)] = current_rank
    return rank_map


def ranking_table_figure_height(num_rows: int) -> float:
    return max(9.8, 6.1 + num_rows * 0.44)


def draw_ranking_table_panel(
    ax,
    rows: Sequence[Dict],
    panel_title: str,
    metric_columns: Sequence[Tuple[str, str]],
    panel_tag: str,
) -> None:
    ax.set_axis_off()
    n_rows = len(rows)

    col_defs = [
        ("model_label", "Model", 3.80),
    ] + [(metric_key, metric_label, 1.88) for metric_key, metric_label in metric_columns]

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
        if row.get("variant") in HIGHLIGHT_VARIANTS:
            base_fill = highlight_row_fill

        x = 0.0
        for col_idx, (metric_key, _, width) in enumerate(col_defs):
            cell_fill = base_fill
            text_color = "#111827"
            fontweight = "normal"

            if metric_key == "model_label" and row.get("variant") in HIGHLIGHT_VARIANTS:
                text_color = "#b91c1c"
                fontweight = "bold"
            elif metric_key in metric_name_set:
                metric_rank = cell_ranks.get((str(row.get("variant", "")), metric_key))
                if metric_rank == 1:
                    fontweight = "bold"
                    text_color = "#b91c1c"
                elif metric_rank == 2:
                    fontweight = "bold"
                    text_color = "#2563eb"

            rect = patches.Rectangle(
                (x, y),
                width,
                row_h,
                facecolor=cell_fill,
                edgecolor=border_color,
                linewidth=0.7,
            )
            ax.add_patch(rect)

            value = row[metric_key]
            text = value if metric_key == "model_label" else format_table_value(metric_key, value, row)
            ha = "left" if metric_key == "model_label" else "center"
            x_text = x + 0.10 if metric_key == "model_label" else x + width / 2
            text_fontsize = 9.2 if metric_key in metric_name_set else 10.2
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

            if col_idx == 0 and row.get("variant") in HIGHLIGHT_VARIANTS:
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


def plot_backbone_ranking_table_figure(
    rows: List[Dict],
    output_prefix: Path,
    threshold: float,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(17.2, ranking_table_figure_height(len(rows))), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.12, top=0.895, bottom=0.12, left=0.055, right=0.955)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    one_step_columns, rollout_columns = build_ranking_table_columns(threshold)

    draw_ranking_table_panel(ax1, rows, "One-step Forecasting Performance Ranking", one_step_columns, "(a)")
    draw_ranking_table_panel(ax2, rows, "2024 Rollout Forecasting Performance Ranking", rollout_columns, "(b)")

    fig.suptitle(
        f"Backbone ablation performance under a fixed hotspot threshold ({len(rows)} variants)",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    note_line_1 = (
        f"Error metrics are lower-is-better ({DOWN_ARROW}); skill metrics are higher-is-better ({UP_ARROW}). "
        f"Values are reported as mean\u00b1std when multi-seed summaries are available."
    )
    note_line_2 = (
        f"CSI/F1 are recomputed at threshold = {threshold:.4f}; red and blue metric values denote the best and second-best "
        f"variant in each panel. Highlighted row: {VARIANT_LABELS.get('full', 'full')}."
    )
    fig.text(
        0.5,
        0.045,
        f"{note_line_1}\n{note_line_2}",
        fontsize=9.6,
        color="#4b5563",
        ha="center",
        va="bottom",
        linespacing=1.35,
    )

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def make_metric_bar_figure(
    rows_by_variant: Dict[str, Dict],
    variants: List[str],
    metrics: List[str],
    sig_map: Dict[Tuple[str, str, str], Dict],
    hotspot_threshold: Optional[float],
) -> plt.Figure:
    num_panels = len(metrics)
    ncols = 2
    nrows = int(np.ceil(num_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 4.9 * nrows), squeeze=False)

    x = np.arange(len(variants), dtype=np.float64)
    width = 0.37
    variant_labels = [VARIANT_LABELS.get(v, v) for v in variants]

    for panel_idx, metric in enumerate(metrics):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        for split_idx, split in enumerate(SPLITS):
            color = SPLIT_COLORS[split]
            offsets = x + (split_idx - 0.5) * width
            values = [get_metric_value(rows_by_variant[v], split, metric) for v in variants]
            stds = [get_metric_std(rows_by_variant[v], split, metric) for v in variants]

            bars = ax.bar(
                offsets,
                values,
                width=width,
                yerr=_bar_error_array(stds),
                capsize=2.0,
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.8,
                label=pretty_split_name(split),
                zorder=3,
            )

            value_arr = np.asarray(values, dtype=np.float64)
            std_arr = np.asarray(stds, dtype=np.float64)
            finite_values = value_arr[np.isfinite(value_arr)]
            y_span = 1.0 if finite_values.size == 0 else max(1e-8, float(np.max(finite_values) - np.min(finite_values)))
            y_offset = 0.03 * y_span

            for bar_idx, bar in enumerate(bars):
                variant = variants[bar_idx]
                if variant == "full":
                    continue
                sig_row = sig_map.get((variant, split, metric))
                marker, marker_color = significance_marker_and_color(sig_row)
                if not marker:
                    continue

                bar_top = bar.get_height()
                err = 0.0 if not np.isfinite(std_arr[bar_idx]) else float(std_arr[bar_idx])
                y_text = bar_top + err + y_offset
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_text,
                    marker,
                    color=marker_color,
                    fontsize=12,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    zorder=6,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels, rotation=18, ha="right")
        ax.set_title(pretty_metric_display(metric, hotspot_threshold), fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
        ax.tick_params(labelsize=10)
        ax.set_axisbelow(True)

    for panel_idx in range(num_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.015))

    fig.suptitle("Backbone Ablation: Absolute Metrics (mean +/- std)", fontsize=18, fontweight="bold", y=0.98)
    note = "Marker '*' indicates p<0.05 vs Full under paired-seed sign-flip test. Green: significantly better, Red: significantly worse."
    if hotspot_threshold is not None:
        note += f" CSI/F1 use threshold = {hotspot_threshold:.4f}."
    fig.text(
        0.01,
        0.012,
        note,
        fontsize=10,
        color="#374151",
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])
    return fig


def make_gain_heatmap_figure(
    rows_by_variant: Dict[str, Dict],
    variants: List[str],
    metrics: List[str],
    sig_map: Dict[Tuple[str, str, str], Dict],
    hotspot_threshold: Optional[float],
) -> plt.Figure:
    if "full" not in rows_by_variant:
        raise ValueError("Heatmap requires the 'full' variant as reference.")

    ref_row = rows_by_variant["full"]
    heatmap = np.zeros((len(variants), len(metrics) * len(SPLITS)), dtype=np.float64)
    annot: List[List[str]] = [["" for _ in range(heatmap.shape[1])] for _ in range(heatmap.shape[0])]
    col_labels: List[str] = []

    for split in SPLITS:
        for metric in metrics:
            col_labels.append(f"{pretty_split_name(split)}\n{pretty_metric_display(metric, hotspot_threshold)}")

    for row_idx, variant in enumerate(variants):
        current = rows_by_variant[variant]
        col_idx = 0
        for split in SPLITS:
            for metric in metrics:
                cand = get_metric_value(current, split, metric)
                ref = get_metric_value(ref_row, split, metric)
                value = 0.0 if variant == "full" else signed_gain(cand, ref, metric)
                heatmap[row_idx, col_idx] = value

                marker = ""
                if variant != "full":
                    marker, _ = significance_marker_and_color(sig_map.get((variant, split, metric)))
                annot[row_idx][col_idx] = f"{value:+.3f}{marker}"
                col_idx += 1

    vmax = float(np.nanmax(np.abs(heatmap))) if np.isfinite(np.nanmax(np.abs(heatmap))) else 1.0
    vmax = max(vmax, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(11.0, heatmap.shape[1] * 1.35), max(4.6, heatmap.shape[0] * 0.86 + 2.2)))
    im = ax.imshow(heatmap, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(heatmap.shape[1]))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants], fontsize=11)
    ax.set_title("Backbone Ablation: Signed Gain vs Full (positive is better)", fontsize=16, fontweight="bold", pad=10)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            value = heatmap[i, j]
            text_color = "white" if abs(value) > 0.45 * vmax else "#111827"
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Signed gain relative to Full", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    note = "Signed gain uses metric direction: for MAE/MSE/RMSE lower is better; for R2/SSIM/CSI/F1 higher is better."
    if hotspot_threshold is not None:
        note += f" CSI/F1 use threshold = {hotspot_threshold:.4f}."
    note += " '*' means p<0.05."
    fig.text(0.01, 0.01, note, fontsize=9.5, color="#374151")
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    return fig


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
    requested_order = [name for name in canonical_metric_list(args.variant_order) if name not in EXCLUDED_VARIANTS]

    rows = load_csv_rows(summary_csv)
    rows_by_variant = {
        name: row
        for name, row in ensure_rows_by_key(rows, "experiment").items()
        if name not in EXCLUDED_VARIANTS
    }

    if "full" not in rows_by_variant:
        raise RuntimeError("The summary file does not contain 'full', cannot build ablation reference.")

    variants = reorder_variants(list(rows_by_variant.keys()), requested_order)
    sig_map = load_significance_map(significance_json)
    hotspot_threshold = resolve_hotspot_threshold_label(rows_by_variant, args.hotspot_threshold_label, args.threshold)
    output_threshold = (
        float(hotspot_threshold)
        if hotspot_threshold is not None
        else float(args.threshold if args.threshold is not None else DEFAULT_PRIMARY_HOTSPOT_THRESHOLD)
    )
    ranking_rows = prepare_ranking_table_rows(rows_by_variant, variants)

    fig_bars = make_metric_bar_figure(rows_by_variant, variants, metrics, sig_map, hotspot_threshold)
    bars_archive_prefix, bars_latest_prefix = resolve_output_prefixes(
        output_dir,
        args.output_prefix,
        "SCI_Backbone_Ablation_Metric_Bars",
        output_threshold,
        len(variants),
    )
    bars_png, bars_pdf = save_figure(fig_bars, bars_archive_prefix, dpi=args.dpi)
    update_latest_outputs(bars_archive_prefix, bars_latest_prefix)

    fig_heatmap = make_gain_heatmap_figure(rows_by_variant, variants, metrics, sig_map, hotspot_threshold)
    heat_archive_prefix, heat_latest_prefix = resolve_output_prefixes(
        output_dir,
        args.output_prefix,
        "SCI_Backbone_Ablation_Gain_Heatmap",
        output_threshold,
        len(variants),
    )
    heat_png, heat_pdf = save_figure(fig_heatmap, heat_archive_prefix, dpi=args.dpi)
    update_latest_outputs(heat_archive_prefix, heat_latest_prefix)

    table_archive_prefix, table_latest_prefix = resolve_output_prefixes(
        output_dir,
        args.output_prefix,
        "SCI_Backbone_Ablation_Performance_Table",
        output_threshold,
        len(ranking_rows),
    )
    table_png, table_pdf = plot_backbone_ranking_table_figure(
        ranking_rows,
        table_archive_prefix,
        threshold=output_threshold,
        dpi=args.dpi,
    )
    update_latest_outputs(table_archive_prefix, table_latest_prefix)

    print(f"Loaded summary: {summary_csv}")
    print(f"Loaded significance: {significance_json}")
    print(f"Saved metric-bar archive to: {bars_png}")
    print(f"Saved metric-bar archive to: {bars_pdf}")
    if bars_latest_prefix != bars_archive_prefix:
        print(f"Updated metric-bar latest to: {bars_latest_prefix.with_suffix('.png')}")
        print(f"Updated metric-bar latest to: {bars_latest_prefix.with_suffix('.pdf')}")
    print(f"Saved heatmap archive to: {heat_png}")
    print(f"Saved heatmap archive to: {heat_pdf}")
    if heat_latest_prefix != heat_archive_prefix:
        print(f"Updated heatmap latest to: {heat_latest_prefix.with_suffix('.png')}")
        print(f"Updated heatmap latest to: {heat_latest_prefix.with_suffix('.pdf')}")
    print(f"Saved performance-table archive to: {table_png}")
    print(f"Saved performance-table archive to: {table_pdf}")
    if table_latest_prefix != table_archive_prefix:
        print(f"Updated performance-table latest to: {table_latest_prefix.with_suffix('.png')}")
        print(f"Updated performance-table latest to: {table_latest_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
