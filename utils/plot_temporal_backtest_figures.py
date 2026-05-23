from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, rcParams
from matplotlib.lines import Line2D


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_ablation_common import load_csv_rows, save_figure, set_publication_style  # noqa: E402


DEFAULT_BACKTEST_DIR = PROJECT_ROOT / "model_outcomes" / "checkpoints_gpr_fishnet_backtest"
DEFAULT_SUMMARY_CSV = DEFAULT_BACKTEST_DIR / "temporal_backtest_summary.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_BACKTEST_DIR / "figures"
DEFAULT_COMPARE_OUTPUT_DIR = PROJECT_ROOT / "model_outcomes" / "temporal_backtest_comparison" / "figures"
DEFAULT_PRIMARY_HOTSPOT_THRESHOLD = 0.2755
DEFAULT_WINDOW_ORDER = ["window_2022", "window_2023", "window_2024"]
DEFAULT_METRICS = ["RMSE", "R2", "CSI", "F1"]
SPLITS = ["test_one_step", "rollout_12m"]
LINE_PNG_DPI = 600
MIN_FULL_WIDTH_PNG_PX = 3740
SPLIT_LABELS = {
    "test_one_step": "One-step",
    "rollout_12m": "12-month Rollout",
}
SPLIT_COLORS = {
    "test_one_step": "#0072B2",
    "rollout_12m": "#E69F00",
}
SPLIT_LINESTYLES = {
    "test_one_step": "-",
    "rollout_12m": "--",
}
OKABE_ITO_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#009E73",  # bluish green
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]
MODEL_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
DOWN_ARROW = "\u2193"
UP_ARROW = "\u2191"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
COMPARE_MODEL_ROOTS = {
    "gpr_fishnet": DEFAULT_BACKTEST_DIR,
    "predrnn": PROJECT_ROOT / "model_outcomes" / "checkpoints_predrnn_backtest",
    "pfgnet": PROJECT_ROOT / "model_outcomes" / "checkpoints_pfgnet_backtest",
}
COMPARE_MODEL_LABELS = {
    "gpr_fishnet": "GPR-FishNet (ours)",
    "predrnn": "PredRNN",
    "pfgnet": "PFGNet",
}
COMPARE_MODEL_COLORS = {
    "gpr_fishnet": OKABE_ITO_COLORS[0],
    "predrnn": OKABE_ITO_COLORS[1],
    "pfgnet": OKABE_ITO_COLORS[2],
}
COMPARE_MODEL_MARKERS = {
    "gpr_fishnet": MODEL_MARKERS[0],
    "predrnn": MODEL_MARKERS[1],
    "pfgnet": MODEL_MARKERS[2],
}
OURS_COMPARE_KEYS = {"gpr_fishnet"}
SCI_TEXT_COLOR = "#111827"
SCI_MUTED_COLOR = "#4b5563"
SCI_LIGHT_RULE = "#e5e7eb"
SCI_MID_RULE = "#9ca3af"
SCI_HEADER_FILL = "#f3f4f6"
SCI_GROUP_FILL = "#fafafa"
SCI_OURS_FILL = "#fff5f6"


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def threshold_tag(hotspot_threshold: float) -> str:
    return f"{hotspot_threshold:.10f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures for temporal backtest windows.")
    parser.add_argument(
        "--compare-models",
        type=str,
        default="",
        help="Optional comma-separated model keys for multi-model comparison, e.g. gpr_fishnet,predrnn,pfgnet.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="Optional explicit summary CSV path. If omitted, the script auto-resolves from --threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for figure exports.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional explicit output prefix without extension. If omitted, archive and latest outputs are written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold selector. If set to a non-primary threshold, the script auto-loads threshold-specific summaries.",
    )
    parser.add_argument(
        "--window-order",
        type=str,
        default=",".join(DEFAULT_WINDOW_ORDER),
        help="Comma-separated window order, e.g. window_2022,window_2023,window_2024.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics for the trend figure.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for PNG/PDF export.")
    return parser.parse_args()


def resolve_summary_path(summary_csv_raw: str, requested_threshold: Optional[float]) -> Path:
    if summary_csv_raw:
        summary_csv = Path(summary_csv_raw)
    else:
        if requested_threshold is None or abs(float(requested_threshold) - DEFAULT_PRIMARY_HOTSPOT_THRESHOLD) < 1e-12:
            summary_csv = DEFAULT_SUMMARY_CSV
        else:
            summary_csv = DEFAULT_BACKTEST_DIR / f"temporal_backtest_summary_threshold_{threshold_tag(float(requested_threshold))}.csv"
    if summary_csv.is_absolute():
        return summary_csv
    return (PROJECT_ROOT / summary_csv).resolve()


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_name_list(raw: str) -> List[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def resolve_default_output_dir(output_dir_raw: str, compare_mode: bool) -> Path:
    if compare_mode and output_dir_raw == str(DEFAULT_OUTPUT_DIR):
        return DEFAULT_COMPARE_OUTPUT_DIR
    return resolve_output_path(Path(output_dir_raw))


def build_artifact_tag(hotspot_threshold: float, window_count: int, timestamp: str) -> str:
    return f"threshold_{threshold_tag(hotspot_threshold)}_{window_count}windows_{timestamp}"


def resolve_output_prefixes(
    output_dir: Path,
    explicit_output_prefix: str,
    figure_label: str,
    hotspot_threshold: float,
    window_count: int,
) -> Tuple[Path, Path]:
    suffix = f"_{figure_label}"
    if explicit_output_prefix:
        output_prefix = Path(explicit_output_prefix)
        if not output_prefix.is_absolute():
            output_prefix = (PROJECT_ROOT / output_prefix).resolve()
        prefixed = output_prefix.with_name(f"{output_prefix.name}{suffix}")
        return prefixed, prefixed

    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    archive_prefix = output_dir / f"{figure_label}_{build_artifact_tag(hotspot_threshold, window_count, timestamp)}"
    latest_prefix = output_dir / f"{figure_label}_latest_threshold_{threshold_tag(hotspot_threshold)}"
    return archive_prefix, latest_prefix


def update_latest_outputs(archive_prefix: Path, latest_prefix: Path) -> None:
    if archive_prefix == latest_prefix:
        return
    for suffix in (".png", ".pdf"):
        shutil.copy2(archive_prefix.with_suffix(suffix), latest_prefix.with_suffix(suffix))


def save_vector_line_figure(fig: plt.Figure, output_prefix: Path, dpi: int = 300) -> Tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    png_width_px = int(round(bbox.width * LINE_PNG_DPI))
    if png_width_px < MIN_FULL_WIDTH_PNG_PX:
        raise ValueError(
            f"Tight PNG export would be {png_width_px}px wide; "
            f"expected at least {MIN_FULL_WIDTH_PNG_PX}px for full-width output."
        )
    fig.savefig(png_path, dpi=LINE_PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def reorder_windows(available: Sequence[str], requested: Sequence[str]) -> List[str]:
    requested_unique: List[str] = []
    for name in requested:
        if name and name not in requested_unique:
            requested_unique.append(name)
    if not requested_unique:
        return list(available)
    return [name for name in requested_unique if name in available]


def model_color(model_key: str, model_index: int = 0) -> str:
    return COMPARE_MODEL_COLORS.get(model_key, OKABE_ITO_COLORS[model_index % len(OKABE_ITO_COLORS)])


def model_marker(model_key: str, model_index: int = 0) -> str:
    return COMPARE_MODEL_MARKERS.get(model_key, MODEL_MARKERS[model_index % len(MODEL_MARKERS)])


def resolve_compare_summary_path(model_key: str, requested_threshold: Optional[float]) -> Path:
    if model_key not in COMPARE_MODEL_ROOTS:
        raise RuntimeError(f"Unsupported compare model key: {model_key}")
    root_dir = COMPARE_MODEL_ROOTS[model_key]
    if requested_threshold is None or abs(float(requested_threshold) - DEFAULT_PRIMARY_HOTSPOT_THRESHOLD) < 1e-12:
        return root_dir / "temporal_backtest_summary.csv"
    return root_dir / f"temporal_backtest_summary_threshold_{threshold_tag(float(requested_threshold))}.csv"


def resolve_hotspot_threshold(rows_by_window: Dict[str, Dict], requested_threshold: Optional[float]) -> Optional[float]:
    if requested_threshold is not None:
        return float(requested_threshold)

    detected_values: List[float] = []
    for row in rows_by_window.values():
        value = row.get("threshold_norm")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not any(abs(parsed - existing) < 1e-12 for existing in detected_values):
            detected_values.append(parsed)

    if len(detected_values) == 1:
        return detected_values[0]
    return None


def pretty_metric_display(metric_name: str, hotspot_threshold: Optional[float]) -> str:
    metric_name = metric_name.upper()
    if metric_name == "R2":
        label = "R\u00b2"
    else:
        label = metric_name
    if hotspot_threshold is not None and metric_name in {"CSI", "F1"}:
        return f"{label}@{hotspot_threshold:.4f}"
    return label


def build_table_columns(hotspot_threshold: Optional[float]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
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
        ("rollout_12m_MAE", f"MAE {DOWN_ARROW}"),
        ("rollout_12m_MSE", f"MSE {DOWN_ARROW}"),
        ("rollout_12m_RMSE", f"RMSE {DOWN_ARROW}"),
        ("rollout_12m_R2", f"R\u00b2 {UP_ARROW}"),
        ("rollout_12m_SSIM", f"SSIM {UP_ARROW}"),
        ("rollout_12m_CSI", f"{pretty_metric_display('CSI', hotspot_threshold)} {UP_ARROW}"),
        ("rollout_12m_F1", f"{pretty_metric_display('F1', hotspot_threshold)} {UP_ARROW}"),
    ]
    return one_step_columns, rollout_columns


def prepare_table_rows(rows_by_window: Dict[str, Dict], windows: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    for window in windows:
        row = dict(rows_by_window[window])
        row["window"] = window
        row["window_label"] = str(row.get("window_label", row.get("test_year", window)))
        rows.append(row)
    rows.sort(key=lambda item: int(item.get("test_year", 0)))
    return rows


def format_table_value(metric_key: str, value: float, row: Dict) -> str:
    std_key = f"{metric_key}_std"
    std_value = row.get(std_key)
    if isinstance(std_value, (int, float, np.integer, np.floating)):
        return f"{value:.4f}\u00b1{float(std_value):.4f}"
    return f"{value:.4f}"


def metric_name_from_key(metric_key: str) -> str:
    for split in SPLITS:
        prefix = f"{split}_"
        if metric_key.startswith(prefix):
            return metric_key.replace(prefix, "", 1)
    return metric_key


def best_comparison_cells(rows: Sequence[Dict], metric_columns: Sequence[Tuple[str, str]]) -> set:
    best_cells = set()
    windows = sorted({int(row["window_order"]) for row in rows})
    for window_order in windows:
        window_rows = [row for row in rows if int(row["window_order"]) == window_order]
        for metric_key, _ in metric_columns:
            values: List[Tuple[Dict, float]] = []
            for row in window_rows:
                try:
                    values.append((row, float(row[metric_key])))
                except (KeyError, TypeError, ValueError):
                    continue
            if not values:
                continue

            metric_name = metric_name_from_key(metric_key)
            if metric_direction(metric_name) == "lower":
                best_value = min(value for _, value in values)
            else:
                best_value = max(value for _, value in values)

            for row, value in values:
                if abs(value - best_value) <= max(1e-12, abs(best_value) * 1e-9):
                    best_cells.add((window_order, str(row["model_key"]), metric_key))
    return best_cells


def comparison_cell_ranks(rows: Sequence[Dict], metric_columns: Sequence[Tuple[str, str]]) -> Dict[Tuple[int, str, str], int]:
    rank_map: Dict[Tuple[int, str, str], int] = {}
    windows = sorted({int(row["window_order"]) for row in rows})
    for window_order in windows:
        window_rows = [row for row in rows if int(row["window_order"]) == window_order]
        for metric_key, _ in metric_columns:
            values: List[Tuple[Dict, float]] = []
            for row in window_rows:
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
                rank_map[(window_order, str(row["model_key"]), metric_key)] = current_rank
    return rank_map


def table_figure_height(num_rows: int) -> float:
    return max(8.0, 4.8 + num_rows * 0.72)


def comparison_table_height(num_rows: int) -> float:
    return max(9.8, 6.1 + num_rows * 0.44)


def format_panel_tag(panel_index: int) -> str:
    return f"({chr(ord('a') + panel_index)})"


def add_axes_panel_tags(
    fig: plt.Figure,
    axes,
    num_panels: int,
    *,
    x_offset: float = -0.034,
    y_offset: float = 0.006,
    fontsize: float = 14.5,
) -> None:
    flat_axes = np.asarray(axes).ravel()
    for panel_index, ax in enumerate(flat_axes[:num_panels]):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + x_offset,
            bbox.y1 + y_offset,
            format_panel_tag(panel_index),
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color=SCI_TEXT_COLOR,
        )


def draw_table_panel(
    ax,
    rows: Sequence[Dict],
    panel_title: str,
    metric_columns: Sequence[Tuple[str, str]],
    panel_tag: str,
) -> None:
    ax.set_axis_off()
    n_rows = len(rows)
    col_defs = [("window_label", "Window", 1.55)] + [(metric_key, metric_label, 1.90) for metric_key, metric_label in metric_columns]

    total_width = sum(width for _, _, width in col_defs)
    header_h = 1.08
    row_h = 0.92
    title_gap = 0.95
    bottom_gap = 0.18
    total_height = title_gap + header_h + n_rows * row_h + bottom_gap

    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    header_y = total_height - title_gap - header_h
    ax.text(
        0.0,
        total_height - 0.10,
        panel_tag,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        total_width / 2,
        total_height - 0.10,
        panel_title,
        fontsize=16,
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
            facecolor="#1f2937",
            edgecolor="white",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + width / 2,
            header_y + header_h / 2,
            label,
            fontsize=11,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        x0 += width

    border_color = "#d5dae3"
    stripe_fill = "#fbfcfe"
    normal_fill = "#ffffff"
    metric_name_set = {name for name, _ in metric_columns}

    for row_index, row in enumerate(rows):
        y = header_y - (row_index + 1) * row_h
        base_fill = stripe_fill if row_index % 2 == 0 else normal_fill

        x = 0.0
        for metric_key, _, width in col_defs:
            rect = patches.Rectangle(
                (x, y),
                width,
                row_h,
                facecolor=base_fill,
                edgecolor=border_color,
                linewidth=0.8,
            )
            ax.add_patch(rect)

            value = row[metric_key]
            text = value if metric_key == "window_label" else format_table_value(metric_key, value, row)
            ha = "left" if metric_key == "window_label" else "center"
            x_text = x + 0.12 if metric_key == "window_label" else x + width / 2
            text_fontsize = 9.4 if metric_key in metric_name_set else 10.5
            ax.text(
                x_text,
                y + row_h / 2,
                text,
                fontsize=text_fontsize,
                color="#111827",
                ha=ha,
                va="center",
            )
            x += width

    ax.add_patch(
        patches.Rectangle(
            (0.0, header_y - n_rows * row_h),
            total_width,
            header_h + n_rows * row_h,
            fill=False,
            edgecolor="#9ca3af",
            linewidth=1.0,
        )
    )


def plot_performance_table_figure(
    rows: List[Dict],
    output_prefix: Path,
    threshold: float,
    dpi: int,
) -> Tuple[Path, Path]:
    fig = plt.figure(figsize=(18, table_figure_height(len(rows))), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.22, top=0.92, bottom=0.045, left=0.035, right=0.985)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    one_step_columns, rollout_columns = build_table_columns(threshold)

    draw_table_panel(ax1, rows, "One-step Forecasting Performance Table", one_step_columns, "(a)")
    draw_table_panel(ax2, rows, "12-month Rollout Forecasting Performance Table", rollout_columns, "(b)")

    fig.suptitle(
        f"Temporal Backtest Performance Table under Fixed Threshold ({len(rows)} Windows)",
        fontsize=20,
        fontweight="bold",
        y=0.975,
    )
    return save_figure(fig, output_prefix, dpi=dpi)


def prepare_comparison_rows(
    model_keys: Sequence[str],
    windows: Sequence[str],
    requested_threshold: Optional[float],
) -> Tuple[List[Dict], float]:
    rows: List[Dict] = []
    detected_threshold: Optional[float] = None

    for model_order, model_key in enumerate(model_keys):
        summary_csv = resolve_compare_summary_path(model_key, requested_threshold)
        if not summary_csv.exists():
            raise RuntimeError(f"Missing comparison summary for {model_key}: {summary_csv}")
        model_rows = load_csv_rows(summary_csv)
        rows_by_window = {str(row.get("window", "")).strip(): row for row in model_rows if str(row.get("window", "")).strip()}

        for window_order, window in enumerate(windows):
            if window not in rows_by_window:
                raise RuntimeError(f"Missing window={window} in comparison summary: {summary_csv}")
            row = dict(rows_by_window[window])
            row["model_key"] = model_key
            row["model_label"] = COMPARE_MODEL_LABELS.get(model_key, model_key)
            row["model_order"] = model_order
            row["window_order"] = window_order
            rows.append(row)

            if detected_threshold is None:
                try:
                    detected_threshold = float(row.get("threshold_norm"))
                except (TypeError, ValueError):
                    detected_threshold = None

    rows.sort(key=lambda row: (int(row["window_order"]), int(row["model_order"])))
    if requested_threshold is not None:
        return rows, float(requested_threshold)
    if detected_threshold is not None:
        return rows, float(detected_threshold)
    return rows, DEFAULT_PRIMARY_HOTSPOT_THRESHOLD


def draw_comparison_table_panel(
    ax,
    rows: Sequence[Dict],
    panel_title: str,
    metric_columns: Sequence[Tuple[str, str]],
    panel_tag: str,
) -> None:
    ax.set_axis_off()
    n_rows = len(rows)
    col_defs = [
        ("window_label", "Window", 1.25),
        ("model_label", "Model", 2.95),
    ] + [(metric_key, metric_label, 1.88) for metric_key, metric_label in metric_columns]

    total_width = sum(width for _, _, width in col_defs)
    header_h = 1.08
    row_h = 0.88
    title_gap = 0.72
    bottom_gap = 0.18
    total_height = title_gap + header_h + n_rows * row_h + bottom_gap

    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    header_y = total_height - title_gap - header_h
    ax.text(
        0.0,
        total_height - 0.10,
        panel_tag,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        total_width / 2,
        total_height - 0.10,
        panel_title,
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
    ours_row_fill = "#fff5f5"
    metric_name_set = {name for name, _ in metric_columns}
    cell_ranks = comparison_cell_ranks(rows, metric_columns)

    for row_index, row in enumerate(rows):
        y = header_y - (row_index + 1) * row_h
        base_fill = stripe_fill if row_index % 2 == 0 else normal_fill
        if row["model_key"] in OURS_COMPARE_KEYS:
            base_fill = ours_row_fill

        x = 0.0
        for col_idx, (metric_key, _, width) in enumerate(col_defs):
            cell_fill = base_fill
            text_color = "#111827"
            fontweight = "normal"

            if metric_key == "model_label" and row["model_key"] in OURS_COMPARE_KEYS:
                text_color = "#b91c1c"
                fontweight = "bold"
            elif metric_key in metric_name_set:
                metric_rank = cell_ranks.get((int(row["window_order"]), str(row["model_key"]), metric_key))
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
            if metric_key == "window_label":
                text = value if int(row["model_order"]) == 0 else ""
            elif metric_key == "model_label":
                text = value
            else:
                text = format_table_value(metric_key, value, row)
            ha = "left" if metric_key in {"window_label", "model_label"} else "center"
            x_text = x + 0.10 if metric_key in {"window_label", "model_label"} else x + width / 2
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

            if col_idx == 0 and row["model_key"] in OURS_COMPARE_KEYS:
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


def export_comparison_table_csv(rows: Sequence[Dict], output_prefix: Path, threshold: float) -> Path:
    import csv

    csv_path = output_prefix.with_suffix(".csv")
    fieldnames = [
        "window_label",
        "model_label",
        "test_one_step_RMSE",
        "test_one_step_RMSE_std",
        "test_one_step_R2",
        "test_one_step_R2_std",
        "test_one_step_CSI",
        "test_one_step_CSI_std",
        "test_one_step_F1",
        "test_one_step_F1_std",
        "rollout_12m_RMSE",
        "rollout_12m_RMSE_std",
        "rollout_12m_R2",
        "rollout_12m_R2_std",
        "rollout_12m_CSI",
        "rollout_12m_CSI_std",
        "rollout_12m_F1",
        "rollout_12m_F1_std",
        "threshold_norm",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, threshold if key == "threshold_norm" else "") for key in fieldnames})
    return csv_path


def plot_comparison_table_figure(
    rows: List[Dict],
    output_prefix: Path,
    threshold: float,
    model_keys: Sequence[str],
    dpi: int,
) -> Tuple[Path, Path]:
    fig = plt.figure(figsize=(17.2, comparison_table_height(len(rows))), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.12, top=0.895, bottom=0.055, left=0.045, right=0.955)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    one_step_columns, rollout_columns = build_table_columns(threshold)

    draw_comparison_table_panel(ax1, rows, "One-step Forecasting Performance Comparison", one_step_columns, "(a)")
    draw_comparison_table_panel(ax2, rows, "12-month Rollout Forecasting Performance Comparison", rollout_columns, "(b)")

    model_labels = [COMPARE_MODEL_LABELS.get(key, key) for key in model_keys]
    fig.suptitle(
        f"Temporal backtest model comparison under a fixed hotspot threshold ({len(model_labels)} models)",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    return save_figure(fig, output_prefix, dpi=dpi)


def make_comparison_trend_figure(
    rows: List[Dict],
    model_keys: Sequence[str],
    windows: Sequence[str],
    metrics: Sequence[str],
    hotspot_threshold: Optional[float],
) -> plt.Figure:
    num_panels = len(metrics)
    ncols = 2
    nrows = int(np.ceil(num_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.15 * nrows), squeeze=False)

    rows_by_model_window: Dict[str, Dict[str, Dict]] = {}
    for row in rows:
        rows_by_model_window.setdefault(str(row["model_key"]), {})[str(row["window"])] = row

    x = np.arange(len(windows), dtype=np.float64)
    x_labels = [str(window).replace("window_", "") for window in windows]
    for panel_index, metric in enumerate(metrics):
        ax = axes[panel_index // ncols][panel_index % ncols]
        panel_values: List[float] = []

        for model_index, model_key in enumerate(model_keys):
            model_rows = rows_by_model_window.get(model_key, {})
            if not model_rows:
                continue
            color = model_color(model_key, model_index)
            marker = model_marker(model_key, model_index)
            is_ours = model_key in OURS_COMPARE_KEYS

            for split in SPLITS:
                values = np.asarray(
                    [float(model_rows[window][f"{split}_{metric}"]) for window in windows],
                    dtype=np.float64,
                )
                stds = np.asarray(
                    [float(model_rows[window].get(f"{split}_{metric}_std", 0.0)) for window in windows],
                    dtype=np.float64,
                )
                panel_values.extend((values - stds).tolist())
                panel_values.extend((values + stds).tolist())
                ax.errorbar(
                    x,
                    values,
                    yerr=stds,
                    marker=marker,
                    markersize=6.6 if is_ours else 5.8,
                    linewidth=2.55 if is_ours else 2.05,
                    elinewidth=1.25 if is_ours else 1.10,
                    capsize=3.4,
                    capthick=1.10,
                    color=color,
                    linestyle=SPLIT_LINESTYLES[split],
                    alpha=0.98 if is_ours else 0.78,
                    markeredgecolor="white",
                    markeredgewidth=0.80,
                    zorder=4 if is_ours else 3,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(pretty_metric_display(metric, hotspot_threshold), fontsize=14.0, fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle="-", linewidth=0.55, alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10.8, width=0.9, length=4.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        if panel_values:
            finite_values = [value for value in panel_values if np.isfinite(value)]
            if finite_values:
                y_min = min(finite_values)
                y_max = max(finite_values)
                y_range = y_max - y_min
                pad = 0.08 * y_range if y_range > 0 else max(abs(y_max) * 0.08, 0.02)
                ax.set_ylim(y_min - pad, y_max + pad)

        direction = metric_direction(metric)
        ax.text(
            0.98,
            0.96,
            "Lower is better" if direction == "lower" else "Higher is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.6,
            color=SCI_MUTED_COLOR,
        )
        if panel_index // ncols == nrows - 1:
            ax.set_xlabel("Test year", fontsize=11.4, fontweight="bold")

    for panel_index in range(num_panels, nrows * ncols):
        axes[panel_index // ncols][panel_index % ncols].axis("off")

    model_handles = [
        Line2D(
            [0],
            [0],
            color=model_color(model_key, model_index),
            linewidth=2.6 if model_key in OURS_COMPARE_KEYS else 2.0,
            marker=model_marker(model_key, model_index),
            markersize=6.4,
            markeredgecolor="white",
            markeredgewidth=0.80,
            label=COMPARE_MODEL_LABELS.get(model_key, model_key),
        )
        for model_index, model_key in enumerate(model_keys)
    ]
    split_handles = [
        Line2D([0], [0], color=SCI_TEXT_COLOR, linewidth=2.2, linestyle="-", label=SPLIT_LABELS["test_one_step"]),
        Line2D([0], [0], color=SCI_TEXT_COLOR, linewidth=2.2, linestyle="--", label=SPLIT_LABELS["rollout_12m"]),
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=len(model_handles),
        frameon=False,
        fontsize=11.0,
        bbox_to_anchor=(0.5, 0.935),
        handlelength=2.2,
        columnspacing=1.6,
    )
    fig.legend(
        handles=split_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10.6,
        bbox_to_anchor=(0.5, 0.895),
        handlelength=2.5,
        columnspacing=1.6,
    )

    fig.suptitle(
        "Temporal backtest metric trends across windows",
        fontsize=16.4,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.86])
    add_axes_panel_tags(fig, axes, num_panels)
    return fig


def metric_direction(metric_name: str) -> str:
    if metric_name.upper() in {"MAE", "MSE", "RMSE"}:
        return "lower"
    return "higher"


def make_metric_trend_figure(
    rows: List[Dict],
    metrics: Sequence[str],
    hotspot_threshold: Optional[float],
) -> plt.Figure:
    num_panels = len(metrics)
    ncols = 2
    nrows = int(np.ceil(num_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 4.9 * nrows), squeeze=False)

    x = np.arange(len(rows), dtype=np.float64)
    x_labels = [str(row.get("window_label", row.get("test_year", ""))) for row in rows]

    for panel_index, metric in enumerate(metrics):
        ax = axes[panel_index // ncols][panel_index % ncols]
        for split in SPLITS:
            key_prefix = split
            values = np.asarray([float(row[f"{key_prefix}_{metric}"]) for row in rows], dtype=np.float64)
            stds = np.asarray([float(row.get(f"{key_prefix}_{metric}_std", 0.0)) for row in rows], dtype=np.float64)
            ax.errorbar(
                x,
                values,
                yerr=stds,
                marker="o",
                markersize=6.4,
                linewidth=2.2,
                elinewidth=1.35,
                capsize=3.4,
                capthick=1.15,
                color=SPLIT_COLORS[split],
                linestyle=SPLIT_LINESTYLES[split],
                label=SPLIT_LABELS[split],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(pretty_metric_display(metric, hotspot_threshold), fontsize=14.4, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)

        direction = metric_direction(metric)
        if direction == "lower":
            ax.annotate(
                "Lower is better",
                xy=(0.98, 0.93),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=10.0,
                color="#475569",
            )
        else:
            ax.annotate(
                "Higher is better",
                xy=(0.98, 0.93),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=10.0,
                color="#475569",
            )

    for panel_index in range(num_panels, nrows * ncols):
        axes[panel_index // ncols][panel_index % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Temporal Backtest Metric Trends Across Windows", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])
    add_axes_panel_tags(fig, axes, num_panels)
    return fig


def main() -> None:
    args = parse_args()
    set_publication_style()
    compare_model_keys = parse_name_list(args.compare_models)
    compare_mode = bool(compare_model_keys)
    output_dir = resolve_default_output_dir(args.output_dir, compare_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    if compare_mode:
        requested_order = [item.strip() for item in args.window_order.split(",") if item.strip()]
        windows = reorder_windows(DEFAULT_WINDOW_ORDER, requested_order)
        comparison_rows, output_threshold = prepare_comparison_rows(compare_model_keys, windows, args.threshold)
        metrics = [item.strip().upper() for item in args.metrics.split(",") if item.strip()]
        compare_archive_prefix, compare_latest_prefix = resolve_output_prefixes(
            output_dir,
            args.output_prefix,
            "SCI_Temporal_Backtest_Model_Comparison_Table",
            output_threshold,
            len(windows),
        )
        compare_png, compare_pdf = plot_comparison_table_figure(
            comparison_rows,
            compare_archive_prefix,
            threshold=output_threshold,
            model_keys=compare_model_keys,
            dpi=args.dpi,
        )
        compare_csv = export_comparison_table_csv(comparison_rows, compare_archive_prefix, output_threshold)
        update_latest_outputs(compare_archive_prefix, compare_latest_prefix)
        if compare_latest_prefix != compare_archive_prefix:
            latest_csv = compare_latest_prefix.with_suffix(".csv")
            latest_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(compare_csv, latest_csv)

        compare_trend_fig = make_comparison_trend_figure(
            comparison_rows,
            model_keys=compare_model_keys,
            windows=windows,
            metrics=metrics,
            hotspot_threshold=output_threshold,
        )
        compare_trend_archive_prefix, compare_trend_latest_prefix = resolve_output_prefixes(
            output_dir,
            args.output_prefix,
            "SCI_Temporal_Backtest_Model_Comparison_Trends",
            output_threshold,
            len(windows),
        )
        compare_trend_png, compare_trend_pdf = save_vector_line_figure(
            compare_trend_fig,
            compare_trend_archive_prefix,
            dpi=args.dpi,
        )
        update_latest_outputs(compare_trend_archive_prefix, compare_trend_latest_prefix)

        print(f"Saved comparison-table archive to: {compare_png}")
        print(f"Saved comparison-table archive to: {compare_pdf}")
        print(f"Saved comparison-table archive to: {compare_csv}")
        if compare_latest_prefix != compare_archive_prefix:
            print(f"Updated comparison-table latest to: {compare_latest_prefix.with_suffix('.png')}")
            print(f"Updated comparison-table latest to: {compare_latest_prefix.with_suffix('.pdf')}")
            print(f"Updated comparison-table latest to: {compare_latest_prefix.with_suffix('.csv')}")
        print(f"Saved comparison-trend archive to: {compare_trend_png}")
        print(f"Saved comparison-trend archive to: {compare_trend_pdf}")
        if compare_trend_latest_prefix != compare_trend_archive_prefix:
            print(f"Updated comparison-trend latest to: {compare_trend_latest_prefix.with_suffix('.png')}")
            print(f"Updated comparison-trend latest to: {compare_trend_latest_prefix.with_suffix('.pdf')}")
        return

    summary_csv = resolve_summary_path(args.summary_csv, args.threshold)

    rows = load_csv_rows(summary_csv)
    rows_by_window = {str(row.get("window", "")).strip(): row for row in rows if str(row.get("window", "")).strip()}
    requested_order = [item.strip() for item in args.window_order.split(",") if item.strip()]
    windows = reorder_windows(list(rows_by_window.keys()), requested_order)
    if not windows:
        raise RuntimeError(f"No matching windows found in summary: {summary_csv}")

    hotspot_threshold = resolve_hotspot_threshold(rows_by_window, args.threshold)
    output_threshold = (
        float(hotspot_threshold)
        if hotspot_threshold is not None
        else float(args.threshold if args.threshold is not None else DEFAULT_PRIMARY_HOTSPOT_THRESHOLD)
    )
    metrics = [item.strip().upper() for item in args.metrics.split(",") if item.strip()]
    table_rows = prepare_table_rows(rows_by_window, windows)

    table_archive_prefix, table_latest_prefix = resolve_output_prefixes(
        output_dir,
        args.output_prefix,
        "SCI_Temporal_Backtest_Performance_Table",
        output_threshold,
        len(table_rows),
    )
    table_png, table_pdf = plot_performance_table_figure(
        table_rows,
        table_archive_prefix,
        threshold=output_threshold,
        dpi=args.dpi,
    )
    update_latest_outputs(table_archive_prefix, table_latest_prefix)

    trend_fig = make_metric_trend_figure(table_rows, metrics, hotspot_threshold)
    trend_archive_prefix, trend_latest_prefix = resolve_output_prefixes(
        output_dir,
        args.output_prefix,
        "SCI_Temporal_Backtest_Metric_Trends",
        output_threshold,
        len(table_rows),
    )
    trend_png, trend_pdf = save_vector_line_figure(trend_fig, trend_archive_prefix, dpi=args.dpi)
    update_latest_outputs(trend_archive_prefix, trend_latest_prefix)

    print(f"Loaded summary: {summary_csv}")
    print(f"Saved performance-table archive to: {table_png}")
    print(f"Saved performance-table archive to: {table_pdf}")
    if table_latest_prefix != table_archive_prefix:
        print(f"Updated performance-table latest to: {table_latest_prefix.with_suffix('.png')}")
        print(f"Updated performance-table latest to: {table_latest_prefix.with_suffix('.pdf')}")
    print(f"Saved trend archive to: {trend_png}")
    print(f"Saved trend archive to: {trend_pdf}")
    if trend_latest_prefix != trend_archive_prefix:
        print(f"Updated trend latest to: {trend_latest_prefix.with_suffix('.png')}")
        print(f"Updated trend latest to: {trend_latest_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
