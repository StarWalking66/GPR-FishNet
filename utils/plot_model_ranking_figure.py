import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, rcParams


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
    load_checkpoint_rows,
    load_json,
    threshold_metric_key,
    threshold_tag,
)


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


OUTPUT_DIR = PROJECT_ROOT / "model_outcomes" / "sci_plots_grand_comparison"
DOWN_ARROW = "\u2193"
UP_ARROW = "\u2191"
OURS_MODEL_LABELS = {
    "checkpoints_gpr_fishnet_final": "GPR-FishNet (ours)",
}
OURS_LABELS = set(OURS_MODEL_LABELS.values())
MODEL_SOURCES = {
    "checkpoints_convlstm_baseline": "NIPS'15",
    "checkpoints_exprecast_final": "ICLR'26",
    "checkpoints_gpr_fishnet_final": "This work",
    "checkpoints_ksa_predrnn_final": "This work",
    "checkpoints_pfgnet_final": "CVPR'26",
    "checkpoints_predrnn_baseline": "NIPS'17",
    "checkpoints_predrnn_v2_baseline": "TPAMI'23",
    "checkpoints_seacast_baseline": "Sci Rep'25",
    "checkpoints_swinlstm_baseline": "ICCV'23",
    "checkpoints_timekan_final": "ICLR'25",
}


def build_panel_columns(threshold: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    threshold_label = f"{threshold:.4f}"
    one_step_columns = [
        ("test_one_step_MAE", f"MAE {DOWN_ARROW}"),
        ("test_one_step_MSE", f"MSE {DOWN_ARROW}"),
        ("test_one_step_RMSE", f"RMSE {DOWN_ARROW}"),
        ("test_one_step_R2", f"R\u00b2 {UP_ARROW}"),
        ("test_one_step_SSIM", f"SSIM {UP_ARROW}"),
        (threshold_metric_key("test_one_step", "CSI", threshold), f"CSI@{threshold_label} {UP_ARROW}"),
        (threshold_metric_key("test_one_step", "F1", threshold), f"F1@{threshold_label} {UP_ARROW}"),
    ]
    rollout_columns = [
        ("rollout_2024_MAE", f"MAE {DOWN_ARROW}"),
        ("rollout_2024_MSE", f"MSE {DOWN_ARROW}"),
        ("rollout_2024_RMSE", f"RMSE {DOWN_ARROW}"),
        ("rollout_2024_R2", f"R\u00b2 {UP_ARROW}"),
        ("rollout_2024_SSIM", f"SSIM {UP_ARROW}"),
        (threshold_metric_key("rollout_2024", "CSI", threshold), f"CSI@{threshold_label} {UP_ARROW}"),
        (threshold_metric_key("rollout_2024", "F1", threshold), f"F1@{threshold_label} {UP_ARROW}"),
    ]
    return one_step_columns, rollout_columns


def prepare_rows(
    model_outcomes_dir: Path = MODEL_OUTCOMES_DIR,
    mask_path: Path = MASK_PATH,
    checkpoint_glob: str = CHECKPOINT_GLOB,
    threshold: float = THRESHOLD,
) -> Tuple[List[Dict], List[Dict[str, str]]]:
    rows, skipped_rows = load_checkpoint_rows(
        model_outcomes_dir=model_outcomes_dir,
        mask_path=mask_path,
        threshold=threshold,
        checkpoint_glob=checkpoint_glob,
        return_skipped=True,
    )
    if not rows:
        raise RuntimeError(
            f"No complete checkpoint results were found under {model_outcomes_dir} with pattern {checkpoint_glob}"
        )

    for row in rows:
        if row["model_dir"] in OURS_MODEL_LABELS:
            row["model_label"] = OURS_MODEL_LABELS[row["model_dir"]]
        row["model_source"] = MODEL_SOURCES.get(row["model_dir"], "-")

    attach_metric_stds(rows, model_outcomes_dir, threshold)

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


def blend_with_white(color: str, strength: float) -> Tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color))
    white = np.ones(3)
    strength = np.clip(strength, 0.0, 1.0)
    return tuple(white * (1.0 - strength) + base * strength)


def rank_fill(rank: int, total: int) -> Tuple[float, float, float]:
    frac = 1.0 if total <= 1 else 1.0 - (rank - 1) / (total - 1)
    return blend_with_white("#2e7d32", 0.10 + 0.25 * frac)


def medal_fill(rank: int) -> Tuple[float, float, float]:
    if rank == 1:
        return mcolors.to_rgb("#f4d35e")
    if rank == 2:
        return mcolors.to_rgb("#d9dde3")
    if rank == 3:
        return mcolors.to_rgb("#e6b89c")
    return mcolors.to_rgb("#f6f7fb")


def format_value(metric_key: str, value: float, row: Dict) -> str:
    if metric_key.endswith("_rank") or metric_key == "overall_rank":
        return str(int(value))
    std_key = f"{metric_key}_std"
    std_value = row.get(std_key)
    if isinstance(std_value, (int, float, np.integer, np.floating)):
        return f"{value:.4f}\u00b1{float(std_value):.4f}"
    return f"{value:.4f}"


def figure_height_for_rows(num_rows: int) -> float:
    return max(9.8, 6.1 + num_rows * 0.44)


def summarize_present_models(rows: Sequence[Dict]) -> str:
    present_ours = [row["model_label"] for row in rows if row["model_label"] in OURS_LABELS]
    if not present_ours:
        return "Highlighted rows denote custom/internal models when present."
    return "Highlighted rows: " + ", ".join(present_ours) + "."


def _read_std_payload(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    payload = load_json(path)
    std_payload = payload.get("std")
    if not isinstance(std_payload, dict):
        return {}
    result: Dict[str, float] = {}
    for key, value in std_payload.items():
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def attach_metric_stds(rows: Sequence[Dict], model_outcomes_dir: Path, threshold: float) -> None:
    threshold_file_tag = threshold_tag(threshold)
    for row in rows:
        checkpoint_dir = model_outcomes_dir / row["model_dir"]

        for split_name in ("test_one_step", "rollout_2024"):
            split_std_map = _read_std_payload(checkpoint_dir / f"{split_name}_metrics.json")
            for metric_name, metric_std in split_std_map.items():
                row[f"{split_name}_{metric_name}_std"] = metric_std

            threshold_std_map = _read_std_payload(
                checkpoint_dir / f"{split_name}_metrics_threshold_{threshold_file_tag}.json"
            )
            for metric_name, metric_std in threshold_std_map.items():
                row[f"{threshold_metric_key(split_name, metric_name, threshold)}_std"] = metric_std


def draw_panel(
    ax,
    rows: Sequence[Dict],
    panel_title: str,
    metric_columns: Sequence[Tuple[str, str]],
    panel_tag: str,
) -> None:
    ax.set_axis_off()
    n_rows = len(rows)

    col_defs = [
        ("model_label", "Model", 3.25),
        ("model_source", "Source", 1.40),
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
    ours_row_fill = "#fff5f5"

    metric_name_set = {name for name, _ in metric_columns}

    for row_idx, row in enumerate(rows):
        y = header_y - (row_idx + 1) * row_h
        base_fill = stripe_fill if row_idx % 2 == 0 else normal_fill
        if row["model_label"] in OURS_LABELS:
            base_fill = ours_row_fill

        x = 0.0
        for col_idx, (metric_key, _, width) in enumerate(col_defs):
            cell_fill = base_fill
            text_color = "#111827"
            fontweight = "normal"

            if metric_key in {"model_label", "model_source"} and row["model_label"] in OURS_LABELS:
                text_color = "#b91c1c"
                fontweight = "bold"
            elif metric_key.endswith("_rank"):
                cell_fill = blend_with_white("#2563eb", 0.08)
            elif metric_key in metric_name_set:
                metric_rank = int(row[f"{metric_key}_rank"])
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
            text = value if metric_key in {"model_label", "model_source"} else format_value(metric_key, value, row)
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

            if col_idx == 0 and row["model_label"] in OURS_LABELS:
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


def plot_model_ranking_figure(
    rows: List[Dict],
    output_prefix: Path,
    threshold: float = THRESHOLD,
    dpi: int = 300,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.4, figure_height_for_rows(len(rows))), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.12, top=0.895, bottom=0.12, left=0.045, right=0.965)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    one_step_columns, rollout_columns = build_panel_columns(threshold)

    draw_panel(ax1, rows, "One-step Forecasting Performance Ranking", one_step_columns, "(a)")
    draw_panel(ax2, rows, "2024 Rollout Forecasting Performance Ranking", rollout_columns, "(b)")

    fig.suptitle(
        f"Model performance ranking under a unified ocean mask ({len(rows)} models)",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    note_line_1 = (
        f"Error metrics are lower-is-better ({DOWN_ARROW}); skill metrics are higher-is-better ({UP_ARROW}). "
        f"Values are reported as mean\u00b1std when multi-seed summaries are available."
    )
    note_line_2 = (
        f"CSI/F1 are recomputed at threshold = {threshold:.4f}; Source denotes original venue/year; rows are ordered by the overall average rank. "
        f"{summarize_present_models(rows)}"
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

    print(f"Saved PNG to: {png_path}")
    print(f"Saved PDF to: {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SCI-style model ranking figure with metric arrows.")
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
        help="Shared ocean-mask path used for threshold metric recomputation.",
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
        help="Optional explicit output prefix without extension. If omitted, the script writes both archive and latest files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Threshold used for CSI/F1 metric display and checkpoint scanning.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=CHECKPOINT_GLOB,
        help="Glob pattern used to discover checkpoint directories.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for PNG/PDF export.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any discovered checkpoint directory is skipped.",
    )
    return parser.parse_args()


def resolve_output_prefixes(
    output_dir: Path,
    explicit_output_prefix: str,
    threshold: float,
    model_count: int,
) -> Tuple[Path, Path]:
    if explicit_output_prefix:
        output_prefix = Path(explicit_output_prefix)
        if not output_prefix.is_absolute():
            output_prefix = (PROJECT_ROOT / output_prefix).resolve()
        return output_prefix, output_prefix

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_prefix = output_dir / f"SCI_Model_Ranking_Figure_{build_artifact_tag(threshold, model_count, timestamp)}"
    latest_prefix = output_dir / f"SCI_Model_Ranking_Figure_latest_threshold_{threshold_tag(threshold)}"
    return archive_prefix, latest_prefix


def update_latest_outputs(archive_prefix: Path, latest_prefix: Path) -> None:
    if archive_prefix == latest_prefix:
        return
    for suffix in (".png", ".pdf"):
        shutil.copy2(archive_prefix.with_suffix(suffix), latest_prefix.with_suffix(suffix))


def print_scan_report(rows: List[Dict], skipped_rows: List[Dict[str, str]]) -> None:
    print(f"Included checkpoint dirs: {len(rows)}")
    print(f"Skipped checkpoint dirs: {len(skipped_rows)}")
    for item in skipped_rows:
        print(f"  - {item['model_dir']}: {item['reason']}")


def main() -> None:
    args = parse_args()
    model_outcomes_dir = Path(args.model_outcomes_dir)
    mask_path = Path(args.mask_path)
    output_dir = Path(args.output_dir)

    if not model_outcomes_dir.is_absolute():
        model_outcomes_dir = (PROJECT_ROOT / model_outcomes_dir).resolve()
    if not mask_path.is_absolute():
        mask_path = (PROJECT_ROOT / mask_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    rows, skipped_rows = prepare_rows(
        model_outcomes_dir=model_outcomes_dir,
        mask_path=mask_path,
        checkpoint_glob=args.checkpoint_glob,
        threshold=args.threshold,
    )
    if args.strict and skipped_rows:
        print_scan_report(rows, skipped_rows)
        raise RuntimeError("Strict mode enabled and some checkpoint directories were skipped.")

    archive_prefix, latest_prefix = resolve_output_prefixes(
        output_dir=output_dir,
        explicit_output_prefix=args.output_prefix,
        threshold=args.threshold,
        model_count=len(rows),
    )
    archive_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_model_ranking_figure(rows, archive_prefix, threshold=args.threshold, dpi=args.dpi)
    update_latest_outputs(archive_prefix, latest_prefix)

    print(f"Saved figure archive to: {archive_prefix.with_suffix('.png')}")
    print(f"Saved figure archive to: {archive_prefix.with_suffix('.pdf')}")
    if latest_prefix != archive_prefix:
        print(f"Updated latest figure to: {latest_prefix.with_suffix('.png')}")
        print(f"Updated latest figure to: {latest_prefix.with_suffix('.pdf')}")
    print_scan_report(rows, skipped_rows)


if __name__ == "__main__":
    main()
