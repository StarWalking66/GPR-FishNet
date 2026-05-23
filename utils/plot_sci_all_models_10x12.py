import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    load_checkpoint_rows,
    threshold_metric_key,
    threshold_tag,
)


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


OUTPUT_DIR = PROJECT_ROOT / "model_outcomes" / "sci_plots_grand_comparison"
HIGHLIGHT_LABELS = {"GPR-FishNet (ours)"}


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
    fixed_seed: int | None,
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


def prediction_source_tag(prediction_source: str, fixed_seed: int | None) -> str:
    if prediction_source == "mean":
        return "predsrc_mean"
    if prediction_source == "selected_seed":
        return "predsrc_selected_seed"
    if prediction_source == "fixed_seed":
        if fixed_seed is None:
            raise ValueError("fixed_seed must be provided when prediction_source='fixed_seed'")
        return f"predsrc_seed_{fixed_seed}"
    raise ValueError(f"Unsupported prediction_source: {prediction_source}")


def get_train_99p(data_dir: Path) -> float:
    params_path = data_dir / "ais_norm_params.npy"
    if params_path.exists():
        params = np.load(params_path, allow_pickle=True).item()
        return float(params.get("train_99p", 1.0))
    return 1.0


def post_process_predictions(pred: np.ndarray, train_99p: float, noise_threshold: float = 0.05) -> np.ndarray:
    physical_pred = np.asarray(pred, dtype=np.float32) * float(train_99p)
    physical_pred[physical_pred < noise_threshold] = 0.0
    return physical_pred


def normalize_spatiotemporal_array(array_path: Path) -> np.ndarray:
    array = np.squeeze(np.load(array_path))
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected [T, H, W] after squeeze for {array_path}, got {array.shape}")
    return array.astype(np.float32)


def prepare_ranked_rows(
    model_outcomes_dir: Path,
    mask_path: Path,
    checkpoint_glob: str,
    threshold: float,
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


def find_rollout_targets_path(model_outcomes_dir: Path, ranked_rows: Sequence[Dict]) -> Path:
    for row in ranked_rows:
        target_path = model_outcomes_dir / row["model_dir"] / "rollout_2024_targets.npy"
        if target_path.exists():
            return target_path
    raise FileNotFoundError("No rollout_2024_targets.npy file found in the discovered checkpoint directories.")


def collect_visual_rows(
    data_dir: Path,
    model_outcomes_dir: Path,
    ranked_rows: Sequence[Dict],
    noise_threshold: float,
    prediction_source: str,
    fixed_seed: int | None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, str]]]:
    train_99p = get_train_99p(data_dir)

    ais_val_path = data_dir / "ais_val.npy"
    if not ais_val_path.exists():
        raise FileNotFoundError(f"History source not found: {ais_val_path}")

    targets_path = find_rollout_targets_path(model_outcomes_dir, ranked_rows)
    targets_2024 = normalize_spatiotemporal_array(targets_path) * train_99p
    horizon = int(targets_2024.shape[0])

    history_full = normalize_spatiotemporal_array(ais_val_path) * train_99p
    if history_full.shape[0] < horizon:
        raise ValueError(
            f"History series is shorter than the rollout horizon: {history_full.shape[0]} < {horizon}"
        )
    history_2023 = history_full[-horizon:]

    visual_rows: List[Dict[str, np.ndarray]] = [
        {"label": "History (2023)", "data": history_2023},
        {"label": "G.T. (2024)", "data": targets_2024},
    ]
    skipped_rows: List[Dict[str, str]] = []

    for row in ranked_rows:
        checkpoint_dir = model_outcomes_dir / row["model_dir"]
        try:
            pred_path = resolve_prediction_artifact_path(
                checkpoint_dir=checkpoint_dir,
                artifact_name="rollout_2024_preds.npy",
                prediction_source=prediction_source,
                fixed_seed=fixed_seed,
            )
        except Exception as exc:
            skipped_rows.append(
                {"model_dir": row["model_dir"], "reason": str(exc)}
            )
            continue
        if not pred_path.exists():
            skipped_rows.append(
                {"model_dir": row["model_dir"], "reason": f"missing {pred_path.name} at {pred_path.parent}"}
            )
            continue

        preds = post_process_predictions(
            normalize_spatiotemporal_array(pred_path),
            train_99p=train_99p,
            noise_threshold=noise_threshold,
        )
        if preds.shape != targets_2024.shape:
            skipped_rows.append(
                {
                    "model_dir": row["model_dir"],
                    "reason": f"rollout prediction shape {preds.shape} does not match targets {targets_2024.shape}",
                }
            )
            continue

        visual_rows.append(
            {
                "label": row["model_label"],
                "data": preds,
                "model_dir": row["model_dir"],
            }
        )

    return visual_rows, skipped_rows


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


def figure_size_for_grid(num_rows: int, num_cols: int) -> Tuple[float, float]:
    return max(18.0, num_cols * 2.2), max(8.0, num_rows * 2.4)


def render_figure(
    visual_rows: Sequence[Dict[str, np.ndarray]],
    data_dir: Path,
    output_prefix: Path,
    dpi: int = 300,
) -> None:
    if not visual_rows:
        raise RuntimeError("No visual rows are available for plotting.")

    num_rows = len(visual_rows)
    num_cols = int(visual_rows[0]["data"].shape[0])
    _, height, width = visual_rows[0]["data"].shape

    fig_w, fig_h = figure_size_for_grid(num_rows, num_cols)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        num_rows,
        num_cols,
        figure=fig,
        left=0.06,
        right=0.91,
        bottom=0.05,
        top=0.96,
        wspace=0.25,
        hspace=0.40,
    )

    vmax = max(float(np.max(row["data"])) for row in visual_rows)
    if vmax <= 0:
        vmax = 0.1
    norm_hot = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)
    cmap = "jet"
    x_ticks, x_labels, y_ticks, y_labels = build_geo_ticks(data_dir, height, width)

    im = None
    for row_idx, row in enumerate(visual_rows):
        for col_idx in range(num_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(row["data"][col_idx], cmap=cmap, norm=norm_hot, origin="lower")

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

                is_highlight = row["label"] in HIGHLIGHT_LABELS
                ax.set_ylabel(
                    row["label"],
                    fontsize=15,
                    fontweight="heavy" if is_highlight else "bold",
                    color="darkred" if is_highlight else "black",
                    labelpad=12,
                )
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", direction="in", length=2, width=0.5, colors="#999999")

            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.8)

            if row_idx == 0:
                ax.set_title(f"Month {col_idx + 1:02d}", fontsize=14, fontweight="bold", pad=12)

    if im is None:
        raise RuntimeError("Failed to render any image tiles.")

    cbar_ax = fig.add_axes([0.925, 0.05, 0.012, 0.91])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("AIS Fishing Effort (hours/day)", fontsize=16, fontweight="bold", labelpad=15)
    cbar.ax.tick_params(labelsize=14)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)

    print(f"Saved PNG to: {png_path}")
    print(f"Saved PDF to: {pdf_path}")


def resolve_output_prefixes(
    output_dir: Path,
    explicit_output_prefix: str,
    threshold: float,
    model_count: int,
    prediction_source_name: str,
) -> Tuple[Path, Path]:
    if explicit_output_prefix:
        output_prefix = Path(explicit_output_prefix)
        if not output_prefix.is_absolute():
            output_prefix = (PROJECT_ROOT / output_prefix).resolve()
        return output_prefix, output_prefix

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_prefix = output_dir / (
        f"SCI_Grand_Comparison_{build_artifact_tag(threshold, model_count, timestamp)}_{prediction_source_name}"
    )
    latest_prefix = output_dir / (
        f"SCI_Grand_Comparison_latest_threshold_{threshold_tag(threshold)}_{prediction_source_name}"
    )
    return archive_prefix, latest_prefix


def update_latest_outputs(archive_prefix: Path, latest_prefix: Path) -> None:
    if archive_prefix == latest_prefix:
        return
    for suffix in (".png", ".pdf"):
        shutil.copy2(archive_prefix.with_suffix(suffix), latest_prefix.with_suffix(suffix))


def print_scan_report(
    ranked_rows: Sequence[Dict],
    metric_skipped_rows: Sequence[Dict[str, str]],
    visual_skipped_rows: Sequence[Dict[str, str]],
) -> None:
    print(f"Ranked checkpoint dirs: {len(ranked_rows)}")
    print(f"Skipped during metrics scan: {len(metric_skipped_rows)}")
    for item in metric_skipped_rows:
        print(f"  - {item['model_dir']}: {item['reason']}")
    print(f"Skipped during visualization scan: {len(visual_skipped_rows)}")
    for item in visual_skipped_rows:
        print(f"  - {item['model_dir']}: {item['reason']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-scan model_outcomes and render a grand comparison grid for rollout predictions."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "ST_FishNet_Features"),
        help="Directory containing ais_val.npy and normalization metadata.",
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
        help="Shared ocean-mask path used to rank models consistently with the ranking scripts.",
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
        "--noise-threshold",
        type=float,
        default=0.05,
        help="Physical-hours threshold used to suppress tiny prediction noise after denormalization.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for PNG/PDF export.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any checkpoint is skipped during metrics scan or visualization scan.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prediction_source == "fixed_seed" and args.fixed_seed is None:
        raise ValueError("--fixed-seed must be provided when --prediction-source fixed_seed is used.")
    if args.prediction_source != "fixed_seed" and args.fixed_seed is not None:
        print("Ignoring --fixed-seed because --prediction-source is not fixed_seed.")

    data_dir = Path(args.data_dir)
    model_outcomes_dir = Path(args.model_outcomes_dir)
    mask_path = Path(args.mask_path)
    output_dir = Path(args.output_dir)

    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    if not model_outcomes_dir.is_absolute():
        model_outcomes_dir = (PROJECT_ROOT / model_outcomes_dir).resolve()
    if not mask_path.is_absolute():
        mask_path = (PROJECT_ROOT / mask_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    ranked_rows, metric_skipped_rows = prepare_ranked_rows(
        model_outcomes_dir=model_outcomes_dir,
        mask_path=mask_path,
        checkpoint_glob=args.checkpoint_glob,
        threshold=args.threshold,
    )
    visual_rows, visual_skipped_rows = collect_visual_rows(
        data_dir=data_dir,
        model_outcomes_dir=model_outcomes_dir,
        ranked_rows=ranked_rows,
        noise_threshold=args.noise_threshold,
        prediction_source=args.prediction_source,
        fixed_seed=args.fixed_seed,
    )

    if args.strict and (metric_skipped_rows or visual_skipped_rows):
        print_scan_report(ranked_rows, metric_skipped_rows, visual_skipped_rows)
        raise RuntimeError("Strict mode enabled and some checkpoints were skipped.")

    archive_prefix, latest_prefix = resolve_output_prefixes(
        output_dir=output_dir,
        explicit_output_prefix=args.output_prefix,
        threshold=args.threshold,
        model_count=max(len(visual_rows) - 2, 0),
        prediction_source_name=prediction_source_tag(args.prediction_source, args.fixed_seed),
    )
    archive_prefix.parent.mkdir(parents=True, exist_ok=True)
    render_figure(visual_rows, data_dir, archive_prefix, dpi=args.dpi)
    update_latest_outputs(archive_prefix, latest_prefix)

    print(f"Saved comparison archive to: {archive_prefix.with_suffix('.png')}")
    print(f"Saved comparison archive to: {archive_prefix.with_suffix('.pdf')}")
    if latest_prefix != archive_prefix:
        print(f"Updated latest comparison to: {latest_prefix.with_suffix('.png')}")
        print(f"Updated latest comparison to: {latest_prefix.with_suffix('.pdf')}")
    print_scan_report(ranked_rows, metric_skipped_rows, visual_skipped_rows)


if __name__ == "__main__":
    main()
