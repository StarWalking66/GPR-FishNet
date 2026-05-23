from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ST_FishNet_Features"
DEFAULT_MASK_PATH = DEFAULT_DATA_DIR / "all_vars_train_mask_intersection.npy"
DEFAULT_MODEL_OUTCOMES_DIR = PROJECT_ROOT / "model_outcomes"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_OUTCOMES_DIR / "threshold_selection"
DEFAULT_ENV_VARS = ["thetao", "chl", "uo", "vo", "so", "zos", "o2"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
METRICS = ("CSI", "F1", "Precision", "Recall")
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class ModelSpec:
    model_dir: str
    model_label: str
    checkpoint_filename: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "checkpoints_gpr_fishnet_final": ModelSpec(
        model_dir="checkpoints_gpr_fishnet_final",
        model_label="GPR-FishNet (ours)",
        checkpoint_filename="best_gpr_fishnet.pth",
    ),
    "checkpoints_predrnn_baseline": ModelSpec(
        model_dir="checkpoints_predrnn_baseline",
        model_label="PredRNN",
        checkpoint_filename="best_predrnn.pth",
    ),
    "checkpoints_predrnn_v2_baseline": ModelSpec(
        model_dir="checkpoints_predrnn_v2_baseline",
        model_label="PredRNN-V2",
        checkpoint_filename="best_predrnn_v2.pth",
    ),
    "checkpoints_convlstm_baseline": ModelSpec(
        model_dir="checkpoints_convlstm_baseline",
        model_label="ConvLSTM",
        checkpoint_filename="best_convlstm.pth",
    ),
    "checkpoints_pfgnet_final": ModelSpec(
        model_dir="checkpoints_pfgnet_final",
        model_label="PFGNet",
        checkpoint_filename="best_pfgnet.pth",
    ),
    "checkpoints_exprecast_final": ModelSpec(
        model_dir="checkpoints_exprecast_final",
        model_label="ExPreCast",
        checkpoint_filename="best_exprecast.pth",
    ),
    "checkpoints_timekan_final": ModelSpec(
        model_dir="checkpoints_timekan_final",
        model_label="TimeKAN",
        checkpoint_filename="best_timekan.pth",
    ),
    "checkpoints_seacast_baseline": ModelSpec(
        model_dir="checkpoints_seacast_baseline",
        model_label="SeaCast",
        checkpoint_filename="best_seacast.pth",
    ),
    "checkpoints_swinlstm_baseline": ModelSpec(
        model_dir="checkpoints_swinlstm_baseline",
        model_label="SwinLSTM",
        checkpoint_filename="best_swinlstm_baseline.pth",
    ),
}

MODEL_ORDER = list(MODEL_SPECS.keys())


def threshold_tag(threshold: float) -> str:
    return f"{threshold:.10f}".rstrip("0").rstrip(".").replace("-", "neg_").replace(".", "p")


def parse_model_list(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(MODEL_ORDER)
    aliases = {
        "gpr": "checkpoints_gpr_fishnet_final",
        "gpr-fishnet": "checkpoints_gpr_fishnet_final",
        "predrnn": "checkpoints_predrnn_baseline",
        "predrnn-v2": "checkpoints_predrnn_v2_baseline",
        "predrnn_v2": "checkpoints_predrnn_v2_baseline",
        "convlstm": "checkpoints_convlstm_baseline",
        "pfgnet": "checkpoints_pfgnet_final",
        "exprecast": "checkpoints_exprecast_final",
        "timekan": "checkpoints_timekan_final",
        "seacast": "checkpoints_seacast_baseline",
        "swinlstm": "checkpoints_swinlstm_baseline",
    }
    model_dirs: List[str] = []
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        normalized = aliases.get(key.lower(), key)
        if normalized not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{key}'. Use one of: all, {', '.join(MODEL_SPECS)}")
        if normalized not in model_dirs:
            model_dirs.append(normalized)
    if not model_dirs:
        raise ValueError("No models selected.")
    return model_dirs


def parse_seed_list(raw: str) -> List[int]:
    seeds = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    return seeds or list(DEFAULT_SEEDS)


def load_train_99p(data_dir: Path) -> float:
    params_path = data_dir / "ais_norm_params.npy"
    if not params_path.exists():
        return 1.0
    params = np.load(params_path, allow_pickle=True).item()
    return float(params.get("train_99p", 1.0))


def load_mask(mask_path: Path, img_size: Tuple[int, int]) -> np.ndarray:
    if mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        if mask.shape != img_size:
            raise RuntimeError(f"Mask shape {mask.shape} does not match img_size={img_size}.")
        return mask
    return np.ones(img_size, dtype=bool)


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


def extract_valid_values(preds: np.ndarray, targets: np.ndarray, mask_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    preds_3d = normalize_spatiotemporal_array(preds)
    targets_3d = normalize_spatiotemporal_array(targets)
    if preds_3d.shape != targets_3d.shape:
        raise ValueError(f"Pred/target shape mismatch: {preds_3d.shape} vs {targets_3d.shape}")
    valid_mask = expand_mask(mask_2d, preds_3d.shape)
    p = preds_3d[valid_mask]
    t = targets_3d[valid_mask]
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
        "pred_hotspot_ratio": float(np.mean(pred_bin)),
        "target_hotspot_ratio": float(np.mean(target_bin)),
    }


def selection_score(row: Dict[str, float], metric_name: str, target_hotspot_ratio: float = 0.20) -> float:
    metric = metric_name.upper()
    if metric == "CSI_F1_MEAN":
        return 0.5 * (float(row["CSI_mean"]) + float(row["F1_mean"]))
    if metric == "TARGET_HOTSPOT_RATIO":
        return -abs(float(row["target_hotspot_ratio_mean"]) - float(target_hotspot_ratio))
    metric_lookup = {name.upper(): name for name in METRICS}
    if metric not in metric_lookup:
        raise ValueError(f"Unsupported selection metric: {metric_name}")
    return float(row[f"{metric_lookup[metric]}_mean"])


def aggregate_seed_metric_rows(seed_rows: List[Dict[str, float]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for metric_name in (*METRICS, "pred_hotspot_ratio", "target_hotspot_ratio"):
        values = np.asarray([float(row[metric_name]) for row in seed_rows], dtype=np.float64)
        result[f"{metric_name}_mean"] = float(np.mean(values))
        result[f"{metric_name}_std"] = float(np.std(values, ddof=0))
    return result


def create_model(model_dir: str, mask_2d: np.ndarray, args: argparse.Namespace) -> Any:
    from baseline.models.SwinLSTM_B_model import SwinLSTMBaseline
    from baseline.models.convlstm_model import ConvLSTMBaseline
    from baseline.models.exprecast_model import ExPreCast
    from baseline.models.pfgnet_model import PFGNet
    from baseline.models.predrnn_model import PredRNN
    from baseline.models.predrnn_v2_model import PredRNNV2
    from baseline.models.seacast_model import SeaCast
    from baseline.models.timekan_model import TimeKAN
    from main_model.gpr_fishnet import GPRFishNet

    img_size = (args.height, args.width)
    in_chans = len(args.env_vars) + 1
    hidden_dim = args.hidden_dim
    pred_len = args.pred_len

    if model_dir == "checkpoints_gpr_fishnet_final":
        return GPRFishNet(in_chans=in_chans, hidden_dim=hidden_dim, img_size=img_size, num_layers=2)
    if model_dir == "checkpoints_predrnn_baseline":
        return PredRNN(in_chans=in_chans, hidden_dim=hidden_dim, img_size=img_size, num_layers=2, pred_len=pred_len)
    if model_dir == "checkpoints_predrnn_v2_baseline":
        return PredRNNV2(in_chans=in_chans, hidden_dim=hidden_dim, img_size=img_size, num_layers=2, pred_len=pred_len)
    if model_dir == "checkpoints_convlstm_baseline":
        return ConvLSTMBaseline(
            in_channels=in_chans,
            hidden_channels=[64, 64],
            kernel_size=(3, 3),
            pred_len=pred_len,
            img_size=img_size,
        )
    if model_dir == "checkpoints_pfgnet_final":
        return PFGNet(in_chans=in_chans, hidden_dim=hidden_dim, seq_len=args.seq_len, img_size=img_size, num_layers=4, pred_len=pred_len)
    if model_dir == "checkpoints_exprecast_final":
        return ExPreCast(in_chans=in_chans, hidden_dim=hidden_dim, img_size=img_size, num_layers=2, pred_len=pred_len)
    if model_dir == "checkpoints_timekan_final":
        return TimeKAN(in_chans=in_chans, hidden_dim=hidden_dim, seq_len=args.seq_len, img_size=img_size, pred_len=pred_len)
    if model_dir == "checkpoints_seacast_baseline":
        return SeaCast.from_mask(
            mask_2d,
            in_chans=in_chans,
            hidden_dim=hidden_dim,
            seq_len=args.seq_len,
            pred_len=pred_len,
            temporal_layers=1,
            graph_blocks=2,
            coarse_factor=4,
        )
    if model_dir == "checkpoints_swinlstm_baseline":
        return SwinLSTMBaseline(
            img_size=img_size,
            patch_size=2,
            in_chans=in_chans,
            embed_dim=64,
            depths=(2, 2),
            num_heads=(4, 4),
            window_size=4,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.05,
        )
    raise ValueError(f"Unsupported model: {model_dir}")


def candidate_run_dirs(model_root: Path, spec: ModelSpec, seed_mode: str, seeds: Sequence[int]) -> List[Tuple[str, Path]]:
    if seed_mode == "root":
        return [("root", model_root)]
    run_dirs = []
    for seed in seeds:
        seed_dir = model_root / f"seed_{seed}"
        checkpoint_path = seed_dir / spec.checkpoint_filename
        if checkpoint_path.exists():
            run_dirs.append((f"seed_{seed}", seed_dir))
    if run_dirs:
        return run_dirs
    return [("root", model_root)]


def cache_paths(output_dir: Path, model_dir: str, run_name: str) -> Tuple[Path, Path]:
    cache_dir = output_dir / "validation_predictions" / model_dir / run_name
    return cache_dir / "val_preds.npy", cache_dir / "val_targets.npy"


def run_validation_inference_for_checkpoint(
    model_dir: str,
    checkpoint_path: Path,
    loader: Any,
    mask_2d: np.ndarray,
    args: argparse.Namespace,
    device: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    from utils.output_clamp import apply_output_clamp

    model = create_model(model_dir, mask_2d, args)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    preds_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            preds = apply_output_clamp(model(inputs))
            preds_list.append(preds.detach().cpu().numpy().astype(np.float32))
            targets_list.append(targets.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds_list, axis=0), np.concatenate(targets_list, axis=0)


def ensure_validation_predictions(
    model_dirs: Sequence[str],
    args: argparse.Namespace,
    mask_2d: np.ndarray,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if args.reuse_val_preds:
        for model_dir in model_dirs:
            spec = MODEL_SPECS[model_dir]
            model_root = Path(args.model_outcomes_dir) / model_dir
            run_dirs = candidate_run_dirs(model_root, spec, args.seed_mode, args.seeds)
            for run_name, _ in run_dirs:
                preds_path, targets_path = cache_paths(Path(args.output_dir), model_dir, run_name)
                if not (preds_path.exists() and targets_path.exists()):
                    raise FileNotFoundError(f"Cached validation predictions not found: {preds_path}")
                records.append(
                    {
                        "model_dir": model_dir,
                        "model_label": spec.model_label,
                        "run_name": run_name,
                        "preds_path": preds_path,
                        "targets_path": targets_path,
                        "source": "cached_validation_predictions",
                    }
                )
        return records

    import torch
    from torch.utils.data import DataLoader
    from process.dataset import STFishNetUltimateDataset

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    val_dataset = STFishNetUltimateDataset(
        str(args.data_dir),
        args.env_vars,
        split="val",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    for model_dir in model_dirs:
        spec = MODEL_SPECS[model_dir]
        model_root = Path(args.model_outcomes_dir) / model_dir
        if not model_root.exists():
            print(f"[WARN] Skip missing model directory: {model_root}")
            continue

        run_dirs = candidate_run_dirs(model_root, spec, args.seed_mode, args.seeds)
        for run_name, run_dir in run_dirs:
            checkpoint_path = run_dir / spec.checkpoint_filename
            if not checkpoint_path.exists():
                print(f"[WARN] Skip missing checkpoint: {checkpoint_path}")
                continue

            preds_path, targets_path = cache_paths(Path(args.output_dir), model_dir, run_name)
            if preds_path.exists() and targets_path.exists() and not args.force_inference:
                source = "cached_validation_predictions"
            else:
                print(f"[INFO] Validation inference: {spec.model_label} / {run_name}")
                preds, targets = run_validation_inference_for_checkpoint(
                    model_dir=model_dir,
                    checkpoint_path=checkpoint_path,
                    loader=val_loader,
                    mask_2d=mask_2d,
                    args=args,
                    device=device,
                )
                preds_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(preds_path, preds)
                np.save(targets_path, targets)
                source = str(checkpoint_path)

            records.append(
                {
                    "model_dir": model_dir,
                    "model_label": spec.model_label,
                    "run_name": run_name,
                    "preds_path": preds_path,
                    "targets_path": targets_path,
                    "source": source,
                }
            )

    if not records:
        raise RuntimeError("No validation predictions were generated or loaded.")
    return records


def build_thresholds(args: argparse.Namespace) -> np.ndarray:
    if args.num_points < 2:
        raise ValueError("--num-points must be at least 2")
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.num_points, dtype=np.float64)
    extras = [args.focus_threshold]
    if args.include_thresholds:
        extras.extend(float(item.strip()) for item in args.include_thresholds.split(",") if item.strip())
    thresholds = np.unique(np.sort(np.concatenate([thresholds, np.asarray(extras, dtype=np.float64)])))
    return thresholds


def scan_thresholds(
    records: Sequence[Dict[str, Any]],
    thresholds: np.ndarray,
    mask_2d: np.ndarray,
    train_99p: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seed_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    records_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        records_by_model.setdefault(record["model_dir"], []).append(record)

    for model_dir in MODEL_ORDER:
        model_records = records_by_model.get(model_dir, [])
        if not model_records:
            continue
        spec = MODEL_SPECS[model_dir]

        values_by_run = []
        for record in model_records:
            pred_values, target_values = extract_valid_values(
                np.load(record["preds_path"]),
                np.load(record["targets_path"]),
                mask_2d,
            )
            values_by_run.append((record, pred_values, target_values))

        for threshold in thresholds:
            threshold_seed_rows = []
            for record, pred_values, target_values in values_by_run:
                metric_row = compute_binary_metrics(pred_values, target_values, float(threshold))
                seed_row: Dict[str, Any] = {
                    "model_dir": model_dir,
                    "model_label": spec.model_label,
                    "run_name": record["run_name"],
                    "threshold_norm": float(threshold),
                    "threshold_hours_per_day": float(threshold * train_99p),
                    "source": record["source"],
                }
                seed_row.update(metric_row)
                seed_rows.append(seed_row)
                threshold_seed_rows.append(metric_row)

            aggregate = aggregate_seed_metric_rows(threshold_seed_rows)
            aggregate_row: Dict[str, Any] = {
                "model_dir": model_dir,
                "model_label": spec.model_label,
                "threshold_norm": float(threshold),
                "threshold_hours_per_day": float(threshold * train_99p),
                "num_runs": int(len(threshold_seed_rows)),
            }
            aggregate_row.update(aggregate)
            aggregate_rows.append(aggregate_row)

    return seed_rows, aggregate_rows


def select_best_threshold(
    aggregate_rows: Sequence[Dict[str, Any]],
    selection_model: str,
    selection_metric: str,
    focus_threshold: float,
    target_hotspot_ratio: float,
) -> Dict[str, Any]:
    model_rows = [row for row in aggregate_rows if row["model_dir"] == selection_model]
    if not model_rows:
        raise RuntimeError(f"No aggregate rows found for selection model: {selection_model}")

    ranked = sorted(
        model_rows,
        key=lambda row: selection_score(row, selection_metric, target_hotspot_ratio),
        reverse=True,
    )
    best_row = ranked[0]
    focus_row = min(model_rows, key=lambda row: abs(float(row["threshold_norm"]) - focus_threshold))
    focus_rank = next(index for index, row in enumerate(ranked, start=1) if row is focus_row)
    best_score = selection_score(best_row, selection_metric, target_hotspot_ratio)
    focus_score = selection_score(focus_row, selection_metric, target_hotspot_ratio)

    return {
        "selection_model": selection_model,
        "selection_model_label": MODEL_SPECS[selection_model].model_label,
        "selection_metric": selection_metric.upper(),
        "target_hotspot_ratio": float(target_hotspot_ratio),
        "best_threshold_norm": float(best_row["threshold_norm"]),
        "best_threshold_hours_per_day": float(best_row["threshold_hours_per_day"]),
        "best_score": float(best_score),
        "focus_threshold_norm": float(focus_row["threshold_norm"]),
        "focus_threshold_hours_per_day": float(focus_row["threshold_hours_per_day"]),
        "focus_score": float(focus_score),
        "focus_rank": int(focus_rank),
        "num_thresholds": int(len(model_rows)),
        "score_gap_focus_minus_best": float(focus_score - best_score),
        "best_row": dict(best_row),
        "focus_row": dict(focus_row),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def describe_selection_rule(selection: Dict[str, Any]) -> str:
    metric = str(selection["selection_metric"]).upper()
    if metric == "TARGET_HOTSPOT_RATIO":
        return f"match validation target hotspot-area ratio = {selection['target_hotspot_ratio']:.2f}"
    if metric == "CSI_F1_MEAN":
        return "maximize validation mean(CSI, F1)"
    return f"maximize validation {metric}"


def plot_validation_selection(
    aggregate_rows: Sequence[Dict[str, Any]],
    selection: Dict[str, Any],
    output_prefix: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as exc:
        print(f"[WARN] Matplotlib unavailable, skip figure: {exc}")
        return False

    model_rows = [row for row in aggregate_rows if row["model_dir"] == selection["selection_model"]]
    if not model_rows:
        return False

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["axes.linewidth"] = 1.1
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), facecolor="white")
    axes = axes.ravel()
    panel_tags = ("(a)", "(b)", "(c)", "(d)")
    x = np.asarray([float(row["threshold_norm"]) for row in model_rows], dtype=np.float64)
    best_t = float(selection["best_threshold_norm"])
    focus_t = float(selection["focus_threshold_norm"])

    for idx, metric_name in enumerate(METRICS):
        ax = axes[idx]
        y = np.asarray([float(row[f"{metric_name}_mean"]) for row in model_rows], dtype=np.float64)
        y_std = np.asarray([float(row[f"{metric_name}_std"]) for row in model_rows], dtype=np.float64)
        ax.plot(x, y, color="#B91C1C", linewidth=2.5)
        if np.any(y_std > 0):
            ax.fill_between(x, y - y_std, y + y_std, color="#B91C1C", alpha=0.14, linewidth=0)
        ax.axvline(best_t, color="#111827", linestyle="--", linewidth=1.6, label=f"Selected = {best_t:.4f}")
        if abs(focus_t - best_t) > 1e-10:
            ax.axvline(focus_t, color="#374151", linestyle=":", linewidth=1.5, label=f"Focus = {focus_t:.4f}")
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.set_ylim(0.0, 1.02)
        ax.set_title(metric_name, fontsize=14.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.6)
        ax.text(-0.12, 1.05, panel_tags[idx], transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
        if idx % 2 == 0:
            ax.set_ylabel("Validation Score", fontsize=12, fontweight="bold")
        if idx >= 2:
            ax.set_xlabel("Normalized Threshold", fontsize=12, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.03), fontsize=11)
    fig.suptitle(
        f"Validation-Based Hotspot Threshold Selection ({selection['selection_model_label']})",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.085,
        (
            f"Selection rule: {describe_selection_rule(selection)}; "
            f"selected threshold = {best_t:.4f} "
            f"({selection['best_threshold_hours_per_day']:.2f} hours/day)."
        ),
        ha="center",
        fontsize=10.5,
        color="#374151",
    )
    fig.tight_layout(rect=[0.04, 0.105, 0.98, 0.94])

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def resolve_output_prefixes(output_dir: Path, explicit_output_prefix: str, selection_metric: str) -> Tuple[Path, Path]:
    if explicit_output_prefix:
        prefix = Path(explicit_output_prefix)
        if not prefix.is_absolute():
            prefix = (PROJECT_ROOT / prefix).resolve()
        return prefix, prefix
    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    archive = output_dir / f"validation_threshold_selection_{selection_metric.lower()}_{timestamp}"
    latest = output_dir / f"validation_threshold_selection_{selection_metric.lower()}_latest"
    return archive, latest


def copy_latest_outputs(archive_prefix: Path, latest_prefix: Path) -> None:
    if archive_prefix == latest_prefix:
        return
    suffixes = [
        "_aggregate_curve.csv",
        "_seed_curve.csv",
        "_selection.json",
        ".png",
        ".pdf",
    ]
    for suffix in suffixes:
        archive_path = archive_prefix.with_suffix(suffix) if suffix.startswith(".") else archive_prefix.with_name(f"{archive_prefix.name}{suffix}")
        latest_path = latest_prefix.with_suffix(suffix) if suffix.startswith(".") else latest_prefix.with_name(f"{latest_prefix.name}{suffix}")
        if archive_path.exists():
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive_path, latest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select an operating hotspot threshold on the validation set.")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Preprocessed ST-FishNet feature directory.")
    parser.add_argument("--mask-path", type=str, default=str(DEFAULT_MASK_PATH), help="Shared ocean mask.")
    parser.add_argument("--model-outcomes-dir", type=str, default=str(DEFAULT_MODEL_OUTCOMES_DIR), help="Checkpoint root directory.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--output-prefix", type=str, default="", help="Optional explicit output prefix.")
    parser.add_argument("--models", type=str, default="gpr", help="Comma-separated model aliases/dirs, or 'all'.")
    parser.add_argument("--selection-model", type=str, default="checkpoints_gpr_fishnet_final", help="Model used to select the operating threshold.")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="F1",
        choices=["CSI", "F1", "Precision", "Recall", "CSI_F1_MEAN", "TARGET_HOTSPOT_RATIO"],
        help="Validation rule used to choose the threshold.",
    )
    parser.add_argument(
        "--target-hotspot-ratio",
        type=float,
        default=0.20,
        help="Target validation hotspot-area ratio used when --selection-metric TARGET_HOTSPOT_RATIO.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.10, help="Minimum normalized threshold.")
    parser.add_argument("--threshold-max", type=float, default=0.60, help="Maximum normalized threshold.")
    parser.add_argument("--num-points", type=int, default=101, help="Number of thresholds in the linear scan.")
    parser.add_argument("--focus-threshold", type=float, default=0.3175, help="Threshold to report/rank alongside the selected optimum.")
    parser.add_argument("--include-thresholds", type=str, default="", help="Extra comma-separated thresholds to include exactly.")
    parser.add_argument("--seq-len", type=int, default=12, help="Input sequence length.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction length.")
    parser.add_argument("--batch-size", type=int, default=2, help="Validation inference batch size.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Shared hidden dimension.")
    parser.add_argument("--height", type=int, default=64, help="Grid height.")
    parser.add_argument("--width", type=int, default=96, help="Grid width.")
    parser.add_argument("--env-vars", type=str, default=",".join(DEFAULT_ENV_VARS), help="Comma-separated environmental variables.")
    parser.add_argument("--seed-mode", type=str, default="seeds", choices=["seeds", "root"], help="Use seed checkpoints or the root selected checkpoint.")
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS), help="Comma-separated seed ids.")
    parser.add_argument("--device", type=str, default="", help="Torch device override, e.g. cpu or cuda.")
    parser.add_argument("--force-inference", action="store_true", help="Regenerate cached validation predictions.")
    parser.add_argument("--reuse-val-preds", action="store_true", help="Reuse cached validation predictions and skip Torch inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = Path(args.data_dir)
    args.mask_path = Path(args.mask_path)
    args.model_outcomes_dir = Path(args.model_outcomes_dir)
    args.output_dir = Path(args.output_dir)
    if not args.data_dir.is_absolute():
        args.data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    if not args.mask_path.is_absolute():
        args.mask_path = (PROJECT_ROOT / args.mask_path).resolve()
    if not args.model_outcomes_dir.is_absolute():
        args.model_outcomes_dir = (PROJECT_ROOT / args.model_outcomes_dir).resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (PROJECT_ROOT / args.output_dir).resolve()

    args.env_vars = [item.strip() for item in args.env_vars.split(",") if item.strip()]
    args.seeds = parse_seed_list(args.seeds)

    model_dirs = parse_model_list(args.models)
    selection_model = parse_model_list(args.selection_model)[0]
    if selection_model not in model_dirs:
        model_dirs = [selection_model, *model_dirs]
    if not 0.0 <= args.threshold_min < args.threshold_max <= 1.0:
        raise ValueError("Threshold range must satisfy 0 <= min < max <= 1.")

    mask_2d = load_mask(args.mask_path, (args.height, args.width))
    train_99p = load_train_99p(args.data_dir)
    thresholds = build_thresholds(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = ensure_validation_predictions(model_dirs, args, mask_2d)
    seed_rows, aggregate_rows = scan_thresholds(records, thresholds, mask_2d, train_99p)
    selection = select_best_threshold(
        aggregate_rows,
        selection_model,
        args.selection_metric,
        args.focus_threshold,
        args.target_hotspot_ratio,
    )

    archive_prefix, latest_prefix = resolve_output_prefixes(args.output_dir, args.output_prefix, selection["selection_metric"])
    aggregate_csv = archive_prefix.with_name(f"{archive_prefix.name}_aggregate_curve.csv")
    seed_csv = archive_prefix.with_name(f"{archive_prefix.name}_seed_curve.csv")
    selection_json = archive_prefix.with_name(f"{archive_prefix.name}_selection.json")

    write_csv(aggregate_csv, aggregate_rows)
    write_csv(seed_csv, seed_rows)
    write_json(
        selection_json,
        {
            "selection": selection,
            "config": {
                "data_dir": str(args.data_dir),
                "mask_path": str(args.mask_path),
                "model_outcomes_dir": str(args.model_outcomes_dir),
                "models": model_dirs,
                "selection_model": selection_model,
                "selection_metric": args.selection_metric,
                "threshold_min": args.threshold_min,
                "threshold_max": args.threshold_max,
                "num_thresholds": int(len(thresholds)),
                "focus_threshold": args.focus_threshold,
                "target_hotspot_ratio": args.target_hotspot_ratio,
                "seed_mode": args.seed_mode,
                "seeds": args.seeds,
                "train_99p": train_99p,
            },
        },
    )
    figure_written = plot_validation_selection(aggregate_rows, selection, archive_prefix)
    copy_latest_outputs(archive_prefix, latest_prefix)

    print(f"Selection model: {selection['selection_model_label']}")
    print(f"Selection metric: {selection['selection_metric']}")
    print(
        "Best validation threshold: "
        f"{selection['best_threshold_norm']:.4f} "
        f"({selection['best_threshold_hours_per_day']:.4f} hours/day), "
        f"score={selection['best_score']:.6f}"
    )
    print(
        "Focus threshold: "
        f"{selection['focus_threshold_norm']:.4f}, "
        f"rank={selection['focus_rank']}/{selection['num_thresholds']}, "
        f"score={selection['focus_score']:.6f}, "
        f"gap={selection['score_gap_focus_minus_best']:.6f}"
    )
    print(f"Saved aggregate curve CSV to: {aggregate_csv}")
    print(f"Saved seed curve CSV to: {seed_csv}")
    print(f"Saved selection JSON to: {selection_json}")
    if figure_written:
        print(f"Saved selection figure to: {archive_prefix.with_suffix('.png')}")
        print(f"Saved selection figure to: {archive_prefix.with_suffix('.pdf')}")
    if latest_prefix != archive_prefix:
        print(f"Updated latest outputs under prefix: {latest_prefix}")


if __name__ == "__main__":
    main()
