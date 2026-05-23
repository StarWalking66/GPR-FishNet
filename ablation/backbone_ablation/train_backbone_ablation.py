from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import asdict
from typing import Dict, List

# Windows OpenMP runtime fallback:
# some Python stacks (e.g., torch + MKL-linked deps) can load duplicated libiomp5md.dll.
# This env var is an unsafe workaround but unblocks execution in mixed environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ablation.backbone_ablation.models_backbone_ablation import (  # noqa: E402
    BACKBONE_VARIANT_DESCRIPTIONS,
    BACKBONE_VARIANTS,
    BackboneAblationNet,
)
from ablation.common_experiment import (  # noqa: E402
    ExperimentConfig,
    collect_summary_row,
    collect_summary_row_for_threshold,
    load_json,
    run_multi_seed_experiment,
    save_json,
    save_rows,
)
from ablation.statistical_tests import run_significance_analysis  # noqa: E402
from utils.multi_seed_experiment import (  # noqa: E402
    finalize_multi_seed_experiment,
    resolve_seed_save_dir,
    save_additional_threshold_evaluations,
    threshold_tag,
)


DEFAULT_ENV_VARS = ["thetao", "chl", "uo", "vo", "so", "zos", "o2"]
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ST_FishNet_Features")
DEFAULT_MASK_PATH = os.path.join(DEFAULT_DATA_DIR, "all_vars_train_mask_intersection.npy")
DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "model_outcomes", "ablation", "backbone_ablation")
DEFAULT_MAIN_MODEL_DIR = os.path.join(PROJECT_ROOT, "model_outcomes", "checkpoints_gpr_fishnet_final")
DEFAULT_HOTSPOT_THRESHOLD = 0.2755
DEFAULT_EXTRA_HOTSPOT_THRESHOLDS = "0.3175"


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in parse_csv_list(raw):
        try:
            parsed = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid float value in threshold list: {item}") from exc
        values.append(parsed)
    return values


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backbone ablation for GPR-FishNet.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory of preprocessed ST FishNet features.")
    parser.add_argument("--mask-path", type=str, default=DEFAULT_MASK_PATH, help="Path of shared valid-ocean mask npy.")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR, help="Root output directory for backbone ablation.")
    parser.add_argument(
        "--full-source-dir",
        type=str,
        default=DEFAULT_MAIN_MODEL_DIR,
        help="Existing main-model result directory reused as the full backbone variant.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(BACKBONE_VARIANTS),
        help=f"Comma-separated variants. Available: {', '.join(BACKBONE_VARIANTS)}",
    )
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--train-loss", type=str, default="mse", choices=["mse", "mae"], help="Training loss type.")
    parser.add_argument("--hotspot-threshold", type=float, default=DEFAULT_HOTSPOT_THRESHOLD, help="Threshold for CSI/F1.")
    parser.add_argument(
        "--extra-hotspot-thresholds",
        type=str,
        default=DEFAULT_EXTRA_HOTSPOT_THRESHOLDS,
        help="Optional comma-separated extra thresholds for recomputed CSI/F1/Precision/Recall exports, e.g. 0.3175.",
    )
    parser.add_argument("--seq-len", type=int, default=12, help="Input sequence length.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction horizon per sample.")
    parser.add_argument("--rollout-horizon", type=int, default=12, help="2024 rollout horizon.")
    parser.add_argument("--rollout-start-index", type=int, default=144, help="Start index of 2024 rollout.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden channels in ST-LSTM stack.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of ST-LSTM layers.")
    return parser


def _copy_file_if_exists(src: str, dst: str) -> None:
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _source_seed_dir(source_dir: str, seed: int) -> str:
    candidate = os.path.join(source_dir, f"seed_{seed}")
    if os.path.isdir(candidate):
        return candidate
    return source_dir


def _load_valid_mask(mask_path: str, shape: tuple[int, ...]) -> np.ndarray:
    if os.path.exists(mask_path):
        mask = np.load(mask_path).astype(np.float32)
    else:
        mask = np.ones(shape[-2:], dtype=np.float32)
    while mask.ndim < len(shape):
        mask = np.expand_dims(mask, axis=0)
    return np.broadcast_to(mask > 0.5, shape)


def _compute_threshold_metrics(preds: np.ndarray, targets: np.ndarray, mask_path: str, hotspot_threshold: float) -> Dict[str, float]:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    valid_mask = _load_valid_mask(mask_path, preds.shape)

    p = preds[valid_mask]
    t = targets[valid_mask]
    if p.size == 0:
        return {"CSI": 0.0, "F1": 0.0}

    pred_bin = (p >= hotspot_threshold).astype(np.uint8)
    target_bin = (t >= hotspot_threshold).astype(np.uint8)

    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))

    csi = float(tp / (tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-8))
    return {"CSI": csi, "F1": f1}


def import_main_model_as_full_variant(
    source_dir: str,
    save_dir: str,
    config: ExperimentConfig,
    metadata: Dict[str, object],
    checkpoint_filename: str,
    extra_hotspot_thresholds: List[float],
) -> Dict[str, object]:
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Full-model source directory does not exist: {source_dir}")

    os.makedirs(save_dir, exist_ok=True)
    config_payload = asdict(config)
    config_payload["env_vars"] = list(DEFAULT_ENV_VARS)
    config_payload["device"] = "imported_existing_results"
    config_payload["metadata"] = {**metadata, "import_source_dir": source_dir}
    save_json(config_payload, os.path.join(save_dir, "experiment_config.json"))

    run_rows: List[Dict] = []
    for seed in config.seeds:
        src_seed_dir = _source_seed_dir(source_dir, seed)
        if not os.path.isdir(src_seed_dir):
            raise FileNotFoundError(f"Missing source seed directory for seed={seed}: {src_seed_dir}")

        dst_seed_dir = resolve_seed_save_dir(save_dir, seed, config.seeds)
        os.makedirs(dst_seed_dir, exist_ok=True)

        src_run_summary_path = os.path.join(src_seed_dir, "run_summary.json")
        src_test_metrics_path = os.path.join(src_seed_dir, "test_one_step_metrics.json")
        src_rollout_metrics_path = os.path.join(src_seed_dir, "rollout_2024_metrics.json")
        if not os.path.exists(src_run_summary_path):
            raise FileNotFoundError(f"Missing run summary for imported full model: {src_run_summary_path}")
        if not os.path.exists(src_test_metrics_path) or not os.path.exists(src_rollout_metrics_path):
            raise FileNotFoundError(f"Missing metrics json for imported full model seed={seed}: {src_seed_dir}")

        test_preds_path = os.path.join(src_seed_dir, "test_one_step_preds.npy")
        test_targets_path = os.path.join(src_seed_dir, "test_one_step_targets.npy")
        rollout_preds_path = os.path.join(src_seed_dir, "rollout_2024_preds.npy")
        rollout_targets_path = os.path.join(src_seed_dir, "rollout_2024_targets.npy")
        if not all(os.path.exists(path) for path in [test_preds_path, test_targets_path, rollout_preds_path, rollout_targets_path]):
            raise FileNotFoundError(f"Missing prediction arrays for imported full model seed={seed}: {src_seed_dir}")

        test_metrics = dict(load_json(src_test_metrics_path))
        rollout_metrics = dict(load_json(src_rollout_metrics_path))
        run_summary = dict(load_json(src_run_summary_path))

        test_threshold_metrics = _compute_threshold_metrics(np.load(test_preds_path), np.load(test_targets_path), config.mask_path, config.hotspot_threshold)
        rollout_threshold_metrics = _compute_threshold_metrics(np.load(rollout_preds_path), np.load(rollout_targets_path), config.mask_path, config.hotspot_threshold)

        test_metrics.update(test_threshold_metrics)
        rollout_metrics.update(rollout_threshold_metrics)
        run_summary["test_one_step_CSI"] = float(test_threshold_metrics["CSI"])
        run_summary["test_one_step_F1"] = float(test_threshold_metrics["F1"])
        run_summary["rollout_2024_CSI"] = float(rollout_threshold_metrics["CSI"])
        run_summary["rollout_2024_F1"] = float(rollout_threshold_metrics["F1"])

        save_json(test_metrics, os.path.join(dst_seed_dir, "test_one_step_metrics.json"))
        save_json(rollout_metrics, os.path.join(dst_seed_dir, "rollout_2024_metrics.json"))
        save_json(run_summary, os.path.join(dst_seed_dir, "run_summary.json"))

        _copy_file_if_exists(os.path.join(src_seed_dir, "training_history.json"), os.path.join(dst_seed_dir, "training_history.json"))
        _copy_file_if_exists(test_preds_path, os.path.join(dst_seed_dir, "test_one_step_preds.npy"))
        _copy_file_if_exists(test_targets_path, os.path.join(dst_seed_dir, "test_one_step_targets.npy"))
        _copy_file_if_exists(os.path.join(src_seed_dir, "test_one_step_inputs.npy"), os.path.join(dst_seed_dir, "test_one_step_inputs.npy"))
        _copy_file_if_exists(rollout_preds_path, os.path.join(dst_seed_dir, "rollout_2024_preds.npy"))
        _copy_file_if_exists(rollout_targets_path, os.path.join(dst_seed_dir, "rollout_2024_targets.npy"))
        _copy_file_if_exists(
            os.path.join(src_seed_dir, "best_gpr_fishnet.pth"),
            os.path.join(dst_seed_dir, checkpoint_filename),
        )

        run_rows.append(run_summary)

    finalize_multi_seed_experiment(
        base_save_dir=save_dir,
        seeds=config.seeds,
        run_rows=run_rows,
        mask_path=config.mask_path,
        hotspot_threshold=config.hotspot_threshold,
        best_checkpoint_filename=checkpoint_filename,
    )
    if extra_hotspot_thresholds:
        save_additional_threshold_evaluations(
            base_save_dir=save_dir,
            seeds=config.seeds,
            mask_path=config.mask_path,
            hotspot_thresholds=extra_hotspot_thresholds,
        )
    return collect_summary_row("full", save_dir, extra_fields=metadata)


def main() -> None:
    args = build_parser().parse_args()

    requested_variants = unique_preserve_order(parse_csv_list(args.variants))
    invalid_variants = [name for name in requested_variants if name not in BACKBONE_VARIANTS]
    if invalid_variants:
        raise ValueError(f"Unknown variants: {invalid_variants}. Available: {BACKBONE_VARIANTS}")
    if not requested_variants:
        raise ValueError("No backbone ablation variant selected.")

    seeds = [int(s) for s in parse_csv_list(args.seeds)]
    if not seeds:
        raise ValueError("No seeds selected.")

    extra_hotspot_thresholds = [
        threshold
        for threshold in parse_csv_float_list(args.extra_hotspot_thresholds)
        if abs(threshold - float(args.hotspot_threshold)) > 1e-12
    ]

    config = ExperimentConfig(
        data_dir=args.data_dir,
        mask_path=args.mask_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        rollout_2024_horizon=args.rollout_horizon,
        rollout_start_index=args.rollout_start_index,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        train_loss=args.train_loss,
        hotspot_threshold=args.hotspot_threshold,
        seeds=seeds,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    summary_rows = []

    print("Backbone ablation execution plan:")
    for idx, variant in enumerate(requested_variants, start=1):
        print(f"  [{idx}/{len(requested_variants)}] {variant}: {BACKBONE_VARIANT_DESCRIPTIONS[variant]}")

    summary_metadata_by_variant: Dict[str, Dict[str, object]] = {}
    for variant in requested_variants:
        variant_save_dir = os.path.join(args.save_dir, variant)
        metadata = {
            "ablation_type": "backbone",
            "variant": variant,
            "variant_description": BACKBONE_VARIANT_DESCRIPTIONS[variant],
            "env_vars": DEFAULT_ENV_VARS,
            "hotspot_threshold": float(args.hotspot_threshold),
            "extra_hotspot_thresholds": list(extra_hotspot_thresholds),
        }

        if variant == "full":
            metadata["result_source"] = "imported_existing_main_model"
            summary_row = import_main_model_as_full_variant(
                source_dir=args.full_source_dir,
                save_dir=variant_save_dir,
                config=config,
                metadata=metadata,
                checkpoint_filename="best_backbone_full.pth",
                extra_hotspot_thresholds=extra_hotspot_thresholds,
            )
        else:
            metadata["result_source"] = "trained_with_backbone_ablation"

            def model_builder(in_chans: int, exp_cfg: ExperimentConfig, variant_name: str = variant) -> BackboneAblationNet:
                return BackboneAblationNet(
                    variant=variant_name,
                    in_chans=in_chans,
                    hidden_dim=exp_cfg.hidden_dim,
                    img_size=exp_cfg.img_size,
                    num_layers=exp_cfg.num_layers,
                )

            summary_row = run_multi_seed_experiment(
                experiment_name=variant,
                save_dir=variant_save_dir,
                env_vars=DEFAULT_ENV_VARS,
                model_builder=model_builder,
                config=config,
                best_checkpoint_filename=f"best_backbone_{variant}.pth",
                metadata=metadata,
                extra_hotspot_thresholds=extra_hotspot_thresholds,
            )
        summary_rows.append(summary_row)
        summary_metadata_by_variant[variant] = metadata

    summary_csv_path = os.path.join(args.save_dir, "backbone_ablation_summary.csv")
    summary_json_path = os.path.join(args.save_dir, "backbone_ablation_summary.json")
    save_rows(summary_rows, summary_csv_path, summary_json_path)
    print(f"Backbone ablation summary saved to:\n  {summary_csv_path}\n  {summary_json_path}")

    significance_result = run_significance_analysis(
        group_save_dir=args.save_dir,
        reference_experiment="full",
        candidate_experiments=requested_variants,
        output_prefix="backbone_significance_vs_full",
        alpha=0.05,
        n_bootstrap=10000,
        n_permutation=20000,
        random_seed=2026,
    )
    if significance_result is not None:
        print(
            "Backbone significance analysis saved to:\n"
            f"  {significance_result['csv_path']}\n"
            f"  {significance_result['json_path']}"
        )
    else:
        print("Backbone significance analysis skipped because reference experiment 'full' was not available.")

    for threshold in extra_hotspot_thresholds:
        tag = threshold_tag(threshold)
        threshold_summary_rows = []
        for variant in requested_variants:
            threshold_metadata = dict(summary_metadata_by_variant[variant])
            threshold_metadata["hotspot_threshold"] = float(threshold)
            threshold_metadata["threshold_metric_source"] = "recomputed_from_saved_predictions"
            threshold_summary_rows.append(
                collect_summary_row_for_threshold(variant, os.path.join(args.save_dir, variant), threshold, threshold_metadata)
            )

        threshold_summary_csv_path = os.path.join(args.save_dir, f"backbone_ablation_summary_threshold_{tag}.csv")
        threshold_summary_json_path = os.path.join(args.save_dir, f"backbone_ablation_summary_threshold_{tag}.json")
        save_rows(threshold_summary_rows, threshold_summary_csv_path, threshold_summary_json_path)
        print(
            f"Backbone threshold-specific summary saved to:\n"
            f"  {threshold_summary_csv_path}\n"
            f"  {threshold_summary_json_path}"
        )

        threshold_significance_result = run_significance_analysis(
            group_save_dir=args.save_dir,
            reference_experiment="full",
            candidate_experiments=requested_variants,
            output_prefix=f"backbone_significance_vs_full_threshold_{tag}",
            summary_runs_filename=f"summary_runs_threshold_{tag}.csv",
            alpha=0.05,
            n_bootstrap=10000,
            n_permutation=20000,
            random_seed=2026,
        )
        if threshold_significance_result is not None:
            print(
                "Backbone threshold-specific significance analysis saved to:\n"
                f"  {threshold_significance_result['csv_path']}\n"
                f"  {threshold_significance_result['json_path']}"
            )
        else:
            print(
                f"Backbone threshold-specific significance analysis skipped for threshold={threshold:.4f} "
                "because threshold-specific summaries were not available."
            )


if __name__ == "__main__":
    main()
