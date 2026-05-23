from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Windows OpenMP runtime fallback:
# some Python stacks (e.g., torch + MKL-linked deps) can load duplicated libiomp5md.dll.
# This env var is an unsafe workaround but unblocks execution in mixed environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ablation.common_experiment import (
    ExperimentConfig,
    collect_summary_row_for_threshold,
    run_multi_seed_experiment,
    save_rows,
)
from ablation.statistical_tests import run_significance_analysis
from main_model.gpr_fishnet import GPRFishNet
from utils.multi_seed_experiment import threshold_tag


DEFAULT_ENV_VARS = ["thetao", "chl", "uo", "vo", "so", "zos", "o2"]
MAIN_MODEL_SIGNATURE = "GPR-FishNet: STLSTM+ARP+MSSP+CAR(ContextAwareRouter)+ReLU"
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ST_FishNet_Features")
DEFAULT_MASK_PATH = os.path.join(DEFAULT_DATA_DIR, "all_vars_train_mask_intersection.npy")
DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "model_outcomes", "ablation", "env_factor_ablation")
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


def unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def build_zero_fill_transforms(
    dropped_env_indices: Sequence[int],
) -> Tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], Optional[Callable[[np.ndarray], np.ndarray]]]:
    dropped = sorted(set(int(i) for i in dropped_env_indices))
    if not dropped:
        return None, None

    def batch_transform(inputs: torch.Tensor) -> torch.Tensor:
        transformed = inputs.clone()
        transformed[:, :, dropped, :, :] = 0.0
        return transformed

    def rollout_transform(current_input: np.ndarray) -> np.ndarray:
        transformed = current_input.copy()
        transformed[:, dropped, :, :] = 0.0
        return transformed

    return batch_transform, rollout_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-by-one (leave-one-out) environmental-factor ablation for GPR-FishNet."
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory of preprocessed ST FishNet features.")
    parser.add_argument("--mask-path", type=str, default=DEFAULT_MASK_PATH, help="Path of shared valid-ocean mask npy.")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR, help="Root output directory for env-factor ablation.")
    parser.add_argument(
        "--env-vars",
        type=str,
        default=",".join(DEFAULT_ENV_VARS),
        help="Comma-separated base environmental factors before one-by-one dropping.",
    )
    parser.add_argument(
        "--drop-factors",
        type=str,
        default="",
        help="Comma-separated factors to ablate. Default is all factors in --env-vars.",
    )
    parser.add_argument("--skip-full", action="store_true", help="Do not run the full-factor reference setting.")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--train-loss", type=str, default="mse", choices=["mse", "mae"], help="Training loss type.")
    parser.add_argument("--hotspot-threshold", type=float, default=0.2755, help="Threshold for CSI/F1.")
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


def build_experiment_plan(base_env_vars: List[str], drop_factors: List[str], skip_full: bool) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []

    if not skip_full:
        plan.append(
            {
                "name": "full_factors",
                "dropped_factor": "none",
                "kept_env_vars": list(base_env_vars),
                "dropped_env_indices": [],
            }
        )

    for factor in drop_factors:
        kept = [v for v in base_env_vars if v != factor]
        dropped_index = base_env_vars.index(factor)
        plan.append(
            {
                "name": f"drop_{factor}",
                "dropped_factor": factor,
                "kept_env_vars": kept,
                "dropped_env_indices": [dropped_index],
            }
        )
    return plan


def main() -> None:
    args = build_parser().parse_args()

    base_env_vars = parse_csv_list(args.env_vars)
    if len(base_env_vars) < 2:
        raise ValueError("At least two environmental factors are required for leave-one-out ablation.")

    if len(set(base_env_vars)) != len(base_env_vars):
        raise ValueError(f"Duplicate factor found in --env-vars: {base_env_vars}")

    drop_factors = parse_csv_list(args.drop_factors) if args.drop_factors else list(base_env_vars)
    drop_factors = unique_preserve_order(drop_factors)
    invalid_drop = [f for f in drop_factors if f not in base_env_vars]
    if invalid_drop:
        raise ValueError(f"Unknown factors in --drop-factors: {invalid_drop}. Valid: {base_env_vars}")

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

    plan = build_experiment_plan(base_env_vars, drop_factors, args.skip_full)
    if not plan:
        raise ValueError("No environment-factor ablation experiment selected.")

    os.makedirs(args.save_dir, exist_ok=True)

    print("Environmental-factor one-by-one ablation plan:")
    print(f"Model signature: {MAIN_MODEL_SIGNATURE}")
    for idx, item in enumerate(plan, start=1):
        print(
            f"  [{idx}/{len(plan)}] {item['name']} | dropped={item['dropped_factor']} "
            f"| kept={item['kept_env_vars']}"
        )

    summary_rows = []
    summary_metadata_by_experiment: Dict[str, Dict[str, object]] = {}
    for item in plan:
        experiment_name = str(item["name"])
        kept_env_vars = list(item["kept_env_vars"])
        dropped_factor = str(item["dropped_factor"])
        dropped_env_indices = [int(i) for i in item["dropped_env_indices"]]
        experiment_save_dir = os.path.join(args.save_dir, experiment_name)
        input_batch_transform, rollout_input_transform = build_zero_fill_transforms(dropped_env_indices)

        metadata = {
            "ablation_type": "env_factor_leave_one_out",
            "experiment": experiment_name,
            "dropped_factor": dropped_factor,
            "kept_env_vars": kept_env_vars,
            "num_kept_factors": len(kept_env_vars),
            "dropped_env_indices": dropped_env_indices,
            "input_ablation_strategy": "zero_fill_dropped_env_channels_with_fixed_in_chans",
            "base_env_vars": base_env_vars,
            "model_signature": MAIN_MODEL_SIGNATURE,
            "hotspot_threshold": float(args.hotspot_threshold),
            "extra_hotspot_thresholds": list(extra_hotspot_thresholds),
        }

        def model_builder(in_chans: int, exp_cfg: ExperimentConfig) -> GPRFishNet:
            return GPRFishNet(
                in_chans=in_chans,
                hidden_dim=exp_cfg.hidden_dim,
                img_size=exp_cfg.img_size,
                num_layers=exp_cfg.num_layers,
            )

        summary_row = run_multi_seed_experiment(
            experiment_name=experiment_name,
            save_dir=experiment_save_dir,
            env_vars=base_env_vars,
            model_builder=model_builder,
            config=config,
            best_checkpoint_filename=f"best_env_factor_{experiment_name}.pth",
            metadata=metadata,
            input_batch_transform=input_batch_transform,
            rollout_input_transform=rollout_input_transform,
            extra_hotspot_thresholds=extra_hotspot_thresholds,
        )
        summary_rows.append(summary_row)
        summary_metadata_by_experiment[experiment_name] = metadata

    summary_csv_path = os.path.join(args.save_dir, "env_factor_leave_one_out_summary.csv")
    summary_json_path = os.path.join(args.save_dir, "env_factor_leave_one_out_summary.json")
    save_rows(summary_rows, summary_csv_path, summary_json_path)
    print(f"Env-factor ablation summary saved to:\n  {summary_csv_path}\n  {summary_json_path}")

    significance_result = run_significance_analysis(
        group_save_dir=args.save_dir,
        reference_experiment="full_factors",
        candidate_experiments=[str(item["name"]) for item in plan],
        output_prefix="env_factor_significance_vs_full_factors",
        alpha=0.05,
        n_bootstrap=10000,
        n_permutation=20000,
        random_seed=2026,
    )
    if significance_result is not None:
        print(
            "Env-factor significance analysis saved to:\n"
            f"  {significance_result['csv_path']}\n"
            f"  {significance_result['json_path']}"
        )
    else:
        print("Env-factor significance analysis skipped because reference experiment 'full_factors' was not available.")

    for threshold in extra_hotspot_thresholds:
        tag = threshold_tag(threshold)
        threshold_summary_rows = []
        for item in plan:
            experiment_name = str(item["name"])
            threshold_metadata = dict(summary_metadata_by_experiment[experiment_name])
            threshold_metadata["hotspot_threshold"] = float(threshold)
            threshold_metadata["threshold_metric_source"] = "recomputed_from_saved_predictions"
            threshold_summary_rows.append(
                collect_summary_row_for_threshold(
                    experiment_name,
                    os.path.join(args.save_dir, experiment_name),
                    threshold,
                    threshold_metadata,
                )
            )

        threshold_summary_csv_path = os.path.join(args.save_dir, f"env_factor_leave_one_out_summary_threshold_{tag}.csv")
        threshold_summary_json_path = os.path.join(args.save_dir, f"env_factor_leave_one_out_summary_threshold_{tag}.json")
        save_rows(threshold_summary_rows, threshold_summary_csv_path, threshold_summary_json_path)
        print(
            f"Env-factor threshold-specific summary saved to:\n"
            f"  {threshold_summary_csv_path}\n"
            f"  {threshold_summary_json_path}"
        )

        threshold_significance_result = run_significance_analysis(
            group_save_dir=args.save_dir,
            reference_experiment="full_factors",
            candidate_experiments=[str(plan_item["name"]) for plan_item in plan],
            output_prefix=f"env_factor_significance_vs_full_factors_threshold_{tag}",
            summary_runs_filename=f"summary_runs_threshold_{tag}.csv",
            alpha=0.05,
            n_bootstrap=10000,
            n_permutation=20000,
            random_seed=2026,
        )
        if threshold_significance_result is not None:
            print(
                "Env-factor threshold-specific significance analysis saved to:\n"
                f"  {threshold_significance_result['csv_path']}\n"
                f"  {threshold_significance_result['json_path']}"
            )
        else:
            print(
                f"Env-factor threshold-specific significance analysis skipped for threshold={threshold:.4f} "
                "because threshold-specific summaries were not available."
            )


if __name__ == "__main__":
    main()
