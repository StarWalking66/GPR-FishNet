from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.models.pfgnet_model import PFGNet
from baseline.models.predrnn_model import PredRNN
from baseline.train.fair_benchmark_config import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HOTSPOT_THRESHOLD,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_GRAD_NORM,
    MIN_LR,
    PRED_LEN,
    SEQ_LEN,
    TRAIN_LOSS,
    WEIGHT_DECAY,
)
from timewindowback.train_gpr_fishnet_backtest import (
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_VARS,
    DEFAULT_MASK_PATH,
    TemporalWindowDataset,
    WINDOW_CONFIGS,
    build_seed_run_row,
    evaluate,
    format_range,
    finalize_window_experiment,
    load_full_series,
    load_time_axis,
    masked_mae_loss,
    masked_mse_loss,
    parse_float_list,
    parse_int_list,
    rollout_predict_window,
    save_additional_threshold_evaluations,
    save_example_predictions,
    save_json,
    save_rollout_results,
    save_root_summaries,
    set_seed,
    test_and_save_one_step,
)
from utils.multi_seed_experiment import DEFAULT_SEEDS, resolve_seed_save_dir


DEFAULT_PREDRNN_SAVE_DIR = PROJECT_ROOT / "model_outcomes" / "checkpoints_predrnn_backtest"
DEFAULT_PFGNET_SAVE_DIR = PROJECT_ROOT / "model_outcomes" / "checkpoints_pfgnet_backtest"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    save_dir: Path
    best_checkpoint_filename: str
    model_signature: str


def parse_name_list(raw: str) -> List[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def build_model_specs(args: argparse.Namespace) -> Dict[str, ModelSpec]:
    return {
        "predrnn": ModelSpec(
            key="predrnn",
            display_name="PredRNN",
            save_dir=args.predrnn_save_dir,
            best_checkpoint_filename="best_predrnn.pth",
            model_signature="PredRNN(STLSTM backbone)",
        ),
        "pfgnet": ModelSpec(
            key="pfgnet",
            display_name="PFGNet",
            save_dir=args.pfgnet_save_dir,
            best_checkpoint_filename="best_pfgnet.pth",
            model_signature="PFGNet(SimVP-style PFG blocks)",
        ),
    }


def create_model(model_spec: ModelSpec, args: argparse.Namespace) -> torch.nn.Module:
    in_chans = len(args.env_vars) + 1
    if model_spec.key == "predrnn":
        return PredRNN(
            in_chans=in_chans,
            hidden_dim=args.hidden_dim,
            img_size=(64, 96),
            num_layers=args.predrnn_num_layers,
        )
    if model_spec.key == "pfgnet":
        return PFGNet(
            in_chans=in_chans,
            hidden_dim=args.hidden_dim,
            seq_len=args.seq_len,
            img_size=(64, 96),
            num_layers=args.pfgnet_num_layers,
        )
    raise ValueError(f"Unsupported model key: {model_spec.key}")


def build_model_root_config(
    model_spec: ModelSpec,
    windows: Sequence,
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> Dict[str, object]:
    return {
        "model_key": model_spec.key,
        "model_label": model_spec.display_name,
        "model_signature": model_spec.model_signature,
        "windows": [asdict(window) for window in windows],
        "window_years": [window.test_year for window in windows],
        "env_vars": list(args.env_vars),
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "batch_size": int(args.batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "train_loss": str(args.train_loss),
        "hotspot_threshold": float(args.hotspot_threshold),
        "extra_hotspot_thresholds": list(args.extra_hotspot_thresholds),
        "hidden_dim": int(args.hidden_dim),
        "predrnn_num_layers": int(args.predrnn_num_layers),
        "pfgnet_num_layers": int(args.pfgnet_num_layers),
        "lr_scheduler_factor": float(args.lr_scheduler_factor),
        "lr_scheduler_patience": int(args.lr_scheduler_patience),
        "min_lr": float(args.min_lr),
        "early_stopping_patience": int(args.early_stopping_patience),
        "max_grad_norm": float(args.max_grad_norm),
        "seeds": list(seeds),
        "data_dir": str(args.data_dir),
        "mask_path": str(args.mask_path),
        "save_dir": str(model_spec.save_dir),
    }


def build_window_metadata(
    model_spec: ModelSpec,
    window,
    time_axis,
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> Dict[str, object]:
    return {
        "model_key": model_spec.key,
        "model_label": model_spec.display_name,
        "model_signature": model_spec.model_signature,
        "window": asdict(window),
        "window_name": window.name,
        "test_year": int(window.test_year),
        "train_range": format_range(time_axis, window.train_start, window.train_end),
        "val_range": format_range(time_axis, window.val_start, window.val_end),
        "test_range": format_range(time_axis, window.test_start, window.test_end),
        "rollout_range": format_range(time_axis, window.test_start, window.test_end),
        "env_vars": list(args.env_vars),
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "batch_size": int(args.batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "train_loss": str(args.train_loss),
        "hotspot_threshold": float(args.hotspot_threshold),
        "extra_hotspot_thresholds": list(args.extra_hotspot_thresholds),
        "hidden_dim": int(args.hidden_dim),
        "predrnn_num_layers": int(args.predrnn_num_layers),
        "pfgnet_num_layers": int(args.pfgnet_num_layers),
        "seeds": list(seeds),
    }


def train_single_window_for_model(
    model_spec: ModelSpec,
    window,
    env_full,
    ais_full,
    time_axis,
    args: argparse.Namespace,
    mask_tensor,
    device: torch.device,
    seeds: Sequence[int],
) -> None:
    window_save_dir = model_spec.save_dir / window.name
    window_save_dir.mkdir(parents=True, exist_ok=True)
    save_json(build_window_metadata(model_spec, window, time_axis, args, seeds), window_save_dir / "experiment_config.json")

    train_dataset = TemporalWindowDataset(
        env_full,
        ais_full,
        split_start=window.train_start,
        split_end=window.train_end,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split_name="train",
    )
    val_dataset = TemporalWindowDataset(
        env_full,
        ais_full,
        split_start=window.val_start,
        split_end=window.val_end,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split_name="val",
    )
    test_dataset = TemporalWindowDataset(
        env_full,
        ais_full,
        split_start=window.test_start,
        split_end=window.test_end,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        split_name="test",
    )

    run_rows: List[Dict] = []
    use_amp = device.type == "cuda"
    mask_np = mask_tensor.detach().cpu().numpy() if mask_tensor is not None else None

    for seed in seeds:
        set_seed(seed)
        run_save_dir = Path(resolve_seed_save_dir(str(window_save_dir), seed, seeds))
        run_save_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        model = create_model(model_spec, args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.min_lr,
        )
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_epochs = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_model_path = run_save_dir / model_spec.best_checkpoint_filename

        print(f"\n[{model_spec.display_name} | {window.name}] Starting training with seed={seed}...")
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(
                train_loader,
                desc=f"{model_spec.key} | {window.name} | Seed {seed} | Epoch {epoch + 1}/{args.epochs} [Train]",
            )

            for batch_index, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                targets = targets.to(device)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    preds = model(inputs)
                    if args.train_loss.lower() == "mae":
                        loss = masked_mae_loss(preds, targets, mask=mask_tensor)
                    else:
                        loss = masked_mse_loss(preds, targets, mask=mask_tensor)
                    loss = loss / args.accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if ((batch_index + 1) % args.accumulation_steps == 0) or ((batch_index + 1) == len(train_loader)):
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                real_loss = loss.item() * args.accumulation_steps
                running_loss += real_loss
                pbar.set_postfix({"loss": f"{real_loss:.6f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            avg_train_loss = running_loss / max(len(train_loader), 1)
            avg_val_loss = evaluate(model, val_loader, device, args.train_loss, mask_tensor)
            scheduler.step(avg_val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

            history["train_loss"].append(float(avg_train_loss))
            history["val_loss"].append(float(avg_val_loss))
            history["lr"].append(current_lr)

            print(
                f"{model_spec.display_name} | {window.name} | Seed {seed} | Epoch {epoch + 1:03d} | "
                f"Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f} | LR = {current_lr:.2e}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"[{model_spec.display_name} | {window.name}] Best model updated: {best_model_path}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.early_stopping_patience:
                    print(f"[{model_spec.display_name} | {window.name}] Early stopping triggered at epoch {epoch + 1}.")
                    break

        save_json(history, run_save_dir / "training_history.json")

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)

        one_step_metrics, one_step_preds, one_step_targets = test_and_save_one_step(
            model,
            test_loader,
            device,
            run_save_dir,
            mask_tensor,
            args.hotspot_threshold,
        )
        save_example_predictions(run_save_dir, one_step_preds, one_step_targets, prefix="test_one_step_example")

        rollout_preds, rollout_targets = rollout_predict_window(
            model,
            env_full,
            ais_full,
            device,
            args.seq_len,
            window.rollout_horizon,
            window.rollout_start_index,
            mask_np,
        )
        rollout_metrics = save_rollout_results(
            run_save_dir,
            rollout_preds,
            rollout_targets,
            mask_tensor,
            args.hotspot_threshold,
        )
        save_example_predictions(run_save_dir, rollout_preds, rollout_targets, prefix="rollout_12m_example")

        run_row = build_seed_run_row(seed, best_val_loss, best_epoch, one_step_metrics, rollout_metrics)
        save_json(run_row, run_save_dir / "run_summary.json")
        run_rows.append(run_row)

        print(f"\n[{model_spec.display_name} | {window.name}] Final one-step test metrics:")
        for key, value in one_step_metrics.items():
            print(f"  {key}: {value:.6f}")
        print(f"\n[{model_spec.display_name} | {window.name}] Final 12-month rollout metrics:")
        for key, value in rollout_metrics.items():
            print(f"  {key}: {value:.6f}")

    finalize_window_experiment(
        base_save_dir=window_save_dir,
        seeds=seeds,
        run_rows=run_rows,
        mask_path=args.mask_path,
        hotspot_threshold=args.hotspot_threshold,
        best_checkpoint_filename=model_spec.best_checkpoint_filename,
    )
    save_additional_threshold_evaluations(
        base_save_dir=window_save_dir,
        seeds=seeds,
        mask_path=args.mask_path,
        hotspot_thresholds=args.extra_hotspot_thresholds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal backtests for baseline models.")
    parser.add_argument(
        "--models",
        type=str,
        default="predrnn,pfgnet",
        help="Comma-separated model keys to run: predrnn,pfgnet.",
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory containing feature npy files.")
    parser.add_argument("--mask-path", type=str, default=str(DEFAULT_MASK_PATH), help="Shared ocean mask path.")
    parser.add_argument("--predrnn-save-dir", type=str, default=str(DEFAULT_PREDRNN_SAVE_DIR), help="PredRNN backtest output root.")
    parser.add_argument("--pfgnet-save-dir", type=str, default=str(DEFAULT_PFGNET_SAVE_DIR), help="PFGNet backtest output root.")
    parser.add_argument("--window-years", type=str, default="2022,2023,2024", help="Comma-separated test years to run.")
    parser.add_argument("--env-vars", type=str, default=",".join(DEFAULT_ENV_VARS), help="Comma-separated environment variables.")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Input sequence length.")
    parser.add_argument("--pred-len", type=int, default=PRED_LEN, help="Prediction length.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--accumulation-steps", type=int, default=ACCUMULATION_STEPS, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Adam weight decay.")
    parser.add_argument("--train-loss", type=str, default=TRAIN_LOSS, help="Training loss name: mse or mae.")
    parser.add_argument("--hotspot-threshold", type=float, default=HOTSPOT_THRESHOLD, help="Primary hotspot threshold.")
    parser.add_argument(
        "--extra-hotspot-thresholds",
        type=str,
        default="0.3175",
        help="Comma-separated extra thresholds to recompute CSI/F1 after training.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Shared model hidden dimension.")
    parser.add_argument("--predrnn-num-layers", type=int, default=2, help="PredRNN STLSTM layer count.")
    parser.add_argument("--pfgnet-num-layers", type=int, default=4, help="PFGNet translator block count.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=LR_SCHEDULER_FACTOR, help="ReduceLROnPlateau factor.")
    parser.add_argument("--lr-scheduler-patience", type=int, default=LR_SCHEDULER_PATIENCE, help="ReduceLROnPlateau patience.")
    parser.add_argument("--min-lr", type=float, default=MIN_LR, help="ReduceLROnPlateau minimum learning rate.")
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM, help="Gradient clipping max norm.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = Path(args.data_dir).resolve()
    args.mask_path = Path(args.mask_path).resolve()
    args.predrnn_save_dir = Path(args.predrnn_save_dir).resolve()
    args.pfgnet_save_dir = Path(args.pfgnet_save_dir).resolve()
    args.env_vars = [item.strip() for item in args.env_vars.split(",") if item.strip()]
    args.extra_hotspot_thresholds = parse_float_list(args.extra_hotspot_thresholds)

    model_keys = parse_name_list(args.models)
    seeds = parse_int_list(args.seeds, DEFAULT_SEEDS)
    requested_years = parse_int_list(args.window_years, [2022, 2023, 2024])
    windows = [WINDOW_CONFIGS[year] for year in requested_years if year in WINDOW_CONFIGS]
    if not windows:
        raise RuntimeError(f"No valid windows selected from: {requested_years}")

    all_specs = build_model_specs(args)
    selected_specs = [all_specs[key] for key in model_keys if key in all_specs]
    if not selected_specs:
        raise RuntimeError(f"No valid baseline models selected from: {model_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_tensor = None
    if args.mask_path.exists():
        mask_tensor = torch.from_numpy(np.load(args.mask_path).astype(np.float32)).to(device)

    env_full, ais_full = load_full_series(args.data_dir, args.env_vars)
    time_axis = load_time_axis(args.data_dir)
    if len(time_axis) != len(env_full):
        raise RuntimeError(
            f"Time axis length mismatch: len(time_axis)={len(time_axis)} vs len(env_full)={len(env_full)}"
        )

    for model_spec in selected_specs:
        model_spec.save_dir.mkdir(parents=True, exist_ok=True)
        save_json(build_model_root_config(model_spec, windows, args, seeds), model_spec.save_dir / "backtest_config.json")

        for window in windows:
            train_single_window_for_model(
                model_spec=model_spec,
                window=window,
                env_full=env_full,
                ais_full=ais_full,
                time_axis=time_axis,
                args=args,
                mask_tensor=mask_tensor,
                device=device,
                seeds=seeds,
            )

        save_root_summaries(
            save_dir=model_spec.save_dir,
            windows=windows,
            time_axis=time_axis,
            hotspot_threshold=args.hotspot_threshold,
            extra_thresholds=args.extra_hotspot_thresholds,
        )

        print(f"\n{model_spec.display_name} temporal backtest completed.")
        print(f"Saved root summary to: {model_spec.save_dir / 'temporal_backtest_summary.csv'}")
        for window in windows:
            print(f"Saved window outputs to: {model_spec.save_dir / window.name}")


if __name__ == "__main__":
    main()
