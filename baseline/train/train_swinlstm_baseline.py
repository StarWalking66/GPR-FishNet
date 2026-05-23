import os
import sys
import json
import random
from typing import Dict, Optional, Tuple, List
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.output_clamp import apply_output_clamp
from utils.multi_seed_experiment import (
    DEFAULT_SEEDS,
    build_seed_run_row,
    finalize_multi_seed_experiment,
    resolve_seed_save_dir,
)

# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "models") 
sys.path.append(model_dir)

from process.dataset import STFishNetUltimateDataset
from baseline.models.SwinLSTM_B_model import SwinLSTM, SwinLSTMBaseline as ModelSwinLSTMBaseline
from baseline.train.fair_benchmark_config import (
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    ENV_VARS,
    HOTSPOT_THRESHOLD,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_GRAD_NORM,
    MIN_LR,
    PRED_LEN,
    ROLLOUT_2024_HORIZON,
    ROLLOUT_START_INDEX,
    SEQ_LEN,
    TRAIN_LOSS,
    WEIGHT_DECAY,
)

# =====================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(obj: Dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

def resolve_model_profile(profile: str):
    profiles = {
        "paper_taxi": {
            "patch_size": 4, "embed_dim": 96, "depths": (12,), 
            "num_heads": (8,), "window_size": 4, "drop_path_rate": 0.1,
        },
        "fish_small": {
            "patch_size": 4, "embed_dim": 64, "depths": (4,), 
            "num_heads": (4,), "window_size": 4, "drop_path_rate": 0.05,
        },
        "fish_fine": {
            "patch_size": 2, "embed_dim": 64, "depths": (4,), 
            "num_heads": (4,), "window_size": 4, "drop_path_rate": 0.05,
        },
        "fish_fair": {
            "patch_size": 2, "embed_dim": 64, "depths": (2, 2),
            "num_heads": (4, 4), "window_size": 4, "drop_path_rate": 0.05,
        },
    }
    if profile not in profiles:
        raise ValueError(f"Unknown MODEL_PROFILE: {profile}, available: {list(profiles.keys())}")
    return profiles[profile]

def expand_mask_like(ref_tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask = mask.float().to(ref_tensor.device)
    if mask.ndim == 2:
        while mask.ndim < ref_tensor.ndim:
            mask = mask.unsqueeze(0)
    elif mask.ndim == 3:
        if ref_tensor.ndim == 4:
            mask = mask.unsqueeze(1)
        elif ref_tensor.ndim == 5:
            mask = mask.unsqueeze(1).unsqueeze(1)
    return (mask > 0).float().expand_as(ref_tensor)

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None: return torch.mean((pred - target) ** 2)
    expanded_mask = expand_mask_like(pred, mask)
    diff2 = ((pred - target) ** 2) * expanded_mask
    denom = expanded_mask.sum().clamp_min(1.0)
    return diff2.sum() / denom

def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None: return torch.mean(torch.abs(pred - target))
    expanded_mask = expand_mask_like(pred, mask)
    diff1 = torch.abs(pred - target) * expanded_mask
    denom = expanded_mask.sum().clamp_min(1.0)
    return diff1.sum() / denom

def compute_metrics(
    preds: np.ndarray, targets: np.ndarray, mask_tensor: Optional[torch.Tensor] = None, hotspot_threshold: float = 0.2755,
) -> Dict[str, float]:
    preds_t = torch.from_numpy(preds).float()
    targets_t = torch.from_numpy(targets).float()

    valid_mask = torch.ones_like(preds_t) if mask_tensor is None else expand_mask_like(preds_t, mask_tensor.cpu())
    valid_mask_bool = valid_mask > 0.5
    
    p = preds_t[valid_mask_bool].numpy().astype(np.float64)
    t = targets_t[valid_mask_bool].numpy().astype(np.float64)

    if p.size == 0: return {"MAE": 0.0, "MSE": 0.0, "RMSE": 0.0, "R2": 0.0, "SSIM": 0.0, "CSI": 0.0, "F1": 0.0}

    diff = p - t
    mae, mse = float(np.abs(diff).mean()), float((diff ** 2).mean())
    rmse = float(np.sqrt(mse))

    ss_res, ss_tot = np.sum(diff ** 2), np.sum((t - np.mean(t)) ** 2) + 1e-8
    r2 = float(1.0 - ss_res / ss_tot)

    mu_x, mu_y, var_x, var_y = np.mean(p), np.mean(t), np.var(p), np.var(t)
    cov_xy = np.mean((p - mu_x) * (t - mu_y))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = float(((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))

    pred_bin, target_bin = (p >= hotspot_threshold).astype(np.uint8), (t >= hotspot_threshold).astype(np.uint8)
    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))

    csi = float(tp / (tp + fp + fn + 1e-8))
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "SSIM": ssim, "CSI": csi, "F1": f1}

def load_full_series(data_dir: str, env_vars: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    env_list = []
    for var in env_vars:
        train_arr = np.load(os.path.join(data_dir, f"{var}_train.npy"))
        val_arr = np.load(os.path.join(data_dir, f"{var}_val.npy"))
        test_arr = np.load(os.path.join(data_dir, f"{var}_test.npy"))
        full_arr = np.concatenate([train_arr, val_arr, test_arr], axis=0)
        env_list.append(full_arr)
    env_full = np.concatenate(env_list, axis=1).astype(np.float32)

    ais_train = np.load(os.path.join(data_dir, "ais_train.npy"))
    ais_val = np.load(os.path.join(data_dir, "ais_val.npy"))
    ais_test = np.load(os.path.join(data_dir, "ais_test.npy"))
    ais_full = np.concatenate([ais_train, ais_val, ais_test], axis=0).astype(np.float32)

    return env_full, ais_full

# =====================================================================
class BaselineWrapper(nn.Module):
    """
    Baseline SwinLSTM wrapper.
    Input:  [B, T, 8, H, W]
    Output: [B, 1, 1, H, W]
    """
    def __init__(
        self,
        img_size=(64, 96),
        patch_size=4,
        in_chans=8,
        embed_dim=96,
        depths=(12,),
        num_heads=(8,),
        window_size=4,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.core = SwinLSTM(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.num_layers = len(depths)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, t_hist, _, _, _ = inputs.shape
        states = [None] * self.num_layers
        cur_out = None

        for t in range(t_hist):
            frame_t = inputs[:, t]
            cur_out, states = self.core(frame_t, states)
        return cur_out.unsqueeze(1)

# =====================================================================
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, loss_name: str, mask_tensor: Optional[torch.Tensor] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = apply_output_clamp(model(inputs))
            loss = masked_mae_loss(preds, targets, mask=mask_tensor) if loss_name.lower() == "mae" else masked_mse_loss(preds, targets, mask=mask_tensor)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def test_and_save_one_step(
    model: nn.Module, loader: DataLoader, device: torch.device, save_dir: str, mask_tensor: Optional[torch.Tensor] = None, hotspot_threshold: float = 0.2755, file_prefix: str = "test_one_step",
):
    model.eval()
    all_preds, all_targets, all_inputs = [], [], []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"[{file_prefix.upper()}] Inference"):
            inputs = inputs.to(device)
            preds = apply_output_clamp(model(inputs))

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)

    pred_path = os.path.join(save_dir, f"{file_prefix}_preds.npy")
    target_path = os.path.join(save_dir, f"{file_prefix}_targets.npy")
    input_path = os.path.join(save_dir, f"{file_prefix}_inputs.npy")
    metrics_path = os.path.join(save_dir, f"{file_prefix}_metrics.json")

    np.save(pred_path, all_preds)
    np.save(target_path, all_targets)
    np.save(input_path, all_inputs)

    metrics = compute_metrics(
        preds=all_preds, targets=all_targets, mask_tensor=mask_tensor, hotspot_threshold=hotspot_threshold,
    )
    save_json(metrics, metrics_path)

    print(f"\n[{file_prefix}] metrics:")
    for k, v in metrics.items():
        print(f"  {k} = {v:.6f}")

    return metrics, all_preds, all_targets


def rollout_predict_2024(
    model: nn.Module, data_dir: str, env_vars: List[str], device: torch.device, seq_len: int = 12, rollout_horizon: int = 12, rollout_start_index: int = 144, mask_np: Optional[np.ndarray] = None,
):
    model.eval()
    env_full, ais_full = load_full_series(data_dir, env_vars)

    hist_env = env_full[rollout_start_index - seq_len: rollout_start_index].copy()  
    hist_ais = ais_full[rollout_start_index - seq_len: rollout_start_index].copy()  

    preds_2024, targets_2024 = [], []

    with torch.no_grad():
        for step in range(rollout_horizon):
            current_input = np.concatenate([hist_env, hist_ais], axis=1)  
            input_tensor = torch.from_numpy(current_input[None]).float().to(device)  

            pred = apply_output_clamp(model(input_tensor))             
            pred_step = pred[0, 0].cpu().numpy()

            if mask_np is not None:
                pred_step[0, mask_np == 0] = 0.0

            target_step = ais_full[rollout_start_index + step]  
            env_step = env_full[rollout_start_index + step]     

            preds_2024.append(pred_step)
            targets_2024.append(target_step)

            hist_env = np.concatenate([hist_env[1:], env_step[None]], axis=0)
            hist_ais = np.concatenate([hist_ais[1:], pred_step[None]], axis=0)

    preds_2024 = np.stack(preds_2024, axis=0).astype(np.float32)      
    targets_2024 = np.stack(targets_2024, axis=0).astype(np.float32)  

    return preds_2024, targets_2024


def save_rollout_results(
    save_dir: str, preds_2024: np.ndarray, targets_2024: np.ndarray, mask_tensor: Optional[torch.Tensor] = None, hotspot_threshold: float = 0.2755, file_prefix: str = "rollout_2024",
):
    os.makedirs(save_dir, exist_ok=True)
    pred_path = os.path.join(save_dir, f"{file_prefix}_preds.npy")
    target_path = os.path.join(save_dir, f"{file_prefix}_targets.npy")
    metrics_path = os.path.join(save_dir, f"{file_prefix}_metrics.json")

    np.save(pred_path, preds_2024)
    np.save(target_path, targets_2024)

    metrics = compute_metrics(
        preds=preds_2024, targets=targets_2024, mask_tensor=mask_tensor, hotspot_threshold=hotspot_threshold,
    )
    save_json(metrics, metrics_path)

    print(f"\n[{file_prefix}] metrics:")
    for k, v in metrics.items():
        print(f"  {k} = {v:.6f}")

    return metrics


def save_example_predictions(
    save_dir: str, preds: np.ndarray, targets: np.ndarray, num_examples: int = 5, prefix: str = "example",
):
    num_examples = min(num_examples, preds.shape[0])
    example_dir = os.path.join(save_dir, f"{prefix}_samples")
    os.makedirs(example_dir, exist_ok=True)

    for i in range(num_examples):
        np.save(os.path.join(example_dir, f"{prefix}_{i}_pred.npy"), preds[i])
        np.save(os.path.join(example_dir, f"{prefix}_{i}_target.npy"), targets[i])
    print(f"Saved {num_examples} examples to: {example_dir}")

# =====================================================================
def main():
    # ================= paths =================
    DATA_DIR = r"D:\VsCode Space\SA_PredRNN\data\ST_FishNet_Features"
    MASK_PATH = r"D:\VsCode Space\SA_PredRNN\data\ST_FishNet_Features\all_vars_train_mask_intersection.npy"
    SAVE_DIR = r"./model_outcomes/checkpoints_swinlstm_baseline"
    

    # ================= variables =================
    SEEDS = DEFAULT_SEEDS

    # ================= optimization =================
    MODEL_PROFILE = "fish_fair"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Model profile: {MODEL_PROFILE}")
    print(f"Train loss: {TRAIN_LOSS}")

    # ================= load mask =================
    mask_tensor = None
    if os.path.exists(MASK_PATH):
        mask_np = np.load(MASK_PATH).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).to(device)
    else:
        print("Warning: no mask loaded, training/evaluation will use full grid.")

    profile = resolve_model_profile(MODEL_PROFILE)
    run_rows = []
    for seed in SEEDS:
        set_seed(seed)
        scaler = torch.amp.GradScaler(enabled=use_amp)
        run_save_dir = resolve_seed_save_dir(SAVE_DIR, seed, SEEDS)
        os.makedirs(run_save_dir, exist_ok=True)

        train_dataset = STFishNetUltimateDataset(
            data_dir=DATA_DIR, env_vars=ENV_VARS, split="train", seq_len=SEQ_LEN, pred_len=PRED_LEN, return_future_env=False,
        )
        val_dataset = STFishNetUltimateDataset(
            data_dir=DATA_DIR, env_vars=ENV_VARS, split="val", seq_len=SEQ_LEN, pred_len=PRED_LEN, return_future_env=False,
        )
        test_dataset = STFishNetUltimateDataset(
            data_dir=DATA_DIR, env_vars=ENV_VARS, split="test", seq_len=SEQ_LEN, pred_len=PRED_LEN, return_future_env=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        model = ModelSwinLSTMBaseline(
            img_size=(64, 96),
            patch_size=profile["patch_size"],
            in_chans=8,
            embed_dim=profile["embed_dim"],
            depths=profile["depths"],
            num_heads=profile["num_heads"],
            window_size=profile["window_size"],
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=profile["drop_path_rate"],
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, min_lr=MIN_LR,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_epochs = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_model_path = os.path.join(run_save_dir, "best_swinlstm_baseline.pth")

        print(f"Starting SwinLSTM baseline training with seed={seed}...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(train_loader, desc=f"Seed {seed} | Epoch {epoch + 1}/{EPOCHS} [Train]")

            for i, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    preds = model(inputs)

                    loss = masked_mae_loss(preds, targets, mask=mask_tensor) if TRAIN_LOSS.lower() == "mae" else masked_mse_loss(preds, targets, mask=mask_tensor)
                    loss = loss / ACCUMULATION_STEPS

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                should_step = ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(train_loader))
                if should_step:
                    if use_amp:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                real_loss = loss.item() * ACCUMULATION_STEPS
                running_loss += real_loss
                pbar.set_postfix({"loss": f"{real_loss:.6f}"})

            avg_train_loss = running_loss / max(len(train_loader), 1)
            avg_val_loss = evaluate(
                model=model, loader=val_loader, device=device, loss_name=TRAIN_LOSS, mask_tensor=mask_tensor,
            )

            scheduler.step(avg_val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

            history["train_loss"].append(float(avg_train_loss))
            history["val_loss"].append(float(avg_val_loss))
            history["lr"].append(current_lr)

            print(f"\nSeed {seed} | Epoch {epoch + 1:03d} | Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f} | LR = {current_lr:.2e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"SwinLSTM best model updated: {best_model_path}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        history_path = os.path.join(run_save_dir, "training_history.json")
        save_json(history, history_path)

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)

        one_step_metrics, one_step_preds, one_step_targets = test_and_save_one_step(
            model=model, loader=test_loader, device=device, save_dir=run_save_dir,
            mask_tensor=mask_tensor, hotspot_threshold=HOTSPOT_THRESHOLD, file_prefix="test_one_step",
        )

        save_example_predictions(run_save_dir, one_step_preds, one_step_targets, prefix="test_one_step_example")

        mask_np_for_rollout = None if mask_tensor is None else mask_tensor.detach().cpu().numpy()

        rollout_preds_2024, rollout_targets_2024 = rollout_predict_2024(
            model=model, data_dir=DATA_DIR, env_vars=ENV_VARS, device=device, seq_len=SEQ_LEN,
            rollout_horizon=ROLLOUT_2024_HORIZON, rollout_start_index=ROLLOUT_START_INDEX, mask_np=mask_np_for_rollout,
        )

        rollout_metrics = save_rollout_results(
            save_dir=run_save_dir, preds_2024=rollout_preds_2024, targets_2024=rollout_targets_2024,
            mask_tensor=mask_tensor, hotspot_threshold=HOTSPOT_THRESHOLD, file_prefix="rollout_2024",
        )

        save_example_predictions(run_save_dir, rollout_preds_2024, rollout_targets_2024, prefix="rollout_2024_example")
        run_row = build_seed_run_row(seed, best_val_loss, best_epoch, one_step_metrics, rollout_metrics)
        save_json(run_row, os.path.join(run_save_dir, "run_summary.json"))
        run_rows.append(run_row)
        print("\nSwinLSTM final one-step test metrics:")
        for metric_name, metric_value in one_step_metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")
        print("\nSwinLSTM final 2024 rollout metrics:")
        for metric_name, metric_value in rollout_metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")

    finalize_multi_seed_experiment(
        base_save_dir=SAVE_DIR,
        seeds=SEEDS,
        run_rows=run_rows,
        mask_path=MASK_PATH,
        hotspot_threshold=HOTSPOT_THRESHOLD,
        best_checkpoint_filename="best_swinlstm_baseline.pth",
    )

if __name__ == "__main__":
    main()
