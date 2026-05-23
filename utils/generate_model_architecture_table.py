from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model_outcomes" / "architecture_tables"
DEFAULT_MASK_PATH = PROJECT_ROOT / "data" / "ST_FishNet_Features" / "all_vars_train_mask_intersection.npy"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    role: str
    configuration: str
    factory: Callable[[], Any]


def count_trainable_params(module: Any) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def format_number(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return f"{value:,}"


def resolve_output_prefixes(output_dir: Path, explicit_output_prefix: str) -> Tuple[Path, Path]:
    if explicit_output_prefix:
        prefix = Path(explicit_output_prefix)
        if not prefix.is_absolute():
            prefix = (PROJECT_ROOT / prefix).resolve()
        return prefix, prefix

    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    archive_prefix = output_dir / f"gpr_fishnet_architecture_table_{timestamp}"
    latest_prefix = output_dir / "gpr_fishnet_architecture_table_latest"
    return archive_prefix, latest_prefix


def output_paths(prefix: Path) -> Dict[str, Path]:
    return {
        "architecture_csv": prefix.with_name(f"{prefix.name}_gpr_architecture.csv"),
        "compute_csv": prefix.with_name(f"{prefix.name}_compute_comparison.csv"),
        "markdown": prefix.with_suffix(".md"),
        "xlsx": prefix.with_suffix(".xlsx"),
    }


def copy_latest_outputs(archive_prefix: Path, latest_prefix: Path) -> None:
    if archive_prefix == latest_prefix:
        return
    archive_paths = output_paths(archive_prefix)
    latest_paths = output_paths(latest_prefix)
    for key, archive_path in archive_paths.items():
        if archive_path.exists():
            latest_paths[key].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive_path, latest_paths[key])


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = []
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        body_lines.append("| " + " | ".join(value.replace("\n", " ") for value in values) + " |")
    return "\n".join([header_line, sep_line, *body_lines])


def write_markdown(
    path: Path,
    architecture_headers: Sequence[str],
    architecture_rows: Sequence[Dict[str, Any]],
    compute_headers: Sequence[str],
    compute_rows: Sequence[Dict[str, Any]],
    notes: Sequence[Tuple[str, Any]],
) -> None:
    lines = [
        "# GPR-FishNet architecture and compute summary",
        "",
        "## Main model architecture",
        "",
        markdown_table(architecture_headers, architecture_rows),
        "",
        "## Model-size and compute comparison",
        "",
        markdown_table(compute_headers, compute_rows),
        "",
        "## Notes",
        "",
    ]
    for key, value in notes:
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_xlsx(path: Path, sheet_specs: Sequence[Dict[str, Any]]) -> bool:
    try:
        from utils.generate_model_ranking_xlsx import save_xlsx_fallback
    except Exception:
        return False
    save_xlsx_fallback(list(sheet_specs), path)
    return True


def add_module_param(rows: List[Dict[str, Any]], component: str, params: Optional[int]) -> None:
    for row in rows:
        if row["component"] == component:
            row["trainable_params"] = format_int(params)
            row["params_m"] = format_number(None if params is None else params / 1e6, 4)
            return


def build_static_architecture_rows() -> List[Dict[str, Any]]:
    return [
        {
            "stage": "Input",
            "component": "History tensor",
            "layers": "-",
            "channels": "8 input channels = 7 environmental factors + AIS",
            "kernel_size": "-",
            "dilation": "-",
            "stride_padding": "-",
            "output_shape": "[B, 12, 8, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "seq_len=12, pred_len=1, grid=64x96",
        },
        {
            "stage": "STLSTM encoder",
            "component": "STLSTM layer 1",
            "layers": "1",
            "channels": "input 8, hidden 64, memory 64",
            "kernel_size": "conv_h 3x3; conv_m 3x3; conv_o 3x3; conv_last 1x1",
            "dilation": "1",
            "stride_padding": "stride 1; padding 1 for 3x3",
            "output_shape": "h1 [B, 64, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "forget bias=1.0; LayerNorm on gate tensors",
        },
        {
            "stage": "STLSTM encoder",
            "component": "STLSTM layer 2",
            "layers": "1",
            "channels": "input 64, hidden 64, memory 64",
            "kernel_size": "conv_h 3x3; conv_m 3x3; conv_o 3x3; conv_last 1x1",
            "dilation": "1",
            "stride_padding": "stride 1; padding 1 for 3x3",
            "output_shape": "h2 [B, 64, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "default encoder depth is 2 STLSTM layers",
        },
        {
            "stage": "Fusion",
            "component": "Hidden-state concatenation",
            "layers": "-",
            "channels": "64 + 64 = 128",
            "kernel_size": "-",
            "dilation": "-",
            "stride_padding": "-",
            "output_shape": "fusion_h [B, 128, 64, 96]",
            "trainable_params": "0",
            "params_m": "0.0000",
            "notes": "concat first and last STLSTM hidden states",
        },
        {
            "stage": "ARP",
            "component": "ARP CBAM",
            "layers": "channel + spatial attention",
            "channels": "128 -> 16 -> 128; spatial 2 -> 1",
            "kernel_size": "1x1 channel MLP; 7x7 spatial attention",
            "dilation": "1",
            "stride_padding": "padding 3 for 7x7",
            "output_shape": "h_attn [B, 128, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "CBAM ratio=8 with residual attention output",
        },
        {
            "stage": "ARP",
            "component": "ARP residual predictor",
            "layers": "3 convs + shortcut",
            "channels": "128 -> 64 -> 32 -> 1; shortcut 128 -> 1",
            "kernel_size": "3x3, 3x3, 1x1; shortcut 1x1",
            "dilation": "1",
            "stride_padding": "padding 1 for 3x3",
            "output_shape": "pred_gen [B, 1, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "GroupNorm(8) after first 3x3; LeakyReLU(0.1)",
        },
        {
            "stage": "MSSP",
            "component": "MSSP local branch",
            "layers": "1 conv branch",
            "channels": "129 -> 16",
            "kernel_size": "3x3",
            "dilation": "1",
            "stride_padding": "stride 1; padding 1",
            "output_shape": "local_feat [B, 16, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "input is concat(last AIS observation, h_attn)",
        },
        {
            "stage": "MSSP",
            "component": "MSSP surround branch",
            "layers": "1 dilated conv branch",
            "channels": "129 -> 16",
            "kernel_size": "3x3",
            "dilation": "2",
            "stride_padding": "stride 1; padding 2",
            "output_shape": "surround_feat [B, 16, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "effective receptive field is 5x5",
        },
        {
            "stage": "MSSP",
            "component": "MSSP fusion",
            "layers": "1 conv",
            "channels": "32 -> 1",
            "kernel_size": "1x1",
            "dilation": "1",
            "stride_padding": "stride 1; padding 0",
            "output_shape": "pred_pers [B, 1, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "returns last_obs + persistence_delta",
        },
        {
            "stage": "CAR",
            "component": "Context-aware router",
            "layers": "2 convs + sigmoid",
            "channels": "129 -> 16 -> 2",
            "kernel_size": "3x3, 1x1",
            "dilation": "1",
            "stride_padding": "padding 1 for 3x3",
            "output_shape": "two gates [B, 2, 64, 96]",
            "trainable_params": "",
            "params_m": "",
            "notes": "independent sigmoid gates for ARP and MSSP branches",
        },
        {
            "stage": "Output",
            "component": "Weighted sum + ReLU",
            "layers": "-",
            "channels": "1",
            "kernel_size": "-",
            "dilation": "-",
            "stride_padding": "-",
            "output_shape": "[B, 1, 1, 64, 96]",
            "trainable_params": "0",
            "params_m": "0.0000",
            "notes": "non-negative fishing-ground heatmap",
        },
    ]


def attach_gpr_module_params(rows: List[Dict[str, Any]], model: Any) -> None:
    add_module_param(rows, "STLSTM layer 1", count_trainable_params(model.cell_list[0]))
    add_module_param(rows, "STLSTM layer 2", count_trainable_params(model.cell_list[1]))
    add_module_param(rows, "ARP CBAM", count_trainable_params(model.predictor.cbam))
    add_module_param(
        rows,
        "ARP residual predictor",
        count_trainable_params(model.predictor.conv_net) + count_trainable_params(model.predictor.shortcut),
    )
    add_module_param(rows, "MSSP local branch", count_trainable_params(model.persister.conv_local))
    add_module_param(rows, "MSSP surround branch", count_trainable_params(model.persister.conv_surround))
    add_module_param(rows, "MSSP fusion", count_trainable_params(model.persister.fuse))
    add_module_param(rows, "Context-aware router", count_trainable_params(model.router))


def count_hook_macs(torch: Any, nn: Any, model: Any, sample_input: Any) -> Tuple[int, Tuple[int, ...]]:
    macs_by_module: Dict[str, int] = {}
    hooks = []

    def add_macs(name: str, value: int) -> None:
        macs_by_module[name] = macs_by_module.get(name, 0) + int(value)

    def module_hook(name: str, module: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        tensor_out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(module, nn.Conv2d):
            batch, out_channels, out_h, out_w = tensor_out.shape
            kernel_h, kernel_w = module.kernel_size
            in_channels = module.in_channels // module.groups
            add_macs(name, batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w)
        elif isinstance(module, nn.ConvTranspose2d):
            batch, out_channels, out_h, out_w = tensor_out.shape
            kernel_h, kernel_w = module.kernel_size
            in_channels = module.in_channels // module.groups
            add_macs(name, batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w)
        elif isinstance(module, nn.Linear):
            output_elems = tensor_out.numel() // module.out_features
            add_macs(name, output_elems * module.in_features * module.out_features)
        elif isinstance(module, nn.GRU):
            input_tensor = inputs[0]
            batch = input_tensor.shape[0] if module.batch_first else input_tensor.shape[1]
            seq_len = input_tensor.shape[1] if module.batch_first else input_tensor.shape[0]
            directions = 2 if module.bidirectional else 1
            for layer_idx in range(module.num_layers):
                layer_input_size = module.input_size if layer_idx == 0 else module.hidden_size * directions
                add_macs(
                    name,
                    batch
                    * seq_len
                    * directions
                    * 3
                    * module.hidden_size
                    * (layer_input_size + module.hidden_size),
                )

    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.GRU)):
            hooks.append(
                module.register_forward_hook(
                    lambda module, inputs, output, module_name=module_name: module_hook(
                        module_name,
                        module,
                        inputs,
                        output,
                    )
                )
            )

    model.eval()
    with torch.no_grad():
        output = model(sample_input)

    for hook in hooks:
        hook.remove()

    return int(sum(macs_by_module.values())), tuple(int(dim) for dim in output.shape)


def count_profiler_flops(torch: Any, model: Any, sample_input: Any) -> Optional[int]:
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        return None

    model.eval()
    try:
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=False) as prof:
                model(sample_input)
    except Exception:
        return None

    return int(sum(getattr(event, "flops", 0) or 0 for event in prof.key_averages()))


def load_mask(mask_path: Path, img_size: Tuple[int, int]) -> np.ndarray:
    if mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        if mask.shape != img_size:
            raise RuntimeError(f"Mask shape {mask.shape} does not match img_size={img_size}.")
        return mask
    return np.ones(img_size, dtype=bool)


def build_model_profiles(mask: np.ndarray, args: argparse.Namespace) -> List[ModelProfile]:
    from baseline.models.SwinLSTM_B_model import SwinLSTMBaseline
    from baseline.models.convlstm_model import ConvLSTMBaseline
    from baseline.models.exprecast_model import ExPreCast
    from baseline.models.pfgnet_model import PFGNet
    from baseline.models.predrnn_model import PredRNN
    from baseline.models.predrnn_v2_model import PredRNNV2
    from baseline.models.seacast_model import SeaCast
    from baseline.models.timekan_model import TimeKAN
    from main_model.gpr_fishnet import GPRFishNet

    in_chans = args.in_chans
    hidden_dim = args.hidden_dim
    img_size = (args.height, args.width)
    seq_len = args.seq_len
    pred_len = args.pred_len

    return [
        ModelProfile(
            name="GPR-FishNet",
            role="ours",
            configuration=(
                f"STLSTM+ARP+MSSP+CAR; in={in_chans}; hidden={hidden_dim}; "
                f"STLSTM layers={args.num_layers}; seq_len={seq_len}"
            ),
            factory=lambda: GPRFishNet(
                in_chans=in_chans,
                hidden_dim=hidden_dim,
                img_size=img_size,
                num_layers=args.num_layers,
            ),
        ),
        ModelProfile(
            name="PredRNN",
            role="comparison",
            configuration=f"STLSTM backbone; in={in_chans}; hidden={hidden_dim}; layers=2; seq_len={seq_len}",
            factory=lambda: PredRNN(
                in_chans=in_chans,
                hidden_dim=hidden_dim,
                img_size=img_size,
                num_layers=2,
                pred_len=pred_len,
            ),
        ),
        ModelProfile(
            name="PredRNN-V2",
            role="comparison",
            configuration=f"STLSTM-V2; in={in_chans}; hidden={hidden_dim}; layers=2; seq_len={seq_len}",
            factory=lambda: PredRNNV2(
                in_chans=in_chans,
                hidden_dim=hidden_dim,
                img_size=img_size,
                num_layers=2,
                pred_len=pred_len,
            ),
        ),
        ModelProfile(
            name="ConvLSTM",
            role="comparison",
            configuration=f"ConvLSTM; in={in_chans}; hidden=[64,64]; kernel=3x3; seq_len={seq_len}",
            factory=lambda: ConvLSTMBaseline(
                in_channels=in_chans,
                hidden_channels=[64, 64],
                kernel_size=(3, 3),
                pred_len=pred_len,
                img_size=img_size,
            ),
        ),
        ModelProfile(
            name="PFGNet",
            role="comparison",
            configuration=f"SimVP-style PFG; in={in_chans}; hidden={hidden_dim}; PFG blocks=4; kernel=11",
            factory=lambda: PFGNet(
                in_chans=in_chans,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                img_size=img_size,
                num_layers=4,
                pred_len=pred_len,
            ),
        ),
        ModelProfile(
            name="ExPreCast",
            role="comparison",
            configuration=f"ConvGRU+local attention; in={in_chans}; hidden={hidden_dim}; GRU layers=2",
            factory=lambda: ExPreCast(
                in_chans=in_chans,
                hidden_dim=hidden_dim,
                img_size=img_size,
                num_layers=2,
                pred_len=pred_len,
            ),
        ),
        ModelProfile(
            name="TimeKAN",
            role="comparison",
            configuration=f"Spatial CNN + Temporal KAN; in={in_chans}; hidden={hidden_dim}; degree=3",
            factory=lambda: TimeKAN(
                in_chans=in_chans,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                img_size=img_size,
                pred_len=pred_len,
            ),
        ),
        ModelProfile(
            name="SwinLSTM",
            role="comparison",
            configuration="fish_fair profile: patch=2; embed=64; depths=(2,2); heads=(4,4); window=4",
            factory=lambda: SwinLSTMBaseline(
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
            ),
        ),
        ModelProfile(
            name="SeaCast",
            role="comparison",
            configuration=(
                f"ocean graph; nodes={int(mask.sum())}; hidden={hidden_dim}; "
                "temporal GRU layers=1; graph blocks=2; coarse_factor=4"
            ),
            factory=lambda: SeaCast.from_mask(
                mask,
                in_chans=in_chans,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                pred_len=pred_len,
                temporal_layers=1,
                graph_blocks=2,
                coarse_factor=4,
            ),
        ),
    ]


def profile_models(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    import torch
    import torch.nn as nn

    mask = load_mask(Path(args.mask_path), (args.height, args.width))
    profiles = build_model_profiles(mask, args)
    sample_input = torch.zeros(
        args.batch_size,
        args.seq_len,
        args.in_chans,
        args.height,
        args.width,
        dtype=torch.float32,
    )

    raw_rows: List[Dict[str, Any]] = []
    gpr_flops: Optional[int] = None
    gpr_model: Optional[Any] = None

    for profile in profiles:
        row: Dict[str, Any] = {
            "model": profile.name,
            "role": profile.role,
            "default_configuration": profile.configuration,
            "trainable_params": "",
            "params_m": "",
            "hook_macs_g": "",
            "profiler_flops_g": "",
            "relative_flops_to_gpr": "",
            "gpr_flops_over_model": "",
            "output_shape": "",
            "status": "ok",
        }
        try:
            model = profile.factory()
            params = count_trainable_params(model)
            hook_macs, output_shape = count_hook_macs(torch, nn, model, sample_input)
            profiler_flops = None if args.skip_profiler else count_profiler_flops(torch, model, sample_input)
            if profiler_flops is None:
                profiler_flops = hook_macs * 2
                row["status"] = "ok; profiler unavailable, FLOPs estimated as 2x hook MACs"

            row.update(
                {
                    "trainable_params": format_int(params),
                    "params_m": format_number(params / 1e6, 3),
                    "hook_macs_g": format_number(hook_macs / 1e9, 3),
                    "profiler_flops_g": format_number(profiler_flops / 1e9, 3),
                    "output_shape": str(output_shape),
                }
            )
            raw_rows.append({**row, "_profiler_flops": profiler_flops})
            if profile.name == "GPR-FishNet":
                gpr_flops = profiler_flops
                gpr_model = model
        except Exception as exc:
            row["status"] = f"skipped: {type(exc).__name__}: {exc}"
            raw_rows.append(row)

    for row in raw_rows:
        profiler_flops = row.get("_profiler_flops")
        if gpr_flops and profiler_flops:
            row["relative_flops_to_gpr"] = format_number(profiler_flops / gpr_flops, 3)
            row["gpr_flops_over_model"] = format_number(gpr_flops / profiler_flops, 2)
        row.pop("_profiler_flops", None)

    return raw_rows, gpr_model


def add_unavailable_model_rows(rows: List[Dict[str, Any]]) -> None:
    unavailable = [
        (
            "KSA-PredRNN",
            "comparison",
            "not profiled; source file is not present in baseline/models",
            "skipped: baseline/models/ksa_predrnn_model.py is unavailable in this workspace",
        ),
    ]
    for model, role, config, status in unavailable:
        rows.append(
            {
                "model": model,
                "role": role,
                "default_configuration": config,
                "trainable_params": "",
                "params_m": "",
                "hook_macs_g": "",
                "profiler_flops_g": "",
                "relative_flops_to_gpr": "",
                "gpr_flops_over_model": "",
                "output_shape": "",
                "status": status,
            }
        )


def build_notes(args: argparse.Namespace, compute_rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    included = [row["model"] for row in compute_rows if str(row.get("status", "")).startswith("ok")]
    return [
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("input_shape", f"[B={args.batch_size}, T={args.seq_len}, C={args.in_chans}, H={args.height}, W={args.width}]"),
        ("main_model", "GPR-FishNet uses 2 STLSTM layers with hidden_dim=64 by default."),
        ("arp_kernels", "CBAM spatial 7x7; residual predictor 3x3, 3x3, 1x1; shortcut 1x1; all dilation=1."),
        ("mssp_kernels", "local branch 3x3 dilation=1; surround branch 3x3 dilation=2; fuse 1x1."),
        ("compute_rule", "hook_macs_g counts Conv2d/ConvTranspose2d/Linear/GRU MACs during a real forward pass."),
        (
            "profiler_flops_g",
            "PyTorch CPU profiler FLOPs when available; otherwise 2x hook MACs. Nonlinearities/norm/indexing may be under-counted.",
        ),
        ("profiled_models", ", ".join(included)),
        ("mask_path", str(Path(args.mask_path).resolve())),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPR-FishNet architecture and model-compute tables.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for tables.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional explicit output prefix. If omitted, archive and latest outputs are written.",
    )
    parser.add_argument("--mask-path", type=str, default=str(DEFAULT_MASK_PATH), help="Ocean mask used for SeaCast graph size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for compute profiling.")
    parser.add_argument("--seq-len", type=int, default=12, help="Input sequence length.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction length.")
    parser.add_argument("--in-chans", type=int, default=8, help="Input channels.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GPR-FishNet hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="GPR-FishNet STLSTM layer count.")
    parser.add_argument("--height", type=int, default=64, help="Grid height.")
    parser.add_argument("--width", type=int, default=96, help="Grid width.")
    parser.add_argument("--skip-profiler", action="store_true", help="Skip PyTorch profiler and report FLOPs as 2x hook MACs.")
    parser.add_argument(
        "--no-unavailable-rows",
        action="store_true",
        help="Do not append rows for comparison models whose source files are absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = str(Path(args.output_dir))
    args.mask_path = str(Path(args.mask_path))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    mask_path = Path(args.mask_path)
    if not mask_path.is_absolute():
        mask_path = (PROJECT_ROOT / mask_path).resolve()
    args.mask_path = str(mask_path)

    archive_prefix, latest_prefix = resolve_output_prefixes(output_dir, args.output_prefix)
    archive_prefix.parent.mkdir(parents=True, exist_ok=True)

    architecture_rows = build_static_architecture_rows()
    compute_rows, gpr_model = profile_models(args)
    if gpr_model is not None:
        attach_gpr_module_params(architecture_rows, gpr_model)
    if not args.no_unavailable_rows:
        add_unavailable_model_rows(compute_rows)

    architecture_headers = [
        "stage",
        "component",
        "layers",
        "channels",
        "kernel_size",
        "dilation",
        "stride_padding",
        "output_shape",
        "trainable_params",
        "params_m",
        "notes",
    ]
    compute_headers = [
        "model",
        "role",
        "default_configuration",
        "trainable_params",
        "params_m",
        "hook_macs_g",
        "profiler_flops_g",
        "relative_flops_to_gpr",
        "gpr_flops_over_model",
        "output_shape",
        "status",
    ]
    notes = build_notes(args, compute_rows)
    notes_rows = [{"key": key, "value": value} for key, value in notes]

    paths = output_paths(archive_prefix)
    write_csv(paths["architecture_csv"], architecture_headers, architecture_rows)
    write_csv(paths["compute_csv"], compute_headers, compute_rows)
    write_markdown(
        paths["markdown"],
        architecture_headers,
        architecture_rows,
        compute_headers,
        compute_rows,
        notes,
    )
    xlsx_written = save_xlsx(
        paths["xlsx"],
        [
            {"title": "gpr_architecture", "headers": architecture_headers, "rows": architecture_rows},
            {"title": "compute_comparison", "headers": compute_headers, "rows": compute_rows},
            {"title": "notes", "headers": ["key", "value"], "rows": notes_rows},
        ],
    )
    copy_latest_outputs(archive_prefix, latest_prefix)

    print(f"Saved GPR architecture CSV to: {paths['architecture_csv']}")
    print(f"Saved compute comparison CSV to: {paths['compute_csv']}")
    print(f"Saved Markdown summary to: {paths['markdown']}")
    if xlsx_written:
        print(f"Saved XLSX workbook to: {paths['xlsx']}")
    else:
        print("Skipped XLSX workbook because the stdlib fallback writer could not be imported.")
    if latest_prefix != archive_prefix:
        latest_paths = output_paths(latest_prefix)
        print(f"Updated latest Markdown summary to: {latest_paths['markdown']}")


if __name__ == "__main__":
    main()
