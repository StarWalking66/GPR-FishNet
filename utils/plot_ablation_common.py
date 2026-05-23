from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_ABLATION_OUT_DIR = PROJECT_ROOT / "model_outcomes" / "ablation"


METRIC_DIRECTIONS: Dict[str, str] = {
    "MAE": "lower",
    "MSE": "lower",
    "RMSE": "lower",
    "R2": "higher",
    "SSIM": "higher",
    "CSI": "higher",
    "F1": "higher",
}


SPLIT_LABELS: Dict[str, str] = {
    "test_one_step": "One-step",
    "rollout_2024": "Rollout-2024",
}


def set_publication_style() -> None:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["axes.linewidth"] = 1.0
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.major.width"] = 1.0
    rcParams["ytick.major.width"] = 1.0
    rcParams["axes.labelweight"] = "bold"
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return text
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." in text or "e" in lower:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    continue
                parsed[key] = _parse_scalar(value)
            rows.append(parsed)
    return rows


def load_significance_map(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected list-based significance JSON, got: {type(payload).__name__}")

    sig_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        candidate = str(row.get("candidate_experiment", ""))
        split = str(row.get("split", ""))
        metric = str(row.get("metric", ""))
        if not candidate or not split or not metric:
            continue
        sig_map[(candidate, split, metric)] = row
    return sig_map


def split_metric_key(metric_key: str) -> Tuple[str, str]:
    if metric_key.startswith("test_one_step_"):
        return "test_one_step", metric_key.replace("test_one_step_", "", 1)
    if metric_key.startswith("rollout_2024_"):
        return "rollout_2024", metric_key.replace("rollout_2024_", "", 1)
    return "unknown", metric_key


def metric_direction(metric_name: str) -> str:
    return METRIC_DIRECTIONS.get(metric_name, "higher")


def signed_gain(candidate_value: float, reference_value: float, metric_name: str) -> float:
    direction = metric_direction(metric_name)
    if direction == "lower":
        return reference_value - candidate_value
    return candidate_value - reference_value


def get_metric_value(row: Mapping[str, Any], split: str, metric: str, default: float = np.nan) -> float:
    key = f"{split}_{metric}"
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_metric_std(row: Mapping[str, Any], split: str, metric: str, default: float = np.nan) -> float:
    key = f"{split}_{metric}_std"
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def canonical_metric_list(metrics_raw: str) -> List[str]:
    return [item.strip() for item in metrics_raw.split(",") if item.strip()]


def pretty_metric_name(metric_name: str) -> str:
    metric_name = metric_name.upper()
    if metric_name == "R2":
        return "R2"
    return metric_name


def pretty_split_name(split_name: str) -> str:
    return SPLIT_LABELS.get(split_name, split_name)


def significance_marker_and_color(sig_row: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    if not sig_row:
        return "", "#374151"
    is_sig = bool(sig_row.get("is_significant", False))
    if not is_sig:
        return "", "#374151"
    decision = str(sig_row.get("decision", ""))
    if decision == "better":
        return "*", "#15803d"
    if decision == "worse":
        return "*", "#b91c1c"
    return "*", "#1d4ed8"


def save_figure(fig: plt.Figure, output_prefix: Path, dpi: int = 300) -> Tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def variance_sum_std(std_a: float, std_b: float) -> float:
    if not np.isfinite(std_a) and not np.isfinite(std_b):
        return float("nan")
    a = 0.0 if not np.isfinite(std_a) else float(std_a)
    b = 0.0 if not np.isfinite(std_b) else float(std_b)
    return float(np.sqrt(a * a + b * b))


def ensure_rows_by_key(rows: List[Dict[str, Any]], key_name: str) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(key_name, "")).strip()
        if not key:
            continue
        output[key] = row
    return output


def convert_sig_payload_to_rows(sig_map: Mapping[Tuple[str, str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(sig_map.values())
