from __future__ import annotations

import csv
import itertools
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


METRIC_DIRECTIONS: Dict[str, str] = {
    "MAE": "lower",
    "MSE": "lower",
    "RMSE": "lower",
    "R2": "higher",
    "SSIM": "higher",
    "CSI": "higher",
    "F1": "higher",
}


def _read_summary_runs_csv(experiment_dir: str, summary_runs_filename: str = "summary_runs.csv") -> List[Dict[str, Any]]:
    path = os.path.join(experiment_dir, summary_runs_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing summary_runs.csv: {path}")

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key in {"seed", "best_epoch"}:
                    parsed[key] = int(float(value))
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def _metric_columns(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    metric_keys: List[str] = []
    for key in rows[0].keys():
        if key.startswith("test_one_step_") or key.startswith("rollout_2024_"):
            metric_keys.append(key)
    return metric_keys


def _seed_map(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    mapped: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        mapped[seed] = row
    return mapped


def _paired_metric_values(
    reference_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    metric_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    ref_map = _seed_map(reference_rows)
    cand_map = _seed_map(candidate_rows)
    shared_seeds = sorted(set(ref_map.keys()).intersection(cand_map.keys()))
    if not shared_seeds:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), []

    ref_values: List[float] = []
    cand_values: List[float] = []
    used_seeds: List[int] = []
    for seed in shared_seeds:
        if metric_key not in ref_map[seed] or metric_key not in cand_map[seed]:
            continue
        ref_values.append(float(ref_map[seed][metric_key]))
        cand_values.append(float(cand_map[seed][metric_key]))
        used_seeds.append(seed)
    return np.asarray(ref_values, dtype=np.float64), np.asarray(cand_values, dtype=np.float64), used_seeds


def _bootstrap_mean_diff_ci(
    diffs: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_seed: int = 2026,
) -> Tuple[float, float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(np.mean(diffs))
    if diffs.size == 1:
        return observed, observed, observed

    rng = np.random.default_rng(random_seed)
    n = diffs.size
    sample_indices = rng.integers(0, n, size=(n_bootstrap, n), endpoint=False)
    sample_means = diffs[sample_indices].mean(axis=1)
    low, high = np.quantile(sample_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return observed, float(low), float(high)


def _paired_sign_flip_pvalue(
    diffs: np.ndarray,
    alternative: str = "two-sided",
    n_permutation: int = 20000,
    random_seed: int = 2026,
) -> float:
    if diffs.size == 0:
        return float("nan")
    observed = float(np.mean(diffs))
    n = diffs.size

    if n <= 15:
        all_signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=np.float64)
        null_means = (all_signs * diffs[None, :]).mean(axis=1)
    else:
        rng = np.random.default_rng(random_seed)
        signs = rng.choice([-1.0, 1.0], size=(n_permutation, n))
        null_means = (signs * diffs[None, :]).mean(axis=1)

    eps = 1e-12
    if alternative == "greater":
        return float(np.mean(null_means >= observed - eps))
    if alternative == "less":
        return float(np.mean(null_means <= observed + eps))
    return float(np.mean(np.abs(null_means) >= abs(observed) - eps))


def _parse_split_and_metric(metric_key: str) -> Tuple[str, str]:
    if metric_key.startswith("test_one_step_"):
        return "test_one_step", metric_key.replace("test_one_step_", "", 1)
    if metric_key.startswith("rollout_2024_"):
        return "rollout_2024", metric_key.replace("rollout_2024_", "", 1)
    return "unknown", metric_key


def _decision_label(direction: str, mean_diff: float, p_value: float, alpha: float) -> str:
    if not np.isfinite(mean_diff) or not np.isfinite(p_value):
        return "insufficient_data"
    if p_value >= alpha:
        return "not_significant"
    if direction == "lower":
        if mean_diff < 0:
            return "better"
        if mean_diff > 0:
            return "worse"
        return "equal"
    if direction == "higher":
        if mean_diff > 0:
            return "better"
        if mean_diff < 0:
            return "worse"
        return "equal"
    return "unknown_direction"


def _save_rows_as_csv_and_json(rows: List[Dict[str, Any]], csv_path: str, json_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)

    if not rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_significance_analysis(
    group_save_dir: str,
    reference_experiment: str,
    candidate_experiments: Sequence[str],
    output_prefix: str,
    summary_runs_filename: str = "summary_runs.csv",
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    n_permutation: int = 20000,
    random_seed: int = 2026,
) -> Optional[Dict[str, Any]]:
    reference_dir = os.path.join(group_save_dir, reference_experiment)
    reference_summary = os.path.join(reference_dir, summary_runs_filename)
    if not os.path.exists(reference_summary):
        return None

    reference_rows = _read_summary_runs_csv(reference_dir, summary_runs_filename=summary_runs_filename)
    if not reference_rows:
        return None

    rows: List[Dict[str, Any]] = []
    ref_metric_columns = set(_metric_columns(reference_rows))

    for experiment in candidate_experiments:
        if experiment == reference_experiment:
            continue
        candidate_dir = os.path.join(group_save_dir, experiment)
        candidate_summary = os.path.join(candidate_dir, summary_runs_filename)
        if not os.path.exists(candidate_summary):
            continue

        candidate_rows = _read_summary_runs_csv(candidate_dir, summary_runs_filename=summary_runs_filename)
        if not candidate_rows:
            continue

        metric_columns = sorted(ref_metric_columns.intersection(_metric_columns(candidate_rows)))
        for metric_key in metric_columns:
            ref_values, cand_values, shared_seeds = _paired_metric_values(reference_rows, candidate_rows, metric_key)
            if ref_values.size == 0 or cand_values.size == 0:
                continue

            diffs = cand_values - ref_values
            mean_diff, ci_low, ci_high = _bootstrap_mean_diff_ci(
                diffs=diffs,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                random_seed=random_seed,
            )
            p_value = _paired_sign_flip_pvalue(
                diffs=diffs,
                alternative="two-sided",
                n_permutation=n_permutation,
                random_seed=random_seed + len(rows),
            )

            split_name, metric_name = _parse_split_and_metric(metric_key)
            direction = METRIC_DIRECTIONS.get(metric_name, "unknown")
            decision = _decision_label(direction, mean_diff, p_value, alpha)

            rows.append(
                {
                    "reference_experiment": reference_experiment,
                    "candidate_experiment": experiment,
                    "split": split_name,
                    "metric": metric_name,
                    "metric_key": metric_key,
                    "better_direction": direction,
                    "n_paired_seeds": int(len(shared_seeds)),
                    "shared_seeds": [int(s) for s in shared_seeds],
                    "reference_mean": float(np.mean(ref_values)),
                    "candidate_mean": float(np.mean(cand_values)),
                    "mean_diff_candidate_minus_reference": float(mean_diff),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "p_value_two_sided": float(p_value),
                    "alpha": float(alpha),
                    "is_significant": bool(np.isfinite(p_value) and p_value < alpha),
                    "decision": decision,
                }
            )

    csv_path = os.path.join(group_save_dir, f"{output_prefix}.csv")
    json_path = os.path.join(group_save_dir, f"{output_prefix}.json")
    _save_rows_as_csv_and_json(rows, csv_path, json_path)
    return {"rows": rows, "csv_path": csv_path, "json_path": json_path, "reference_experiment": reference_experiment}
