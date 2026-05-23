import argparse
import os
import subprocess
import sys
import time
from typing import Iterable, List, Sequence


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "model_outcomes", "run_logs", "baseline_train")

PREFERRED_ORDER = [
    "train_convlstm_baseline.py",
    "train_exprecast_baseline.py",
    "train_pfgnet_baseline.py",
    "train_predrnn_baseline.py",
    "train_predrnn_v2_baseline.py",
    "train_seacast_baseline.py",
    "train_swinlstm_baseline.py",
    "train_timekan_baseline.py",
]


def discover_train_scripts() -> List[str]:
    discovered = []
    for name in os.listdir(CURRENT_DIR):
        if not name.endswith(".py"):
            continue
        if name in {"__init__.py", os.path.basename(__file__)}:
            continue
        if name.startswith("train_") and name.endswith("_baseline.py"):
            discovered.append(name)

    preferred_rank = {name: idx for idx, name in enumerate(PREFERRED_ORDER)}
    return sorted(discovered, key=lambda name: (preferred_rank.get(name, len(PREFERRED_ORDER)), name))


def normalize_script_name(name: str) -> str:
    raw = name.strip()
    if not raw:
        return raw
    if raw.endswith(".py"):
        return raw
    if raw.startswith("train_"):
        return f"{raw}.py"
    return f"train_{raw}.py"


def parse_name_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [normalize_script_name(item) for item in raw.split(",") if item.strip()]


def select_scripts(available: Sequence[str], include: Sequence[str], exclude: Sequence[str]) -> List[str]:
    available_set = set(available)

    unknown_include = [name for name in include if name not in available_set]
    if unknown_include:
        raise ValueError(f"Unknown training script(s): {', '.join(unknown_include)}")

    unknown_exclude = [name for name in exclude if name not in available_set]
    if unknown_exclude:
        raise ValueError(f"Unknown excluded script(s): {', '.join(unknown_exclude)}")

    selected = list(include) if include else list(available)
    excluded_set = set(exclude)
    return [name for name in selected if name not in excluded_set]


def ensure_log_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def stream_command(command: Sequence[str], workdir: str, log_path: str) -> int:
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatically run baseline training scripts in sequence.",
    )
    parser.add_argument(
        "--scripts",
        type=str,
        default="",
        help=(
            "Comma-separated training scripts to run. "
            "Supports full filenames or short names like 'predrnn_baseline'. "
            "Default: auto-run all discovered baseline train scripts."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated training scripts to skip.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running the remaining scripts even if one training script fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the execution plan without launching training.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch the training scripts.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_LOG_DIR,
        help="Directory used to save per-script console logs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    available = discover_train_scripts()
    include = parse_name_list(args.scripts)
    exclude = parse_name_list(args.exclude)
    selected = select_scripts(available, include, exclude)

    if not selected:
        raise SystemExit("No baseline training scripts selected.")

    ensure_log_dir(args.log_dir)

    print("Discovered baseline training scripts:")
    for name in available:
        marker = "[run]" if name in selected else "[skip]"
        print(f"  {marker} {name}")

    if args.dry_run:
        print("\nDry run only. No training scripts were launched.")
        return

    total_start = time.time()
    results = []

    for index, script_name in enumerate(selected, start=1):
        script_path = os.path.join(CURRENT_DIR, script_name)
        log_path = os.path.join(args.log_dir, script_name.replace(".py", ".log"))
        command = [args.python, script_path]

        print("\n" + "=" * 80)
        print(f"[STEP {index}/{len(selected)}] Running {script_name}")
        print(f"Log file: {log_path}")
        print("=" * 80)

        start_time = time.time()
        return_code = stream_command(command, PROJECT_ROOT, log_path)
        duration = time.time() - start_time

        status = "SUCCESS" if return_code == 0 else "FAILED"
        print(f"[{status}] {script_name} finished in {format_seconds(duration)}")

        results.append(
            {
                "script": script_name,
                "return_code": return_code,
                "duration": duration,
                "log_path": log_path,
            }
        )

        if return_code != 0 and not args.continue_on_error:
            break

    total_duration = time.time() - total_start
    success_rows = [row for row in results if row["return_code"] == 0]
    failed_rows = [row for row in results if row["return_code"] != 0]

    print("\n" + "=" * 80)
    print("Baseline training summary")
    print("=" * 80)
    print(f"Total elapsed time: {format_seconds(total_duration)}")
    print(f"Successful runs: {len(success_rows)}")
    print(f"Failed runs: {len(failed_rows)}")

    for row in results:
        status = "OK" if row["return_code"] == 0 else "ERR"
        print(
            f"  [{status}] {row['script']} | {format_seconds(row['duration'])} | {row['log_path']}"
        )

    if failed_rows:
        failed_names = ", ".join(row["script"] for row in failed_rows)
        raise SystemExit(f"Some baseline training scripts failed: {failed_names}")


if __name__ == "__main__":
    main()
