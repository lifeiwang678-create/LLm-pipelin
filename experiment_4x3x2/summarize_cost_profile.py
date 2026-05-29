from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-sample runtime, token usage, cost, and optional GPU telemetry."
    )
    parser.add_argument("--metrics-json", required=True, help="Path to a *_metrics.json file.")
    parser.add_argument(
        "--gpu-csv",
        help="Optional GPU monitor CSV produced by nvidia-smi --query-gpu ... -l.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. Defaults to printing only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_json)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    usage = metrics.get("usage_summary", {}) or {}
    cost = metrics.get("cost_estimate", {}) or {}
    scaling = metrics.get("scaling_estimate", {}) or {}

    summary: dict[str, Any] = {
        "metrics_json": str(metrics_path),
        "n_samples": metrics.get("n_samples"),
        "valid_count": metrics.get("valid_count"),
        "invalid_count": metrics.get("invalid_count"),
        "accuracy_all_samples_invalid_as_wrong": metrics.get("accuracy_all_samples_invalid_as_wrong"),
        "single_sample": {
            "average_elapsed_time_sec": usage.get("average_elapsed_time_sec_per_sample"),
            "average_prompt_tokens": usage.get("average_prompt_tokens_per_sample"),
            "average_completion_tokens": usage.get("average_completion_tokens_per_sample"),
            "average_total_tokens": usage.get("average_total_tokens_per_sample"),
        },
        "current_run_totals": {
            "llm_calls": usage.get("total_llm_calls"),
            "elapsed_time_sec": usage.get("total_elapsed_time_sec"),
            "prompt_tokens": usage.get("total_prompt_tokens"),
            "completion_tokens": usage.get("total_completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_total_cost": cost.get("estimated_total_cost"),
        },
        "full_experiment_estimate": scaling,
    }

    if args.gpu_csv:
        summary["gpu"] = summarize_gpu_csv(Path(args.gpu_csv))

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


def summarize_gpu_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"GPU CSV not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                rows.append(row)

    gpu_utils = []
    mem_utils = []
    mem_used = []
    mem_total = []
    power = []
    for row in rows:
        gpu_utils.append(_number_from_row(row, "utilization.gpu"))
        mem_utils.append(_number_from_row(row, "utilization.memory"))
        mem_used.append(_number_from_row(row, "memory.used"))
        mem_total.append(_number_from_row(row, "memory.total"))
        power.append(_number_from_row(row, "power.draw"))

    return {
        "gpu_csv": str(path),
        "sample_count": len(rows),
        "average_gpu_utilization_pct": _avg(gpu_utils),
        "max_gpu_utilization_pct": _max(gpu_utils),
        "average_memory_utilization_pct": _avg(mem_utils),
        "max_memory_utilization_pct": _max(mem_utils),
        "average_memory_used_mib": _avg(mem_used),
        "max_memory_used_mib": _max(mem_used),
        "memory_total_mib": _max(mem_total),
        "average_power_w": _avg(power),
        "max_power_w": _max(power),
    }


def _number_from_row(row: dict[str, str], key_prefix: str) -> float | None:
    for key, value in row.items():
        if key.strip().startswith(key_prefix):
            match = re.search(r"-?\d+(?:\.\d+)?", str(value))
            if match:
                return float(match.group(0))
    return None


def _avg(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _max(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return max(clean)


if __name__ == "__main__":
    main()
