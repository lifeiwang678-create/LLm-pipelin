from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from Output import build_output_handler
from core.runner import run_experiment


DIRECT_PROBES = [
    {
        "name": "strict_label_only",
        "output_type": "label_only",
        "prompt": """ /no_think
You are testing a wearable-sensor classifier JSON interface.
Return STRICT JSON only. Do not include markdown, commentary, or thinking text.
Choose one valid label.

Output format:
{
  "predicted_state": 0 or 1
}""",
    },
    {
        "name": "strict_label_explanation",
        "output_type": "label_explanation",
        "prompt": """ /no_think
You are testing a wearable-sensor classifier JSON interface.
Return STRICT JSON only. Do not include markdown, commentary, or thinking text.
Choose one valid label and include a short explanation.

Output format:
{
  "predicted_state": 0 or 1,
  "explanation": "one short sentence"
}""",
    },
]

PIPELINE_CASES = [
    ("WESAD", "feature_description", "direct", "label_only"),
    ("WESAD", "extra_knowledge", "multi_agent", "label_explanation"),
    ("HHAR", "raw_data", "direct", "label_explanation"),
    ("DREAMT", "feature_description", "few_shot", "label_only"),
    ("DREAMT", "encoded_time_series", "direct", "label_explanation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3 JSON stability against this pipeline.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--output-dir", default="Results/Qwen3Smoke")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--direct-repetitions", type=int, default=3)
    parser.add_argument("--pipeline-balanced-per-label", type=int, default=1)
    parser.add_argument("--pipeline-concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--chat-template-kwargs-json",
        default='{"enable_thinking": false}',
        help="Extra vLLM chat_template_kwargs used for direct probes.",
    )
    parser.add_argument("--skip-direct-probes", action="store_true")
    parser.add_argument("--skip-pipeline-probes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json) if args.report_json else output_dir / "qwen3_smoke_report.json"
    chat_template_kwargs = _parse_json_object(args.chat_template_kwargs_json)

    started = time.perf_counter()
    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "api_url": args.api_url,
        "model": args.model,
        "direct_probe_repetitions": int(args.direct_repetitions),
        "pipeline_balanced_per_label": int(args.pipeline_balanced_per_label),
        "chat_template_kwargs": chat_template_kwargs,
        "direct_probes": [],
        "pipeline_probes": [],
        "ok": False,
    }

    failures: list[str] = []
    if not args.skip_direct_probes:
        direct_results = run_direct_probes(args, chat_template_kwargs)
        report["direct_probes"] = direct_results
        failures.extend(
            item["name"]
            for item in direct_results
            if not item.get("ok")
        )

    if not args.skip_pipeline_probes:
        pipeline_results = run_pipeline_probes(args, output_dir)
        report["pipeline_probes"] = pipeline_results
        failures.extend(
            item["name"]
            for item in pipeline_results
            if not item.get("ok")
        )

    report["elapsed_sec"] = round(time.perf_counter() - started, 3)
    report["failures"] = failures
    report["ok"] = not failures
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Qwen3 smoke report: {report_path}")
    if failures:
        print("Qwen3 JSON smoke FAILED:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Qwen3 JSON smoke PASSED.")


def run_direct_probes(args: argparse.Namespace, chat_template_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for probe in DIRECT_PROBES:
        output_type = probe["output_type"]
        handler = build_output_handler({"type": output_type}, [0, 1])
        for rep in range(max(1, int(args.direct_repetitions))):
            name = f"direct:{probe['name']}:rep{rep + 1}"
            raw = ""
            parsed: dict[str, Any] = {}
            error = ""
            try:
                raw = _chat_completion(
                    args=args,
                    prompt=probe["prompt"],
                    chat_template_kwargs=chat_template_kwargs,
                )
                parsed = handler.parse(raw)
            except Exception as exc:
                error = str(exc)

            contains_thinking = "<think" in raw.lower() or "</think" in raw.lower()
            ok = bool(parsed.get("valid")) and not contains_thinking and not error
            results.append(
                {
                    "name": name,
                    "output_type": output_type,
                    "ok": ok,
                    "valid": bool(parsed.get("valid")),
                    "contains_thinking": contains_thinking,
                    "parse_error": parsed.get("parse_error", error),
                    "raw_response": raw,
                }
            )
            print(f"{name}: {'OK' if ok else 'FAILED'}")
    return results


def _chat_completion(
    *,
    args: argparse.Namespace,
    prompt: str,
    chat_template_kwargs: dict[str, Any],
) -> str:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
    }
    if args.top_p is not None:
        payload["top_p"] = float(args.top_p)
    if args.top_k is not None:
        payload["top_k"] = int(args.top_k)
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs

    response = requests.post(
        f"{args.api_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=int(args.timeout),
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def run_pipeline_probes(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    chat_template_kwargs = _parse_json_object(args.chat_template_kwargs_json)
    for dataset, input_name, lm_name, output_type in PIPELINE_CASES:
        name = f"pipeline:{dataset}:{input_name}:{lm_name}:{output_type}"
        try:
            metrics = run_experiment(
                {
                    "run_name": f"qwen3_smoke_{dataset}_{input_name}_{lm_name}_{output_type}",
                    "result_filename_style": "compact",
                    "labels": [0, 1],
                    "output_dir": str(output_dir),
                    "dataset": {"name": dataset},
                    "data": {
                        "use_input_cache": True,
                        "input_cache_dir": "Processed",
                    },
                    "input": {"type": input_name, "dataset": dataset},
                    "lm_usage": {
                        "type": lm_name,
                        "n_per_class": 1,
                        "random_state": 42,
                        "example_max_chars": _example_max_chars(input_name),
                        "intermediate_max_tokens": 128,
                    },
                    "output": {"type": output_type},
                    "lm_client": {
                        "api_url": args.api_url,
                        "api_key": args.api_key,
                        "model": args.model,
                        "temperature": float(args.temperature),
                        "max_tokens": int(args.max_tokens),
                        "timeout": int(args.timeout),
                        "system_message": "/no_think Return only the requested JSON object.",
                        "chat_template_kwargs": chat_template_kwargs,
                    },
                    "evaluation": {
                        "balanced_per_label": int(args.pipeline_balanced_per_label),
                        "concurrency": int(args.pipeline_concurrency),
                        "log_every": 1,
                    },
                },
                dataset_name=dataset,
            )
            invalid_count = int(metrics.get("invalid_count", 0) or 0)
            n_samples = int(metrics.get("n_samples", 0) or 0)
            ok = invalid_count == 0 and n_samples > 0
            results.append(
                {
                    "name": name,
                    "ok": ok,
                    "n_samples": n_samples,
                    "invalid_count": invalid_count,
                    "accuracy_valid_only": metrics.get("accuracy_valid_only"),
                    "predictions_path": metrics.get("predictions_path"),
                    "metrics_path": metrics.get("metrics_path"),
                }
            )
        except Exception as exc:
            results.append({"name": name, "ok": False, "error": str(exc)})
        print(f"{name}: {'OK' if results[-1].get('ok') else 'FAILED'}")
    return results


def _example_max_chars(input_name: str) -> int | None:
    if input_name == "raw_data":
        return 500
    if input_name == "encoded_time_series":
        return 300
    return 800


def _parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--chat-template-kwargs-json must decode to a JSON object.")
    return obj


if __name__ == "__main__":
    main()
