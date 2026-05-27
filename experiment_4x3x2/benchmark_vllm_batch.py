from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible vLLM batching.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--model", default="qwen2.5-7b-instruct")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--requests-per-level", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--prompt-chars", type=int, default=2500)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_url = args.api_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    jsonl_path = Path(args.output_jsonl)
    csv_path = Path(args.output_csv)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    warmup_count = max(0, int(args.warmup_requests))
    for idx in range(warmup_count):
        _send_request(args, api_url, headers, request_id=f"warmup-{idx}")

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for concurrency in args.concurrency:
            total_requests = max(int(args.requests_per_level), int(concurrency))
            print(
                f"Benchmarking concurrency={concurrency}, "
                f"requests={total_requests}, prompt_chars={args.prompt_chars}, max_tokens={args.max_tokens}",
                flush=True,
            )
            started = time.perf_counter()
            results = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _send_request,
                        args,
                        api_url,
                        headers,
                        f"c{concurrency}-r{idx}",
                    )
                    for idx in range(total_requests)
                ]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    jsonl_file.flush()

            elapsed = time.perf_counter() - started
            summary = _summarize(concurrency, total_requests, elapsed, results)
            rows.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    _write_csv(csv_path, rows)
    print(f"JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")


def _send_request(
    args: argparse.Namespace,
    api_url: str,
    headers: dict[str, str],
    request_id: str,
) -> dict[str, Any]:
    prompt = _make_prompt(args.prompt_chars, request_id)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(args.max_tokens),
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - started
        text = response.text
        if response.status_code >= 400:
            return {
                "request_id": request_id,
                "ok": False,
                "status_code": response.status_code,
                "latency_sec": elapsed,
                "error": _truncate(text),
            }
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_json.get("usage", {}) or {}
        return {
            "request_id": request_id,
            "ok": True,
            "status_code": response.status_code,
            "latency_sec": elapsed,
            "completion_chars": len(str(content)),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "request_id": request_id,
            "ok": False,
            "status_code": None,
            "latency_sec": elapsed,
            "error": _truncate(str(exc)),
        }


def _make_prompt(prompt_chars: int, request_id: str) -> str:
    base = (
        "You are classifying one wearable-sensor time-series window. "
        "Return STRICT JSON only with the schema {\"predicted_state\": 0 or 1}. "
        f"Request id: {request_id}. Sensor summary: "
    )
    pattern = (
        "chest_ecg mean stable; eda moderate; acc small movement; "
        "resp regular; wrist_bvp steady; temporal segment shows mild fluctuation. "
    )
    target = max(int(prompt_chars), len(base))
    repeated = pattern * ((target - len(base)) // len(pattern) + 1)
    return (base + repeated)[:target]


def _summarize(
    concurrency: int,
    total_requests: int,
    elapsed_sec: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    ok_results = [item for item in results if item.get("ok")]
    failed_results = [item for item in results if not item.get("ok")]
    latencies = [float(item["latency_sec"]) for item in ok_results]
    total_tokens = _sum_numeric(ok_results, "total_tokens")
    completion_tokens = _sum_numeric(ok_results, "completion_tokens")
    return {
        "concurrency": int(concurrency),
        "requests": int(total_requests),
        "ok": len(ok_results),
        "failed": len(failed_results),
        "elapsed_sec": round(elapsed_sec, 6),
        "requests_per_sec": _round(len(ok_results) / elapsed_sec if elapsed_sec else None),
        "total_tokens": total_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": _round(total_tokens / elapsed_sec if total_tokens is not None and elapsed_sec else None),
        "completion_tokens_per_sec": _round(
            completion_tokens / elapsed_sec if completion_tokens is not None and elapsed_sec else None
        ),
        "latency_p50_sec": _percentile(latencies, 50),
        "latency_p90_sec": _percentile(latencies, 90),
        "latency_p99_sec": _percentile(latencies, 99),
        "first_error": failed_results[0].get("error", "") if failed_results else "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 6)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(value, 6)


def _sum_numeric(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return int(sum(int(value) for value in values))


def _round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _truncate(text: str, limit: int = 500) -> str:
    text = str(text)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


if __name__ == "__main__":
    main()
