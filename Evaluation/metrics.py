from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.schema import Sample, label_names_for_dataset, target_names


def label_distribution(samples: list[Sample]) -> dict[int, int]:
    return dict(sorted(Counter(sample.label for sample in samples).items()))


def limit_samples(
    samples: list[Sample],
    limit: int | None = None,
    per_subject_limit: int | None = None,
    balanced_per_label: int | None = None,
) -> list[Sample]:
    if balanced_per_label is not None:
        counts: dict[int, int] = {}
        balanced = []
        for sample in samples:
            count = counts.get(sample.label, 0)
            if count >= balanced_per_label:
                continue
            balanced.append(sample)
            counts[sample.label] = count + 1
        samples = balanced

    if per_subject_limit is not None:
        counts: dict[str, int] = {}
        limited = []
        for sample in samples:
            count = counts.get(sample.subject, 0)
            if count >= per_subject_limit:
                continue
            limited.append(sample)
            counts[sample.subject] = count + 1
        samples = limited

    if limit is not None:
        samples = samples[:limit]
    return samples


def summarize_and_save(
    records: list[dict],
    labels: list[int],
    output_dir: str | Path,
    run_name: str,
    config: dict,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if config.get("result_filename_style") == "compact":
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{run_name}_{timestamp}"

    df = pd.DataFrame(records)
    if config.get("result_filename_style") == "compact":
        predictions_path = output_path / f"{stem}.csv"
    else:
        predictions_path = output_path / f"{stem}_predictions.csv"
    metrics_path = output_path / f"{stem}_metrics.json"
    config_path = output_path / f"{stem}_config.json"

    df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    valid_df = df[df["valid"] == True].copy() if "valid" in df.columns else df.copy()
    invalid_count = int(len(df) - len(valid_df))
    dataset = _dataset_name_from_config(config)

    if len(valid_df) > 0:
        y_true = valid_df["y_true"].astype(int).tolist()
        y_pred = valid_df["y_pred"].astype(int).tolist()
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names(labels, dataset),
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    else:
        report = {}
        accuracy = None
        macro_f1 = None
        weighted_f1 = None
        conf_mat = []
    label_names = label_names_for_dataset(dataset)
    all_sample_metrics = _all_sample_metrics_invalid_as_wrong(df, labels)
    usage_summary = _usage_summary(df)
    cost_estimate = _cost_estimate(config, usage_summary)
    scaling_estimate = _scaling_estimate(config, usage_summary, cost_estimate, len(records))

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy_valid_only": accuracy,
        "macro_f1_valid_only": macro_f1,
        "weighted_f1_valid_only": weighted_f1,
        "accuracy_all_samples_invalid_as_wrong": all_sample_metrics["accuracy"],
        "macro_f1_all_samples_invalid_as_wrong": all_sample_metrics["macro_f1"],
        "weighted_f1_all_samples_invalid_as_wrong": all_sample_metrics["weighted_f1"],
        "confusion_matrix": conf_mat,
        "confusion_matrix_valid_only": conf_mat,
        "confusion_matrix_all_samples_invalid_as_wrong": all_sample_metrics["confusion_matrix"],
        "confusion_matrix_labels": labels,
        "confusion_matrix_label_names": [label_names.get(label, str(label)) for label in labels],
        "classification_report": report,
        "classification_report_valid_only": report,
        "n_samples": len(records),
        "valid_count": int(len(valid_df)),
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / len(df) if len(df) else 0.0,
        "usage_summary": usage_summary,
        "cost_estimate": cost_estimate,
        "scaling_estimate": scaling_estimate,
        "predictions_path": str(predictions_path),
        "config_path": str(config_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def _all_sample_metrics_invalid_as_wrong(df: pd.DataFrame, labels: list[int]) -> dict:
    if len(df) == 0:
        return {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "confusion_matrix": [],
        }

    y_true = df["y_true"].astype(int).tolist()
    y_pred: list[int] = []
    for row in df.itertuples(index=False):
        if bool(getattr(row, "valid", True)):
            y_pred.append(int(getattr(row, "y_pred")))
        else:
            y_pred.append(_wrong_label_for(int(getattr(row, "y_true")), labels))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _wrong_label_for(true_label: int, labels: list[int]) -> int:
    for label in labels:
        if label != true_label:
            return label
    return true_label


def _usage_summary(df: pd.DataFrame) -> dict:
    n_samples = len(df)
    total_llm_calls = _sum_numeric_column(df, "llm_call_count", default=0)
    total_prompt_tokens = _sum_nullable_numeric_column(df, "prompt_tokens")
    total_completion_tokens = _sum_nullable_numeric_column(df, "completion_tokens")
    total_tokens = _sum_nullable_numeric_column(df, "total_tokens")
    total_elapsed_time_sec = _sum_numeric_column(df, "elapsed_time_sec", default=0.0)
    token_usage_available_count = _sum_numeric_column(df, "llm_token_usage_available_count", default=0)
    token_usage_missing_count = _sum_numeric_column(df, "llm_token_usage_missing_count", default=0)
    samples_with_token_usage = _count_non_null(df, "total_tokens")

    return {
        "total_llm_calls": int(total_llm_calls),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_elapsed_time_sec": float(total_elapsed_time_sec),
        "average_prompt_tokens_per_sample": _safe_average(total_prompt_tokens, samples_with_token_usage),
        "average_completion_tokens_per_sample": _safe_average(total_completion_tokens, samples_with_token_usage),
        "average_total_tokens_per_sample": _safe_average(total_tokens, samples_with_token_usage),
        "average_elapsed_time_sec_per_sample": _safe_average(total_elapsed_time_sec, n_samples),
        "token_usage_available_count": int(token_usage_available_count),
        "token_usage_missing_count": int(token_usage_missing_count),
        "sample_token_usage_available_count": int(samples_with_token_usage),
        "sample_token_usage_missing_count": int(n_samples - samples_with_token_usage),
    }


def _cost_estimate(config: dict, usage_summary: dict) -> dict:
    cost_config = _cost_config(config)
    input_cost_per_1m = float(cost_config.get("input_cost_per_1m_tokens", 0.0) or 0.0)
    output_cost_per_1m = float(cost_config.get("output_cost_per_1m_tokens", 0.0) or 0.0)
    input_tokens = usage_summary.get("total_prompt_tokens")
    output_tokens = usage_summary.get("total_completion_tokens")
    estimated_input_cost = _token_cost(input_tokens, input_cost_per_1m)
    estimated_output_cost = _token_cost(output_tokens, output_cost_per_1m)
    estimated_total_cost = None
    if estimated_input_cost is not None and estimated_output_cost is not None:
        estimated_total_cost = estimated_input_cost + estimated_output_cost
    return {
        "model": (config.get("lm_client") or {}).get("model"),
        "input_cost_per_1m_tokens": input_cost_per_1m,
        "output_cost_per_1m_tokens": output_cost_per_1m,
        "estimated_input_cost": estimated_input_cost,
        "estimated_output_cost": estimated_output_cost,
        "estimated_total_cost": estimated_total_cost,
    }


def _scaling_estimate(config: dict, usage_summary: dict, cost_estimate: dict, current_samples: int) -> dict:
    scaling_config = _scaling_config(config)
    current_runs = scaling_config.get("current_runs", config.get("current_runs", 1))
    estimated_total_samples = scaling_config.get(
        "estimated_total_samples_for_full_experiment",
        config.get("estimated_total_samples_for_full_experiment"),
    )
    estimated_total_runs = scaling_config.get(
        "estimated_total_runs_for_full_experiment",
        config.get("estimated_total_runs_for_full_experiment"),
    )
    average_total_tokens = usage_summary.get("average_total_tokens_per_sample")
    estimated_total_tokens = None
    if estimated_total_samples is not None and average_total_tokens is not None:
        estimated_total_tokens = float(average_total_tokens) * int(estimated_total_samples)

    estimated_total_cost = None
    sample_count_for_cost = usage_summary.get("sample_token_usage_available_count")
    current_cost = cost_estimate.get("estimated_total_cost")
    if estimated_total_samples is not None and current_cost is not None and sample_count_for_cost:
        estimated_total_cost = (float(current_cost) / int(sample_count_for_cost)) * int(estimated_total_samples)

    return {
        "current_samples": int(current_samples),
        "current_runs": int(current_runs) if current_runs is not None else None,
        "estimated_total_samples_for_full_experiment": int(estimated_total_samples)
        if estimated_total_samples is not None
        else None,
        "estimated_total_runs_for_full_experiment": int(estimated_total_runs)
        if estimated_total_runs is not None
        else None,
        "estimated_total_tokens_for_full_experiment": estimated_total_tokens,
        "estimated_total_cost_for_full_experiment": estimated_total_cost,
    }


def _cost_config(config: dict) -> dict:
    value = config.get("cost_estimate", config.get("cost", {}))
    return value if isinstance(value, dict) else {}


def _scaling_config(config: dict) -> dict:
    value = config.get("scaling_estimate", config.get("scaling", {}))
    return value if isinstance(value, dict) else {}


def _token_cost(tokens: int | float | None, cost_per_1m_tokens: float) -> float | None:
    if tokens is None:
        return None
    return (float(tokens) / 1_000_000.0) * cost_per_1m_tokens


def _sum_numeric_column(df: pd.DataFrame, column: str, default: int | float) -> int | float:
    if column not in df.columns or len(df) == 0:
        return default
    return pd.to_numeric(df[column], errors="coerce").fillna(0).sum()


def _sum_nullable_numeric_column(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns or len(df) == 0:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(values) == 0:
        return None
    return int(values.sum())


def _count_non_null(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns or len(df) == 0:
        return 0
    return int(pd.to_numeric(df[column], errors="coerce").notna().sum())


def _safe_average(total: int | float | None, count: int) -> float | None:
    if total is None or count <= 0:
        return None
    return float(total) / count


def _dataset_name_from_config(config: dict) -> str | None:
    dataset = config.get("dataset")
    if isinstance(dataset, dict):
        return dataset.get("name")
    if isinstance(dataset, str):
        return dataset
    input_config = config.get("input")
    if isinstance(input_config, dict):
        return input_config.get("dataset")
    return None
