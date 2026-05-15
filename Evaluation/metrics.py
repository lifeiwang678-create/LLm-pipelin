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
