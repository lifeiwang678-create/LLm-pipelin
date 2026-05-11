from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from .schema import Sample, target_names


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{run_name}_{timestamp}"

    df = pd.DataFrame(records)
    predictions_path = output_path / f"{stem}_predictions.csv"
    metrics_path = output_path / f"{stem}_metrics.json"
    config_path = output_path / f"{stem}_config.json"

    df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    valid_df = df[df["valid"] == True].copy() if "valid" in df.columns else df.copy()
    invalid_count = int(len(df) - len(valid_df))

    if len(valid_df) > 0:
        y_true = valid_df["y_true"].astype(int).tolist()
        y_pred = valid_df["y_pred"].astype(int).tolist()
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names(labels),
            digits=4,
            output_dict=True,
            zero_division=0,
        )
        accuracy = accuracy_score(y_true, y_pred)
    else:
        report = {}
        accuracy = None

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
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
