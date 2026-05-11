from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from .schema import Sample, target_names


def limit_samples(samples: list[Sample], limit: int | None = None, per_subject_limit: int | None = None) -> list[Sample]:
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

    y_true = df["y_true"].astype(int).tolist()
    y_pred = df["y_pred"].astype(int).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names(labels),
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": report,
        "n_samples": len(records),
        "predictions_path": str(predictions_path),
        "config_path": str(config_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics

