from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_pipeline.evaluation import label_distribution, limit_samples, summarize_and_save
from experiment_pipeline.inputs import build_input_provider
from experiment_pipeline.lm_client import LMStudioClient
from experiment_pipeline.lm_usage import build_lm_usage
from experiment_pipeline.outputs import build_output_handler


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(config_path)
    return config


def choose_subject_split(config: dict) -> tuple[list[str] | None, list[str] | None]:
    data_cfg = config.get("data", {})
    train_subjects = data_cfg.get("train_subjects")
    test_subjects = data_cfg.get("test_subjects") or data_cfg.get("subjects")
    return train_subjects, test_subjects


def validate_fewshot_split(train_subjects: list[str] | None, test_subjects: list[str] | None) -> None:
    if not train_subjects or not test_subjects:
        raise ValueError("few_shot requires explicit data.train_subjects and data.test_subjects.")

    overlap = sorted(set(train_subjects) & set(test_subjects))
    if overlap:
        raise ValueError(f"few_shot leakage: test subjects also appear in examples: {overlap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a modular WESAD LLM experiment.")
    parser.add_argument("--config", default="configs/example_experiment.json", help="Path to JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = [int(label) for label in config.get("labels", [1, 2])]
    input_provider = build_input_provider(config.get("input", {}))
    output_handler = build_output_handler(config.get("output", {}), labels)

    train_subjects, test_subjects = choose_subject_split(config)
    usage_type = str(config.get("lm_usage", {}).get("type", "direct")).lower()

    if usage_type in {"fewshot", "few_shot", "few-shot"}:
        validate_fewshot_split(train_subjects, test_subjects)
        train_samples = input_provider.load(train_subjects, labels)
        if not train_samples:
            raise RuntimeError("Few-shot mode needs non-empty training samples.")
        print(f"Few-shot train label distribution: {label_distribution(train_samples)}")
        eval_samples = input_provider.load(test_subjects, labels)
        example_subjects = {sample.subject for sample in train_samples}
        leaked_subjects = sorted(example_subjects & set(test_subjects))
        if leaked_subjects:
            raise ValueError(f"few_shot leakage after loading samples: {leaked_subjects}")
    else:
        train_samples = []
        eval_samples = input_provider.load(test_subjects, labels)

    eval_cfg = config.get("evaluation", {})
    print(f"Label distribution before sampling: {label_distribution(eval_samples)}")
    eval_samples = limit_samples(
        eval_samples,
        limit=eval_cfg.get("sample_limit"),
        per_subject_limit=eval_cfg.get("per_subject_limit"),
        balanced_per_label=eval_cfg.get("balanced_per_label"),
    )
    sampled_distribution = label_distribution(eval_samples)
    print(f"Label distribution after sampling: {sampled_distribution}")
    if eval_cfg.get("balanced_per_label") is not None:
        expected = int(eval_cfg["balanced_per_label"])
        short = {label: sampled_distribution.get(label, 0) for label in labels if sampled_distribution.get(label, 0) != expected}
        if short:
            raise RuntimeError(
                f"Balanced debug subset failed. Expected {expected} samples per label, got {short}."
            )
    if not eval_samples:
        raise RuntimeError("No evaluation samples found for this configuration.")

    lm_usage = build_lm_usage(
        config.get("lm_usage", {}),
        labels=labels,
        input_name=input_provider.name,
        train_samples=train_samples,
        output_instructions=output_handler.instructions(labels),
    )
    client = LMStudioClient(**config.get("lm_client", {}))

    print(f"Input: {input_provider.name}")
    print(f"LM usage: {lm_usage.name}")
    print(f"Output: {output_handler.name}")
    print(f"Eval samples: {len(eval_samples)}")

    records = []
    total = len(eval_samples)
    for idx, sample in enumerate(eval_samples, 1):
        if hasattr(lm_usage, "predict"):
            parsed = lm_usage.predict(sample)
            raw_response = parsed.get("raw_response", "")
        else:
            prompt = lm_usage.build_prompt(sample)
            raw_response = client.complete(prompt)
            parsed = output_handler.parse(raw_response)
        valid = bool(parsed.get("valid", parsed.get("label") is not None))
        pred = int(parsed["label"]) if valid else ""
        records.append(
            {
                "subject": sample.subject,
                "y_true": sample.label,
                "y_pred": pred,
                "valid": valid,
                "parse_error": parsed.get("parse_error", ""),
                "explanation": parsed.get("explanation", ""),
                "raw_response": raw_response,
                **sample.meta,
            }
        )

        if idx % int(eval_cfg.get("log_every", 10)) == 0 or idx == total:
            print(f"{idx}/{total} done")

    metrics = summarize_and_save(
        records,
        labels=labels,
        output_dir=config.get("output_dir", "outputs_modular"),
        run_name=config.get("run_name", "wesad_llm"),
        config=config,
    )
    print("=" * 50)
    if metrics["accuracy"] is None:
        print("Accuracy: n/a (no valid predictions)")
    else:
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Invalid predictions: {metrics['invalid_count']}/{metrics['n_samples']}")
    print(f"Predictions: {metrics['predictions_path']}")
    print(f"Metrics: {metrics['metrics_path']}")


if __name__ == "__main__":
    main()
