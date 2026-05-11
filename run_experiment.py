from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_pipeline.evaluation import limit_samples, summarize_and_save
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
        train_samples = input_provider.load(train_subjects, labels)
        if not train_samples:
            raise RuntimeError("Few-shot mode needs non-empty training samples.")
        eval_samples = input_provider.load(test_subjects, labels)
    else:
        train_samples = []
        eval_samples = input_provider.load(test_subjects, labels)

    eval_cfg = config.get("evaluation", {})
    eval_samples = limit_samples(
        eval_samples,
        limit=eval_cfg.get("sample_limit"),
        per_subject_limit=eval_cfg.get("per_subject_limit"),
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
        pred = int(parsed["label"])
        records.append(
            {
                "subject": sample.subject,
                "y_true": sample.label,
                "y_pred": pred,
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
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Predictions: {metrics['predictions_path']}")
    print(f"Metrics: {metrics['metrics_path']}")


if __name__ == "__main__":
    main()
