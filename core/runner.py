from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from Dataset import get_dataset_config
from Input import build_input_provider
from LM import build_lm_usage
from Output import build_output_handler

from .evaluation import label_distribution, limit_samples, summarize_and_save
from .lm_client import LMStudioClient
from .splits import validate_fewshot_split


def build_experiment_config(args: Namespace) -> dict:
    dataset_cfg = get_dataset_config(args.dataset)
    lm_timeout = 60 if args.Input == "raw_data" else 30
    max_tokens = 256 if args.Input == "raw_data" else 128

    if args.LM == "few_shot":
        data_cfg = {
            "train_subjects": args.train_subjects or dataset_cfg.get("train_subjects"),
            "test_subjects": args.test_subjects or dataset_cfg.get("test_subjects"),
        }
    else:
        data_cfg = {
            "subjects": args.subjects or dataset_cfg.get("subjects"),
        }

    input_cfg = {
        "type": args.Input,
        "data_dir": dataset_cfg["data_dir"],
    }
    if args.Input == "feature_description":
        input_cfg["pattern"] = dataset_cfg["feature_pattern"]
    if args.Input == "raw_data":
        input_cfg["window_sec"] = 10.0
        input_cfg["stride_sec"] = 15.0

    return {
        "run_name": f"{args.dataset}_{args.Input}_{args.LM}_{args.output}",
        "result_filename_style": "compact",
        "labels": args.labels,
        "output_dir": "Results",
        "data": data_cfg,
        "input": input_cfg,
        "lm_usage": {
            "type": args.LM,
            "n_per_class": 2,
            "random_state": 42,
        },
        "output": {
            "type": args.output,
        },
        "lm_client": {
            "api_url": args.api_url,
            "api_key": args.api_key,
            "model": args.llm,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "timeout": lm_timeout,
        },
        "evaluation": {
            "balanced_per_label": args.balanced_per_label,
            "log_every": args.log_every,
        },
    }


def run_from_args(args: Namespace) -> dict:
    config = build_experiment_config(args)
    return run_experiment(config, dataset_name=args.dataset)


def run_experiment(config: dict, dataset_name: str | None = None) -> dict:
    labels = [int(label) for label in config["labels"]]
    input_provider = build_input_provider(config["input"])
    output_handler = build_output_handler(config["output"], labels)
    usage_type = config["lm_usage"]["type"]

    train_subjects = config["data"].get("train_subjects")
    test_subjects = config["data"].get("test_subjects") or config["data"].get("subjects")

    if usage_type == "few_shot":
        validate_fewshot_split(train_subjects, test_subjects)
        train_samples = input_provider.load(train_subjects, labels)
        print(f"Few-shot train label distribution: {label_distribution(train_samples)}")
        eval_samples = input_provider.load(test_subjects, labels)
    else:
        train_samples = []
        eval_samples = input_provider.load(test_subjects, labels)

    eval_cfg = config["evaluation"]
    print(f"Label distribution before sampling: {label_distribution(eval_samples)}")
    eval_samples = limit_samples(
        eval_samples,
        balanced_per_label=eval_cfg.get("balanced_per_label"),
    )
    print(f"Label distribution after sampling: {label_distribution(eval_samples)}")
    if not eval_samples:
        raise RuntimeError("No evaluation samples found.")

    lm_usage = build_lm_usage(
        config["lm_usage"],
        labels=labels,
        input_name=input_provider.name,
        train_samples=train_samples,
        output_instructions=output_handler.instructions(labels),
    )
    client = LMStudioClient(**config["lm_client"])

    if dataset_name:
        print(f"Dataset: {dataset_name}")
    print(f"Input: {input_provider.name}")
    print(f"LM usage: {lm_usage.name}")
    print(f"Output: {output_handler.name}")
    print(f"Eval samples: {len(eval_samples)}")

    records = []
    for idx, sample in enumerate(eval_samples, 1):
        prompt = lm_usage.build_prompt(sample)
        raw_response = client.complete(prompt)
        parsed = output_handler.parse(raw_response)
        valid = bool(parsed.get("valid", parsed.get("label") is not None))
        records.append(
            {
                "subject": sample.subject,
                "y_true": sample.label,
                "y_pred": int(parsed["label"]) if valid else "",
                "valid": valid,
                "parse_error": parsed.get("parse_error", ""),
                "explanation": parsed.get("explanation", ""),
                "raw_response": raw_response,
                **sample.meta,
            }
        )

        if idx % int(eval_cfg.get("log_every", 10)) == 0 or idx == len(eval_samples):
            print(f"{idx}/{len(eval_samples)} done")

    metrics = summarize_and_save(
        records,
        labels=labels,
        output_dir=config["output_dir"],
        run_name=config["run_name"],
        config=config,
    )
    print("=" * 50)
    if metrics["accuracy"] is None:
        print("Accuracy: n/a (no valid predictions)")
    else:
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Invalid predictions: {metrics['invalid_count']}/{metrics['n_samples']}")
    print(f"Results: {Path(metrics['predictions_path'])}")
    return metrics

