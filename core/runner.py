from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from Dataset import build_dataset_loader, get_dataset_config
from Input import build_input_provider
from LM import build_lm_usage
from Output import build_output_handler

from .evaluation import label_distribution, limit_samples, summarize_and_save
from .lm_client import LMStudioClient
from .splits import validate_fewshot_split


def build_experiment_config(args: Namespace) -> dict:
    dataset_cfg = get_dataset_config(args.dataset)
    long_input = args.Input in {"raw_data", "embedding_alignment", "encoded_time_series"}
    lm_timeout = 60 if long_input else 30
    max_tokens = 384 if long_input else 128
    default_few_shot_n = 1 if long_input else 2
    default_example_max_chars = 1500 if long_input else None
    loader_kwargs = dict(dataset_cfg.get("loader_kwargs", {}))

    if args.LM == "few_shot":
        if args.subjects and not args.test_subjects:
            raise ValueError(
                "--subjects is only used for direct-mode evaluation. "
                "For few_shot, use --train-subjects and --test-subjects explicitly."
            )
        if not args.train_subjects or not args.test_subjects:
            raise ValueError("few_shot requires explicit --train-subjects and --test-subjects.")
        data_cfg = {
            "train_subjects": args.train_subjects,
            "test_subjects": args.test_subjects,
        }
    else:
        data_cfg = {
            "subjects": args.subjects or dataset_cfg.get("subjects"),
        }

    dataset_config = {
        "name": args.dataset,
        "data_dir": dataset_cfg["data_dir"],
        "loader_kwargs": loader_kwargs,
    }
    if args.dataset == "WESAD" and args.Input == "feature_description":
        dataset_config["loader_kwargs"]["window_sec"] = 60.0

    return {
        "run_name": f"{args.dataset}_{args.Input}_{args.LM}_{args.output}",
        "result_filename_style": "compact",
        "labels": args.labels,
        "output_dir": "Results",
        "dataset": dataset_config,
        "data": data_cfg,
        "input": {
            "type": args.Input,
            "dataset": args.dataset,
        },
        "lm_usage": {
            "type": args.LM,
            "n_per_class": args.few_shot_n_per_class or default_few_shot_n,
            "random_state": 42,
            "example_max_chars": args.few_shot_example_max_chars
            if args.few_shot_example_max_chars is not None
            else default_example_max_chars,
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
    dataset_config = _resolve_dataset_config(config, dataset_name)
    dataset_loader = build_dataset_loader(dataset_config)
    input_config = dict(config.get("input", {}))
    input_config.setdefault("dataset", dataset_config.get("name"))
    input_provider = build_input_provider(input_config)
    output_handler = build_output_handler(config["output"], labels)
    usage_type = config["lm_usage"]["type"]

    train_subjects = config["data"].get("train_subjects")
    test_subjects = config["data"].get("test_subjects") or config["data"].get("subjects")

    if usage_type == "few_shot":
        validate_fewshot_split(train_subjects, test_subjects)
        train_sensor_samples = dataset_loader.load(train_subjects, labels)
        print(f"Few-shot train label distribution: {label_distribution(train_sensor_samples)}")
        train_samples = input_provider.transform_all(train_sensor_samples)
        eval_sensor_samples = dataset_loader.load(test_subjects, labels)
    else:
        train_samples = []
        eval_sensor_samples = dataset_loader.load(test_subjects, labels)

    eval_cfg = config["evaluation"]
    print(f"Label distribution before sampling: {label_distribution(eval_sensor_samples)}")
    eval_sensor_samples = limit_samples(
        eval_sensor_samples,
        balanced_per_label=eval_cfg.get("balanced_per_label"),
    )
    eval_samples = input_provider.transform_all(eval_sensor_samples)
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

    print(f"Dataset: {dataset_loader.name}")
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


def _resolve_dataset_config(config: dict, dataset_name: str | None) -> dict:
    config_dataset = dict(config.get("dataset") or {})
    input_dataset = (config.get("input") or {}).get("dataset")
    resolved_name = config_dataset.get("name") or dataset_name or input_dataset
    if resolved_name is None or str(resolved_name).strip() == "":
        raise ValueError(
            "Dataset name is required. Set config['dataset']['name'] or config['input']['dataset']."
        )

    resolved_name = str(resolved_name)
    dataset_cfg = get_dataset_config(resolved_name)
    config_dataset.setdefault("name", resolved_name)
    config_dataset.setdefault("data_dir", dataset_cfg["data_dir"])
    config_dataset.setdefault("loader_kwargs", dataset_cfg.get("loader_kwargs", {}))
    return {
        "name": config_dataset["name"],
        "data_dir": config_dataset["data_dir"],
        "loader_kwargs": config_dataset["loader_kwargs"],
    }
