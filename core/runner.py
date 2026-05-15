from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from Evaluation import label_distribution, limit_samples, summarize_and_save
from Dataset import build_dataset_loader, get_dataset_config
from Input import build_input_provider
from LM import build_lm_usage
from Output import build_output_handler

from .lm_client import LMStudioClient
from .splits import validate_fewshot_split


DEFAULT_LM_CLIENT_CONFIG = {
    "api_url": "http://127.0.0.1:1234/v1",
    "api_key": "lm-studio",
    "model": "qwen2.5-14b-instruct",
    "temperature": 0.0,
    "max_tokens": 128,
    "timeout": 30,
}


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
    config = _normalize_run_config(config, dataset_name)
    labels = [int(label) for label in config["labels"]]
    dataset_config = config["dataset"]
    dataset_loader = build_dataset_loader(dataset_config)
    input_config = dict(config.get("input", {}))
    input_config.setdefault("dataset", dataset_config.get("name"))
    input_provider = build_input_provider(input_config)
    output_handler = build_output_handler(config["output"], labels)
    usage_type = _normalize_lm_usage_type(config["lm_usage"].get("type", "direct"))

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
        limit=eval_cfg.get("sample_limit"),
        per_subject_limit=eval_cfg.get("per_subject_limit"),
        balanced_per_label=eval_cfg.get("balanced_per_label"),
    )
    sampled_distribution = label_distribution(eval_sensor_samples)
    print(f"Label distribution after sampling: {sampled_distribution}")
    if eval_cfg.get("balanced_per_label") is not None:
        expected = int(eval_cfg["balanced_per_label"])
        short = {
            label: sampled_distribution.get(label, 0)
            for label in labels
            if sampled_distribution.get(label, 0) != expected
        }
        if short:
            raise RuntimeError(
                f"Balanced debug subset failed. Expected {expected} samples per label, got {short}."
            )
    eval_samples = input_provider.transform_all(eval_sensor_samples)
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
        if hasattr(lm_usage, "run_agent_pipeline"):
            raw_response = lm_usage.run_agent_pipeline(sample, client)
        else:
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
        print("Macro-F1: n/a")
        print("Weighted-F1: n/a")
    else:
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"Confusion matrix labels: {metrics['confusion_matrix_labels']}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Invalid predictions: {metrics['invalid_count']}/{metrics['n_samples']}")
    print(f"Results: {Path(metrics['predictions_path'])}")
    return metrics


def _normalize_run_config(config: dict, dataset_name: str | None) -> dict:
    normalized = dict(config)
    normalized["labels"] = [int(label) for label in normalized.get("labels", [1, 2])]

    raw_dataset_config = normalized.get("dataset")
    has_explicit_loader_kwargs = (
        isinstance(raw_dataset_config, dict) and "loader_kwargs" in raw_dataset_config
    )
    dataset_config = _resolve_dataset_config(normalized, dataset_name)
    input_type = str((normalized.get("input") or {}).get("type", "feature_description")).lower()
    if dataset_config["name"] == "WESAD" and input_type == "feature_description":
        loader_kwargs = dict(dataset_config.get("loader_kwargs", {}))
        if not has_explicit_loader_kwargs:
            loader_kwargs["window_sec"] = 60.0
        else:
            loader_kwargs.setdefault("window_sec", 60.0)
        dataset_config["loader_kwargs"] = loader_kwargs
    normalized["dataset"] = dataset_config

    input_config = dict(normalized.get("input") or {})
    input_config.setdefault("type", "feature_description")
    input_config.setdefault("dataset", dataset_config["name"])
    normalized["input"] = input_config

    dataset_defaults = get_dataset_config(dataset_config["name"])
    data_config = dict(normalized.get("data") or {})
    if not any(key in data_config for key in ("subjects", "train_subjects", "test_subjects")):
        data_config["subjects"] = dataset_defaults.get("subjects")
    normalized["data"] = data_config

    lm_usage_config = dict(normalized.get("lm_usage") or {})
    lm_usage_config.setdefault("type", "direct")
    normalized["lm_usage"] = lm_usage_config

    output_config = dict(normalized.get("output") or {})
    output_config.setdefault("type", "label_only")
    normalized["output"] = output_config

    lm_client_config = dict(DEFAULT_LM_CLIENT_CONFIG)
    lm_client_config.update(dict(normalized.get("lm_client") or {}))
    normalized["lm_client"] = lm_client_config

    normalized["evaluation"] = dict(normalized.get("evaluation") or {})
    normalized.setdefault("output_dir", "Results")
    normalized.setdefault("run_name", _default_run_name(normalized))
    return normalized


def _resolve_dataset_config(config: dict, dataset_name: str | None) -> dict:
    raw_dataset = config.get("dataset") or {}
    if isinstance(raw_dataset, str):
        config_dataset = {"name": raw_dataset}
    elif isinstance(raw_dataset, dict):
        config_dataset = dict(raw_dataset)
    else:
        raise ValueError("config['dataset'] must be a dataset name string or a dictionary.")

    input_config = config.get("input") or {}
    input_dataset = input_config.get("dataset")
    resolved_name = config_dataset.get("name") or dataset_name or input_dataset
    if resolved_name is None or str(resolved_name).strip() == "":
        raise ValueError(
            "Dataset name is required. Set config['dataset']['name'] or config['input']['dataset']."
        )

    resolved_name = str(resolved_name)
    dataset_cfg = get_dataset_config(resolved_name)
    resolved = dict(config_dataset)
    if "data_dir" not in resolved and input_config.get("data_dir"):
        resolved["data_dir"] = input_config["data_dir"]
    resolved.setdefault("name", resolved_name)
    resolved.setdefault("data_dir", dataset_cfg["data_dir"])
    resolved.setdefault("loader_kwargs", dataset_cfg.get("loader_kwargs", {}))
    return resolved


def _normalize_lm_usage_type(kind: str) -> str:
    normalized = str(kind).strip().lower()
    if normalized in {"fewshot", "few-shot"}:
        return "few_shot"
    if normalized in {"multiagent", "multi-agent"}:
        return "multi_agent"
    return normalized


def _default_run_name(config: dict) -> str:
    dataset = config["dataset"]["name"]
    input_type = config["input"].get("type", "feature_description")
    lm_type = config["lm_usage"].get("type", "direct")
    output_type = config["output"].get("type", "label_only")
    return f"{dataset}_{input_type}_{lm_type}_{output_type}"
