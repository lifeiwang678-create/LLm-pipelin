from __future__ import annotations

import json
import pickle
import re
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from Evaluation import label_distribution, limit_samples, summarize_and_save
from Dataset import build_dataset_loader, get_dataset_config
from Input import build_input_provider
from LM import build_lm_usage
from Output import build_output_handler

from .lm_client import OpenAICompatibleClient
from .splits import normalize_subjects, validate_fewshot_split, validate_subject_independent_split


DEFAULT_LM_CLIENT_CONFIG = {
    "api_url": "http://127.0.0.1:1234/v1",
    "api_key": "lm-studio",
    "model": "qwen2.5-14b-instruct",
    "temperature": 0.0,
    "max_tokens": 128,
    "timeout": 1200,
}


def build_experiment_config(args: Namespace) -> dict:
    dataset_cfg = get_dataset_config(args.dataset)
    long_input = args.Input in {"raw_data", "embedding_alignment", "encoded_time_series", "extra_knowledge"}
    lm_timeout = 1200 if args.LM == "multi_agent" else 600 if long_input else 300
    max_tokens = 256 if args.output == "label_explanation" else 128
    default_few_shot_n = 1 if long_input else 2
    default_example_max_chars = 800 if long_input else None
    default_intermediate_max_tokens = 512
    loader_kwargs = dict(dataset_cfg.get("loader_kwargs", {}))
    if getattr(args, "max_rows", None) is not None:
        if args.dataset != "HHAR":
            raise ValueError("--max-rows is only supported for HHAR.")
        loader_kwargs["max_rows"] = int(args.max_rows)

    if args.LM == "few_shot":
        if args.subjects and not args.test_subjects and not args.train_subjects:
            raise ValueError(
                "--subjects is only used for direct-mode evaluation. "
                "For few_shot, use --train-subjects/--test-subjects or rely on "
                "the dataset's subject-independent defaults."
            )
    data_cfg = {
        "subjects": args.subjects if args.subjects is not None else dataset_cfg.get("subjects"),
        "train_subjects": (
            args.train_subjects
            if args.train_subjects is not None
            else dataset_cfg.get("train_subjects")
        ),
        "test_subjects": (
            args.test_subjects
            if args.test_subjects is not None
            else dataset_cfg.get("test_subjects")
        ),
        "subject_split": (
            args.subject_split
            if getattr(args, "subject_split", None) is not None
            else dataset_cfg.get("subject_split", "subject_independent")
        ),
        "subjects_explicit": args.subjects is not None,
        "train_subjects_explicit": args.train_subjects is not None,
        "test_subjects_explicit": args.test_subjects is not None,
    }
    data_cfg.update(
        {
            "use_processed": bool(getattr(args, "use_processed", False)),
            "processed_dir": getattr(args, "processed_dir", "Processed"),
            "processed_file": getattr(args, "processed_file", None),
            "use_input_cache": bool(getattr(args, "use_input_cache", False)),
            "input_cache_dir": getattr(args, "input_cache_dir", "Processed"),
            "input_cache_file": getattr(args, "input_cache_file", None),
            "train_input_cache_file": getattr(args, "train_input_cache_file", None),
            "eval_input_cache_file": getattr(args, "eval_input_cache_file", None),
        }
    )

    dataset_config = {
        "name": args.dataset,
        "data_dir": args.data_dir or dataset_cfg["data_dir"],
        "loader_kwargs": loader_kwargs,
    }
    input_config = {
        "type": args.Input,
        "dataset": args.dataset,
    }
    if args.Input == "extra_knowledge":
        input_config.update(
            {
                "knowledge_file": getattr(args, "knowledge_file", None),
                "knowledge_text": getattr(args, "knowledge_text", ""),
                "knowledge_mode": getattr(args, "knowledge_mode", None),
            }
        )

    return {
        "run_name": f"{args.dataset}_{args.Input}_{args.LM}_{args.output}",
        "result_filename_style": "compact",
        "labels": args.labels if args.labels is not None else dataset_cfg.get("labels", [0, 1]),
        "output_dir": "Results",
        "dataset": dataset_config,
        "data": data_cfg,
        "input": input_config,
        "lm_usage": {
            "type": args.LM,
            # 旧实现写的是 `args.few_shot_n_per_class or default`,这样用户传 0 会被吞掉。
            # 改成显式 None 判断,让 0 可以原样传递给下游 (虽然 FewShotUsage 仍会拒绝 <1)。
            "n_per_class": (
                args.few_shot_n_per_class
                if args.few_shot_n_per_class is not None
                else default_few_shot_n
            ),
            "random_state": 42,
            "example_selection": getattr(args, "few_shot_example_selection", "leave_one_subject_out"),
            "example_subjects": int(getattr(args, "few_shot_example_subjects", 3) or 3),
            "examples_per_subject_per_label": int(
                getattr(args, "few_shot_examples_per_subject_per_label", 1) or 1
            ),
            "exclude_eval_subject": True,
            "example_max_chars": args.few_shot_example_max_chars
            if args.few_shot_example_max_chars is not None
            else default_example_max_chars,
            # multi_agent 前两步用更大的 token 上限,避免结构化 JSON 被全局 max_tokens 截断。
            "intermediate_max_tokens": (
                int(args.multi_agent_intermediate_max_tokens)
                if getattr(args, "multi_agent_intermediate_max_tokens", None) is not None
                else default_intermediate_max_tokens
            ),
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
            "concurrency": int(getattr(args, "concurrency", 1) or 1),
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

    train_subjects, test_subjects = _resolve_run_subjects(config, dataset_loader, usage_type)
    fewshot_leave_one_out = _fewshot_uses_leave_one_subject_out(config, usage_type)
    if train_subjects:
        print(f"Train subjects: {train_subjects}")
    if test_subjects:
        print(f"Test/eval subjects: {test_subjects}")
    if fewshot_leave_one_out:
        lm_cfg = config.get("lm_usage") or {}
        print(
            "Few-shot example policy: leave_one_subject_out "
            f"(subjects={int(lm_cfg.get('example_subjects', 3))}, "
            f"per_subject_per_label={int(lm_cfg.get('examples_per_subject_per_label', 1))}, "
            f"seed={int(lm_cfg.get('random_state', 42))})"
        )

    if config["data"].get("use_input_cache"):
        if usage_type == "few_shot":
            if fewshot_leave_one_out:
                train_samples = _load_input_cache_samples(config, input_provider, None, labels, role="train")
            else:
                validate_fewshot_split(train_subjects, test_subjects)
                train_samples = _load_input_cache_samples(config, input_provider, train_subjects, labels, role="train")
            print(f"Few-shot example-source label distribution: {label_distribution(train_samples)}")
            eval_sensor_samples = _load_input_cache_samples(config, input_provider, test_subjects, labels, role="eval")
        else:
            train_samples = []
            eval_sensor_samples = _load_input_cache_samples(config, input_provider, test_subjects, labels, role="eval")
    elif usage_type == "few_shot":
        if fewshot_leave_one_out:
            example_subjects = _discover_subjects_for_split(dataset_loader)
            train_sensor_samples = _load_sensor_samples(config, dataset_loader, example_subjects, labels)
        else:
            validate_fewshot_split(train_subjects, test_subjects)
            train_sensor_samples = _load_sensor_samples(config, dataset_loader, train_subjects, labels)
        print(f"Few-shot example-source label distribution: {label_distribution(train_sensor_samples)}")
        train_samples = input_provider.transform_all(train_sensor_samples)
        eval_sensor_samples = _load_sensor_samples(config, dataset_loader, test_subjects, labels)
    else:
        train_samples = []
        eval_sensor_samples = _load_sensor_samples(config, dataset_loader, test_subjects, labels)

    eval_cfg = config["evaluation"]
    print(f"Label distribution before sampling: {label_distribution(eval_sensor_samples)}")
    eval_sensor_samples = limit_samples(
        eval_sensor_samples,
        limit=eval_cfg.get("sample_limit"),
        per_subject_limit=eval_cfg.get("per_subject_limit"),
        balanced_per_label=eval_cfg.get("balanced_per_label"),
        # ===== 修改 (Fix 1): 把平衡采样的随机种子串下去, 默认 42 保证可复现。
        # 旧实现 balanced_per_label 直接顺序取前 N 个样本, 会让 debug 子集偏向
        # 录音早期, 跨实验对比会沾上这种顺序偏差。 =====
        random_state=eval_cfg.get("random_state", 42),
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
    if config["data"].get("use_input_cache"):
        eval_samples = eval_sensor_samples
    else:
        eval_samples = input_provider.transform_all(eval_sensor_samples)
    if not eval_samples:
        raise RuntimeError("No evaluation samples found.")

    output_instructions = output_handler.instructions(labels)
    lm_usage = build_lm_usage(
        config["lm_usage"],
        labels=labels,
        input_name=input_provider.name,
        train_samples=train_samples,
        output_instructions=output_instructions,
        dataset=dataset_loader.name,
    )

    print(f"Dataset: {dataset_loader.name}")
    print(f"Input: {input_provider.name}")
    print(f"LM usage: {lm_usage.name}")
    print(f"Output: {output_handler.name}")
    print(f"Eval samples: {len(eval_samples)}")
    concurrency = max(1, int(eval_cfg.get("concurrency", 1) or 1))
    print(f"LLM request concurrency: {concurrency}")

    # multi_agent 中间步骤的输出 (evidence_extraction、candidate_evaluation) 体量大、
    # 且是嵌套结构,旧实现直接写进 sample.meta 然后展开到 CSV 列里,既污染列、又会
    # 把同一 sample 在多次实验复用时串台。这里改为按 run 写到一个 JSONL 侧文件,
    # 主 CSV 只保留预测结果和已知安全的元数据。
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    trace_stem = f"{config['run_name']}_{timestamp}"
    trace_path = output_dir / f"{trace_stem}_traces.jsonl"
    trace_path.unlink(missing_ok=True)
    trace_written = False

    # 仅保留这些已知安全的 meta key 写进 CSV,避免新 Input/LM 模块往 meta 里塞嵌套结构
    # 后悄悄把 CSV 列结构搞乱。新增 meta 字段需要显式登记到这里。
    sample_meta_safe_keys = {
        "input_type",
        "input_canonical_type",
        "data_path",
        "qa_path",
        "data_index",
        "source",
        "local_index",
        "start_index",
        "end_index",
        "sample_id",
        "true_label",
        "original_label",
        "original_state",
        "activity_label",
        "original_activity_id",
        "window_start",
        "window_end",
        "window_start_sec",
        "window_end_sec",
        "epoch_id",
    }
    records, trace_written = _run_eval_samples(
        eval_samples=eval_samples,
        config=config,
        labels=labels,
        input_name=input_provider.name,
        train_samples=train_samples,
        output_instructions=output_instructions,
        dataset=dataset_loader.name,
        output_handler=output_handler,
        usage_type=usage_type,
        shared_lm_usage=lm_usage if not hasattr(lm_usage, "run_agent_pipeline") else None,
        sample_meta_safe_keys=sample_meta_safe_keys,
        trace_path=trace_path,
        concurrency=concurrency,
        log_every=int(eval_cfg.get("log_every", 10) or 10),
    )

    # 把 trace 文件路径写入 config,让 summarize_and_save 一起记到 metrics.json,方便事后定位
    if trace_written:
        config = dict(config)
        config["agent_trace_path"] = str(trace_path)

    metrics = summarize_and_save(
        records,
        labels=labels,
        output_dir=config["output_dir"],
        run_name=config["run_name"],
        config=config,
    )
    if trace_written:
        metrics["agent_trace_path"] = str(trace_path)
        print(f"Agent traces (multi_agent intermediate outputs): {trace_path}")
    print("=" * 50)
    if metrics["accuracy_valid_only"] is None:
        print("Accuracy valid-only: n/a (no valid predictions)")
        print("Macro-F1 valid-only: n/a")
        print("Weighted-F1 valid-only: n/a")
    else:
        print(f"Accuracy valid-only: {metrics['accuracy_valid_only'] * 100:.2f}%")
        print(f"Macro-F1 valid-only: {metrics['macro_f1_valid_only']:.4f}")
        print(f"Weighted-F1 valid-only: {metrics['weighted_f1_valid_only']:.4f}")
    if metrics["accuracy_all_samples_invalid_as_wrong"] is None:
        print("Accuracy all-samples invalid-as-wrong: n/a")
        print("Macro-F1 all-samples invalid-as-wrong: n/a")
        print("Weighted-F1 all-samples invalid-as-wrong: n/a")
    else:
        print(
            "Accuracy all-samples invalid-as-wrong: "
            f"{metrics['accuracy_all_samples_invalid_as_wrong'] * 100:.2f}%"
        )
        print(f"Macro-F1 all-samples invalid-as-wrong: {metrics['macro_f1_all_samples_invalid_as_wrong']:.4f}")
        print(f"Weighted-F1 all-samples invalid-as-wrong: {metrics['weighted_f1_all_samples_invalid_as_wrong']:.4f}")
    print(f"Confusion matrix labels: {metrics['confusion_matrix_labels']}")
    print(f"Confusion matrix label names: {metrics['confusion_matrix_label_names']}")
    print(f"Confusion matrix valid-only: {metrics['confusion_matrix_valid_only']}")
    print(
        "Confusion matrix all-samples invalid-as-wrong: "
        f"{metrics['confusion_matrix_all_samples_invalid_as_wrong']}"
    )
    print(f"Invalid predictions: {metrics['invalid_count']}/{metrics['n_samples']}")
    usage_summary = metrics.get("usage_summary", {})
    if usage_summary:
        print(f"LLM calls: {usage_summary['total_llm_calls']}")
        print(f"Total tokens: {usage_summary['total_tokens']}")
        print(f"Total LLM elapsed sec: {usage_summary['total_elapsed_time_sec']:.3f}")
        print(f"Token usage missing calls: {usage_summary['token_usage_missing_count']}")
    print(f"Results: {Path(metrics['predictions_path'])}")
    return metrics


def _fewshot_uses_leave_one_subject_out(config: dict, usage_type: str | None = None) -> bool:
    normalized_usage = usage_type or _normalize_lm_usage_type((config.get("lm_usage") or {}).get("type", "direct"))
    if normalized_usage != "few_shot":
        return False
    selection = str((config.get("lm_usage") or {}).get("example_selection", "class_balanced"))
    return selection.strip().lower().replace("-", "_") in {"leave_one_subject_out", "loo", "subject_loo"}


def _resolve_run_subjects(config: dict, dataset_loader, usage_type: str) -> tuple[list[str] | None, list[str] | None]:
    data_config = config.get("data") or {}
    split_mode = str(data_config.get("subject_split", "subject_independent") or "subject_independent").lower()
    subjects = normalize_subjects(data_config.get("subjects"))
    train_subjects = normalize_subjects(data_config.get("train_subjects"))
    test_subjects = normalize_subjects(data_config.get("test_subjects"))
    fewshot_leave_one_out = _fewshot_uses_leave_one_subject_out(config, usage_type)

    if split_mode == "all":
        if usage_type == "few_shot" and not fewshot_leave_one_out:
            train_subjects, test_subjects = _complete_subject_independent_split(
                dataset_loader,
                train_subjects,
                test_subjects,
                subjects,
            )
            validate_fewshot_split(train_subjects, test_subjects)
            return train_subjects, test_subjects
        return train_subjects, subjects

    if split_mode != "subject_independent":
        raise ValueError(
            f"Unknown subject_split={split_mode!r}. "
            "Use 'subject_independent' or 'all'."
        )

    # Explicit --subjects is still allowed as a debug/evaluation subset for
    # direct and multi-agent runs. Formal/default runs use held-out test subjects.
    if (
        usage_type != "few_shot"
        and subjects
        and data_config.get("subjects_explicit")
        and not data_config.get("test_subjects_explicit")
    ):
        return train_subjects, subjects

    train_subjects, test_subjects = _complete_subject_independent_split(
        dataset_loader,
        train_subjects,
        test_subjects,
        subjects,
    )
    validate_subject_independent_split(train_subjects, test_subjects)
    if usage_type == "few_shot" and not fewshot_leave_one_out:
        validate_fewshot_split(train_subjects, test_subjects)
    return train_subjects, test_subjects


def _complete_subject_independent_split(
    dataset_loader,
    train_subjects: list[str] | None,
    test_subjects: list[str] | None,
    subjects: list[str] | None,
) -> tuple[list[str], list[str]]:
    if train_subjects and test_subjects:
        return train_subjects, test_subjects

    all_subjects = subjects or _discover_subjects_for_split(dataset_loader)
    if len(all_subjects) < 2:
        raise ValueError(
            "Subject-independent split requires at least two subjects. "
            "Pass --train-subjects and --test-subjects explicitly."
        )

    if train_subjects and not test_subjects:
        test_subjects = [subject for subject in all_subjects if subject not in set(train_subjects)]
    elif test_subjects and not train_subjects:
        train_subjects = [subject for subject in all_subjects if subject not in set(test_subjects)]
    else:
        train_subjects = [all_subjects[0]]
        test_subjects = all_subjects[1:]

    return train_subjects or [], test_subjects or []


def _discover_subjects_for_split(dataset_loader) -> list[str]:
    discover = getattr(dataset_loader, "_discover_subjects", None)
    if not callable(discover):
        raise ValueError(
            f"{dataset_loader.name} loader cannot auto-discover subjects. "
            "Pass --train-subjects and --test-subjects explicitly."
        )
    subjects = [str(subject) for subject in discover()]
    return sorted(subjects, key=_subject_sort_key)


def _subject_sort_key(subject: str) -> list[object]:
    parts = re.split(r"(\d+)", str(subject))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _load_sensor_samples(config: dict, dataset_loader, subjects, labels: list[int]):
    data_config = config.get("data") or {}
    if data_config.get("use_processed"):
        return _load_processed_sensor_samples(config, dataset_loader, subjects, labels)
    return dataset_loader.load(subjects, labels)


def _run_eval_samples(
    *,
    eval_samples: list,
    config: dict,
    labels: list[int],
    input_name: str,
    train_samples: list,
    output_instructions: str,
    dataset: str,
    output_handler,
    usage_type: str,
    shared_lm_usage,
    sample_meta_safe_keys: set[str],
    trace_path: Path,
    concurrency: int,
    log_every: int,
) -> tuple[list[dict], bool]:
    total = len(eval_samples)
    log_every = max(1, int(log_every or 10))
    records_by_index: list[dict | None] = [None] * total
    trace_written = False

    if total == 0:
        return [], trace_written

    if concurrency <= 1:
        for idx, sample in enumerate(eval_samples, 1):
            result = _run_one_sample(
                idx=idx,
                sample=sample,
                config=config,
                labels=labels,
                input_name=input_name,
                train_samples=train_samples,
                output_instructions=output_instructions,
                dataset=dataset,
                output_handler=output_handler,
                usage_type=usage_type,
                shared_lm_usage=shared_lm_usage,
                sample_meta_safe_keys=sample_meta_safe_keys,
            )
            records_by_index[idx - 1] = result["record"]
            if result["trace_record"]:
                _append_trace_record(trace_path, result["trace_record"])
                trace_written = True

            if idx % log_every == 0 or idx == total:
                print(f"{idx}/{total} done")

        return [record for record in records_by_index if record is not None], trace_written

    workers = min(max(1, int(concurrency)), total)
    print(f"Using {workers} worker threads for LLM requests.")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_one_sample,
                idx=idx,
                sample=sample,
                config=config,
                labels=labels,
                input_name=input_name,
                train_samples=train_samples,
                output_instructions=output_instructions,
                dataset=dataset,
                output_handler=output_handler,
                usage_type=usage_type,
                shared_lm_usage=shared_lm_usage,
                sample_meta_safe_keys=sample_meta_safe_keys,
            ): idx
            for idx, sample in enumerate(eval_samples, 1)
        }
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                raise RuntimeError(f"Evaluation failed at sample {idx}/{total}.") from exc

            records_by_index[idx - 1] = result["record"]
            if result["trace_record"]:
                _append_trace_record(trace_path, result["trace_record"])
                trace_written = True

            completed += 1
            if completed % log_every == 0 or completed == total:
                print(f"{completed}/{total} done")

    missing = [idx for idx, record in enumerate(records_by_index, 1) if record is None]
    if missing:
        raise RuntimeError(f"Missing evaluation records for sample indexes: {missing[:10]}")
    return [record for record in records_by_index if record is not None], trace_written


def _run_one_sample(
    *,
    idx: int,
    sample,
    config: dict,
    labels: list[int],
    input_name: str,
    train_samples: list,
    output_instructions: str,
    dataset: str,
    output_handler,
    usage_type: str,
    sample_meta_safe_keys: set[str],
    shared_lm_usage=None,
) -> dict:
    client = OpenAICompatibleClient(**config["lm_client"])
    lm_usage = shared_lm_usage
    if lm_usage is None:
        lm_usage = build_lm_usage(
            config["lm_usage"],
            labels=labels,
            input_name=input_name,
            train_samples=train_samples,
            output_instructions=output_instructions,
            dataset=dataset,
        )

    usage_start = len(getattr(client, "usage_records", []))
    few_shot_example_subjects = None
    few_shot_example_count = None
    if hasattr(lm_usage, "run_agent_pipeline"):
        raw_response = lm_usage.run_agent_pipeline(sample, client)
    else:
        if hasattr(lm_usage, "build_prompt_with_metadata"):
            prompt, few_shot_example_subjects, few_shot_example_count = lm_usage.build_prompt_with_metadata(sample)
        else:
            prompt = lm_usage.build_prompt(sample)
            if hasattr(lm_usage, "last_example_subjects"):
                few_shot_example_subjects = list(getattr(lm_usage, "last_example_subjects", []) or [])
                few_shot_example_count = int(getattr(lm_usage, "last_example_count", 0) or 0)
        raw_response = client.complete(prompt)

    llm_usage = _aggregate_llm_usage(getattr(client, "usage_records", [])[usage_start:])
    llm_usage.update(_estimate_sample_cost(llm_usage, config))
    parsed = output_handler.parse(raw_response)
    valid = bool(parsed.get("valid", parsed.get("label") is not None))

    trace = getattr(lm_usage, "last_trace", None)
    trace_record = None
    if trace:
        trace_record = {
            "sample_index": idx,
            "subject": sample.subject,
            "y_true": sample.label,
            "trace": trace,
        }
        if hasattr(lm_usage, "last_trace"):
            lm_usage.last_trace = None

    meta = dict(getattr(sample, "meta", {}) or {})
    flat_meta = {k: v for k, v in meta.items() if k in sample_meta_safe_keys}
    predicted_label = int(parsed["label"]) if valid else ""
    record = {
        "sample_id": meta.get("sample_id", idx),
        "dataset": sample.dataset,
        "subject": sample.subject,
        "window_start": meta.get("window_start", meta.get("window_start_sec", "")),
        "window_end": meta.get("window_end", meta.get("window_end_sec", "")),
        "true_label": sample.label,
        "predicted_label": predicted_label,
        "y_true": sample.label,
        "y_pred": predicted_label,
        "valid": valid,
        "parse_error": parsed.get("parse_error", ""),
        "explanation": parsed.get("explanation", ""),
        "raw_response": raw_response,
        "input_type": meta.get("input_type", input_name),
        "lm_type": usage_type,
        "output_type": config["output"].get("type", "label_only"),
        "few_shot_example_subjects": ";".join(few_shot_example_subjects)
        if few_shot_example_subjects is not None
        else "",
        "few_shot_example_count": few_shot_example_count
        if few_shot_example_count is not None
        else "",
        **llm_usage,
        **flat_meta,
    }
    return {"record": record, "trace_record": trace_record}


def _append_trace_record(trace_path: Path, trace_record: dict) -> None:
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")


def _load_processed_sensor_samples(config: dict, dataset_loader, subjects, labels: list[int]):
    data_config = config.get("data") or {}
    dataset_name = dataset_loader.name
    processed_path = _processed_file_path(config, dataset_name)
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed cache not found: {processed_path}. "
            "Run preprocess_datasets.py first, or remove --use-processed."
        )

    with processed_path.open("rb") as f:
        payload = pickle.load(f)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    _validate_cache_metadata(
        config=config,
        metadata=metadata,
        cache_path=processed_path,
        cache_kind="processed dataset",
    )
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"Processed cache must contain a list of SensorSample objects: {processed_path}")

    subject_filter = {str(subject) for subject in subjects} if subjects else None
    label_filter = {int(label) for label in labels}
    filtered = []
    for sample in samples:
        sample_dataset = str(getattr(sample, "dataset", dataset_name))
        sample_subject = str(getattr(sample, "subject", ""))
        try:
            sample_label = int(getattr(sample, "label"))
        except (TypeError, ValueError):
            continue
        if sample_dataset and sample_dataset != dataset_name:
            continue
        if subject_filter and sample_subject not in subject_filter:
            continue
        if sample_label not in label_filter:
            continue
        filtered.append(sample)

    print(
        f"Loaded processed cache: {processed_path} "
        f"({len(filtered)} selected / {len(samples)} stored samples)"
    )
    return filtered


def _processed_file_path(config: dict, dataset_name: str) -> Path:
    data_config = config.get("data") or {}
    explicit = data_config.get("processed_file") or config.get("processed_file")
    if explicit:
        return Path(explicit)
    processed_dir = Path(data_config.get("processed_dir") or config.get("processed_dir") or "Processed")
    return processed_dir / f"{dataset_name}_binary_windows.pkl"


def _load_input_cache_samples(config: dict, input_provider, subjects, labels: list[int], role: str = "eval"):
    data_config = config.get("data") or {}
    dataset_name = config["dataset"]["name"]
    input_cache_path = _input_cache_file_path(config, dataset_name, input_provider.name, role=role)
    if not input_cache_path.exists():
        raise FileNotFoundError(
            f"Input cache not found: {input_cache_path}. "
            "Run preprocess_inputs.py first, or remove --use-input-cache."
        )

    with input_cache_path.open("rb") as f:
        payload = pickle.load(f)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    _validate_cache_metadata(
        config=config,
        metadata=metadata,
        cache_path=input_cache_path,
        cache_kind="input",
        expected_input_type=input_provider.name,
    )
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"Input cache must contain a list of LLMSample objects: {input_cache_path}")

    subject_filter = {str(subject) for subject in subjects} if subjects else None
    label_filter = {int(label) for label in labels}
    filtered = []
    for sample in samples:
        sample_dataset = str(getattr(sample, "dataset", dataset_name))
        sample_subject = str(getattr(sample, "subject", ""))
        try:
            sample_label = int(getattr(sample, "label"))
        except (TypeError, ValueError):
            continue
        if sample_dataset and sample_dataset != dataset_name:
            continue
        if subject_filter and sample_subject not in subject_filter:
            continue
        if sample_label not in label_filter:
            continue
        filtered.append(sample)

    print(
        f"Loaded input cache: {input_cache_path} "
        f"({len(filtered)} selected / {len(samples)} stored samples)"
    )
    return filtered


def _input_cache_file_path(config: dict, dataset_name: str, input_name: str, role: str = "eval") -> Path:
    data_config = config.get("data") or {}
    role_key = "train_input_cache_file" if role == "train" else "eval_input_cache_file"
    role_explicit = data_config.get(role_key) or config.get(role_key)
    if role_explicit:
        return Path(role_explicit)
    explicit = data_config.get("input_cache_file") or config.get("input_cache_file")
    if explicit:
        return Path(explicit)
    input_cache_dir = Path(
        data_config.get("input_cache_dir") or config.get("input_cache_dir") or "Processed"
    )
    return input_cache_dir / f"{dataset_name}_{input_name}_samples.pkl"


def _validate_cache_metadata(
    config: dict,
    metadata: dict,
    cache_path: Path,
    cache_kind: str,
    expected_input_type: str | None = None,
) -> None:
    """Fail fast when a cache was built with stale dataset/input settings."""
    if not isinstance(metadata, dict):
        raise ValueError(f"{cache_kind.capitalize()} cache has no metadata: {cache_path}")

    expected_dataset = str(config["dataset"]["name"])
    actual_dataset = str(metadata.get("dataset") or "")
    if actual_dataset and actual_dataset != expected_dataset:
        raise ValueError(
            f"{cache_kind.capitalize()} cache dataset mismatch: {cache_path}. "
            f"Expected {expected_dataset}, found {actual_dataset}."
        )

    if expected_input_type is not None:
        actual_input_type = metadata.get("input_type")
        if actual_input_type and str(actual_input_type) != str(expected_input_type):
            raise ValueError(
                f"Input cache type mismatch: {cache_path}. "
                f"Expected {expected_input_type}, found {actual_input_type}."
            )

    source_metadata = metadata.get("source_processed_metadata")
    if not isinstance(source_metadata, dict):
        subset_metadata = metadata.get("source_data_subset_metadata")
        if isinstance(subset_metadata, dict):
            source_metadata = subset_metadata.get("source_processed_metadata")
    if isinstance(source_metadata, dict):
        loader_metadata = source_metadata
    else:
        loader_metadata = metadata

    expected_kwargs = dict(config.get("dataset", {}).get("loader_kwargs") or {})
    actual_kwargs = dict(loader_metadata.get("loader_kwargs") or {})
    mismatches = _loader_kwarg_mismatches(expected_dataset, expected_kwargs, actual_kwargs)
    if mismatches:
        detail = "; ".join(
            f"{key}: expected {expected!r}, cache has {actual!r}"
            for key, expected, actual in mismatches
        )
        raise RuntimeError(
            f"Stale {cache_kind} cache detected: {cache_path}. {detail}. "
            "Delete/rebuild the cache with preprocess_datasets.py and preprocess_inputs.py, "
            "or run without --use-processed/--use-input-cache."
        )


def _loader_kwarg_mismatches(
    dataset_name: str,
    expected_kwargs: dict,
    actual_kwargs: dict,
) -> list[tuple[str, object, object]]:
    keys_by_dataset = {
        "WESAD": ("physiology_window_sec", "acc_window_sec", "stride_sec"),
        "HHAR": (
            "window_size",
            "stride_size",
            "sampling_rate",
            "min_samples_per_window",
            "max_gap_sec",
            "include_gyroscope",
            "max_rows",
        ),
        "DREAMT": ("sampling_rate", "epoch_seconds", "stride_seconds", "min_epoch_fraction"),
    }
    keys = keys_by_dataset.get(str(dataset_name).upper(), tuple(expected_kwargs))
    mismatches = []
    for key in keys:
        if key not in expected_kwargs:
            continue
        expected = expected_kwargs.get(key)
        actual = actual_kwargs.get(key, None)
        if not _cache_values_equal(expected, actual):
            mismatches.append((key, expected, actual))
    return mismatches


def _cache_values_equal(expected, actual) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return bool(actual) == expected
    try:
        return abs(float(expected) - float(actual)) < 1e-9
    except (TypeError, ValueError):
        return str(expected) == str(actual)


def _aggregate_llm_usage(usage_records: list[dict]) -> dict:
    call_count = len(usage_records)
    token_available_count = sum(1 for record in usage_records if record.get("total_tokens") is not None)
    token_missing_count = call_count - token_available_count
    return {
        "llm_call_count": call_count,
        "prompt_chars": _sum_numeric_usage(usage_records, "prompt_chars"),
        "completion_chars": _sum_numeric_usage(usage_records, "completion_chars"),
        "total_chars": _sum_numeric_usage(usage_records, "total_chars"),
        "prompt_tokens": _sum_optional_usage(usage_records, "prompt_tokens"),
        "completion_tokens": _sum_optional_usage(usage_records, "completion_tokens"),
        "total_tokens": _sum_optional_usage(usage_records, "total_tokens"),
        "elapsed_time_sec": sum(float(record.get("elapsed_time_sec") or 0.0) for record in usage_records),
        "llm_token_usage_available_count": token_available_count,
        "llm_token_usage_missing_count": token_missing_count,
    }


def _sum_optional_usage(usage_records: list[dict], key: str) -> int | None:
    values = [record.get(key) for record in usage_records if record.get(key) is not None]
    if not values:
        return None
    return int(sum(int(value) for value in values))


def _sum_numeric_usage(usage_records: list[dict], key: str) -> int:
    return int(sum(int(record.get(key) or 0) for record in usage_records))


def _estimate_sample_cost(llm_usage: dict, config: dict) -> dict:
    cost_config = config.get("cost_estimate", config.get("cost", {}))
    if not isinstance(cost_config, dict):
        cost_config = {}
    input_cost_per_1m = float(cost_config.get("input_cost_per_1m_tokens", 0.0) or 0.0)
    output_cost_per_1m = float(cost_config.get("output_cost_per_1m_tokens", 0.0) or 0.0)
    input_cost = _token_cost(llm_usage.get("prompt_tokens"), input_cost_per_1m)
    output_cost = _token_cost(llm_usage.get("completion_tokens"), output_cost_per_1m)
    total_cost = input_cost + output_cost if input_cost is not None and output_cost is not None else None
    return {
        "estimated_input_cost": input_cost,
        "estimated_output_cost": output_cost,
        "estimated_total_cost": total_cost,
    }


def _token_cost(tokens: int | float | None, cost_per_1m_tokens: float) -> float | None:
    if tokens is None:
        return None
    return (float(tokens) / 1_000_000.0) * cost_per_1m_tokens


def _normalize_run_config(config: dict, dataset_name: str | None) -> dict:
    normalized = dict(config)

    raw_dataset_config = normalized.get("dataset")
    dataset_config = _resolve_dataset_config(normalized, dataset_name)
    normalized["dataset"] = dataset_config
    dataset_defaults = get_dataset_config(dataset_config["name"])
    normalized["labels"] = [
        int(label)
        for label in normalized.get("labels", dataset_defaults.get("labels", [0, 1]))
    ]

    input_config = dict(normalized.get("input") or {})
    input_config.setdefault("type", "feature_description")
    input_config.setdefault("dataset", dataset_config["name"])
    normalized["input"] = input_config

    data_config = dict(normalized.get("data") or {})
    data_config.setdefault("subject_split", dataset_defaults.get("subject_split", "subject_independent"))
    if not any(data_config.get(key) is not None for key in ("subjects", "train_subjects", "test_subjects")):
        data_config["subjects"] = dataset_defaults.get("subjects")
        data_config["train_subjects"] = dataset_defaults.get("train_subjects")
        data_config["test_subjects"] = dataset_defaults.get("test_subjects")
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

    evaluation_config = dict(normalized.get("evaluation") or {})
    evaluation_config.setdefault("concurrency", 1)
    normalized["evaluation"] = evaluation_config
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
