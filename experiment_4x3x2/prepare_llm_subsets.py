from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from Dataset import DATASET_REGISTRY, get_dataset_config


INPUT_TYPES = ["raw_data", "feature_description", "encoded_time_series", "extra_knowledge"]
SUBSET_SPECS = {
    "debug": {"per_label": 3, "subject_balanced": False},
    "pilot": {"per_label": 50, "subject_balanced": False},
    "main": {"per_subject_per_label": 100, "subject_balanced": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reusable subject-independent LLM evaluation subsets from "
            "precomputed input caches."
        )
    )
    parser.add_argument("-dataset", default="all", choices=["all", *sorted(DATASET_REGISTRY)])
    parser.add_argument(
        "-Input",
        nargs="*",
        default=["all"],
        help="Input cache(s) to subset. Use all for every official input.",
    )
    parser.add_argument("--input-cache-dir", default="Processed")
    parser.add_argument("--output-dir", default="Processed/LLMSubsets")
    parser.add_argument("--labels", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--test-subjects", nargs="*", help="Override unseen evaluation subjects.")
    parser.add_argument("--debug-per-label", type=int, default=3)
    parser.add_argument("--pilot-per-label", type=int, default=50)
    parser.add_argument("--main-per-subject-per-label", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = sorted(DATASET_REGISTRY) if args.dataset == "all" else [args.dataset]
    inputs = INPUT_TYPES if args.Input == ["all"] else args.Input
    reports = []
    for dataset in datasets:
        for input_name in inputs:
            reports.extend(create_subsets_for_input(dataset, input_name, args))

    print("\nLLM subset generation complete.")
    for report in reports:
        print(
            f"- {report['dataset']} {report['input_type']} {report['subset_name']}: "
            f"{report['sample_count']} samples labels={report['label_distribution']} "
            f"subjects={report['subjects']}"
        )


def create_subsets_for_input(dataset: str, input_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    samples, source_metadata = load_input_cache(dataset, input_name, Path(args.input_cache_dir))
    labels = [int(label) for label in args.labels]
    test_subjects = resolve_test_subjects(dataset, args)
    unseen_samples = [
        sample
        for sample in samples
        if str(getattr(sample, "subject", "")) in test_subjects
        and int(getattr(sample, "label")) in labels
    ]
    if not unseen_samples:
        raise RuntimeError(
            f"No unseen evaluation samples found for {dataset} {input_name}. "
            f"Test subjects: {sorted(test_subjects)}"
        )

    specs = {
        "debug": {"per_label": args.debug_per_label, "subject_balanced": False},
        "pilot": {"per_label": args.pilot_per_label, "subject_balanced": False},
        "main": {
            "per_subject_per_label": args.main_per_subject_per_label,
            "subject_balanced": True,
        },
    }
    reports = []
    for subset_name, spec in specs.items():
        rng = random.Random(args.random_state)
        if spec["subject_balanced"]:
            subset = sample_subject_label_balanced(
                unseen_samples,
                labels=labels,
                subjects=sorted(test_subjects),
                per_subject_per_label=int(spec["per_subject_per_label"]),
                rng=rng,
            )
        else:
            subset = sample_label_balanced(
                unseen_samples,
                labels=labels,
                per_label=int(spec["per_label"]),
                rng=rng,
            )
        reports.append(
            save_subset_cache(
                dataset=dataset,
                input_name=input_name,
                subset_name=subset_name,
                samples=subset,
                source_metadata=source_metadata,
                test_subjects=sorted(test_subjects),
                args=args,
            )
        )
    return reports


def load_input_cache(dataset: str, input_name: str, input_cache_dir: Path):
    path = input_cache_dir / f"{dataset}_{input_name}_samples.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Input cache not found: {path}. Run preprocess_inputs.py first.")
    with path.open("rb") as f:
        payload = pickle.load(f)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"Input cache must contain a list: {path}")
    return samples, metadata


def resolve_test_subjects(dataset: str, args: argparse.Namespace) -> set[str]:
    if args.test_subjects:
        return {str(subject) for subject in args.test_subjects}
    config = get_dataset_config(dataset)
    subjects = config.get("test_subjects")
    if not subjects:
        raise ValueError(
            f"No default test_subjects configured for {dataset}. "
            "Pass --test-subjects explicitly."
        )
    return {str(subject) for subject in subjects}


def sample_label_balanced(samples, labels: list[int], per_label: int, rng: random.Random):
    groups = group_by_label(samples)
    available = [len(groups.get(int(label), [])) for label in labels]
    if not available or min(available) < 1:
        raise RuntimeError("Cannot build label-balanced subset; at least one label group is empty.")
    effective = min(per_label, min(available))
    selected = []
    for label in labels:
        label_samples = list(groups.get(int(label), []))
        rng.shuffle(label_samples)
        selected.extend(label_samples[:effective])
    rng.shuffle(selected)
    return selected


def sample_subject_label_balanced(
    samples,
    labels: list[int],
    subjects: list[str],
    per_subject_per_label: int,
    rng: random.Random,
):
    groups = group_by_subject_label(samples)
    available = []
    for subject in subjects:
        for label in labels:
            available.append(len(groups.get(subject, {}).get(int(label), [])))
    if not available:
        return []
    effective = min(per_subject_per_label, min(available))
    if effective < 1:
        raise RuntimeError("Cannot build subject-label balanced subset; at least one group is empty.")

    selected = []
    for subject in subjects:
        for label in labels:
            group = list(groups[subject][int(label)])
            rng.shuffle(group)
            selected.extend(group[:effective])
    rng.shuffle(selected)
    return selected


def group_by_label(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[int(sample.label)].append(sample)
    return groups


def group_by_subject_label(samples):
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        groups[str(sample.subject)][int(sample.label)].append(sample)
    return groups


def label_distribution(samples) -> dict[str, int]:
    counts = Counter(int(sample.label) for sample in samples)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def subject_label_distribution(samples) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        nested[str(sample.subject)][int(sample.label)] += 1
    return {
        subject: {str(label): int(count) for label, count in sorted(counts.items())}
        for subject, counts in sorted(nested.items())
    }


def save_subset_cache(
    dataset: str,
    input_name: str,
    subset_name: str,
    samples,
    source_metadata: dict[str, Any],
    test_subjects: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir) / dataset / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset}_{input_name}_{subset_name}_samples.pkl"
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Subset cache exists: {path}. Use --overwrite.")

    metadata = {
        "dataset": dataset,
        "input_type": input_name,
        "subset_name": subset_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_protocol": "subject_independent_unseen_user_subset",
        "test_subjects": test_subjects,
        "labels": [int(label) for label in args.labels],
        "sample_count": len(samples),
        "label_distribution": label_distribution(samples),
        "subject_label_distribution": subject_label_distribution(samples),
        "source_input_cache_metadata": source_metadata,
        "format": "LLMSample subset pickle",
    }
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump({"metadata": metadata, "samples": samples}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    json_path = path.with_suffix(".json")
    json_tmp_path = json_path.with_name(json_path.name + ".tmp")
    json_tmp_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    json_tmp_path.replace(json_path)
    return {
        "dataset": dataset,
        "input_type": input_name,
        "subset_name": subset_name,
        "path": str(path),
        "sample_count": len(samples),
        "label_distribution": metadata["label_distribution"],
        "subjects": test_subjects,
    }


if __name__ == "__main__":
    main()
