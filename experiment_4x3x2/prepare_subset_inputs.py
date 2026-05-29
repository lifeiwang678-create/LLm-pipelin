from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from Dataset import DATASET_REGISTRY
from Input import INPUT_REGISTRY, build_input_provider


OFFICIAL_INPUTS = ["raw_data", "feature_description", "encoded_time_series", "extra_knowledge"]
SUBSET_LEVELS = ["debug", "pilot", "main"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build LLM input caches from fixed SensorSample data subsets created by "
            "prepare_data_subsets.py."
        )
    )
    parser.add_argument("-dataset", default="all", choices=["all", *sorted(DATASET_REGISTRY)])
    parser.add_argument(
        "-Input",
        nargs="*",
        default=["all"],
        choices=["all", *sorted(INPUT_REGISTRY)],
        help="Input cache(s) to build from each data subset.",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        default=SUBSET_LEVELS,
        choices=SUBSET_LEVELS,
        help="Subset level(s) to transform.",
    )
    parser.add_argument("--data-subset-dir", default="Processed/DataSubsets")
    parser.add_argument("--output-dir", default="Processed/LLMSubsets")
    parser.add_argument("--knowledge-file", help="Optional external knowledge file for extra_knowledge.")
    parser.add_argument("--knowledge-text", default="", help="Optional inline knowledge for extra_knowledge.")
    parser.add_argument(
        "--knowledge-mode",
        choices=["default", "append", "replace"],
        help="How extra_knowledge uses external knowledge.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = sorted(DATASET_REGISTRY) if args.dataset == "all" else [args.dataset]
    inputs = OFFICIAL_INPUTS if args.Input == ["all"] else args.Input
    reports = []
    for dataset in datasets:
        for subset_name in args.subset:
            samples, subset_metadata = load_data_subset(dataset, subset_name, Path(args.data_subset_dir))
            for input_name in inputs:
                reports.append(transform_subset(dataset, subset_name, input_name, samples, subset_metadata, args))

    print("\nSubset input-cache generation complete.")
    for report in reports:
        print(
            f"- {report['dataset']} {report['subset_name']} {report['input_type']}: "
            f"{report['sample_count']} samples path={report['path']}"
        )


def load_data_subset(dataset: str, subset_name: str, root: Path) -> tuple[list, dict[str, Any]]:
    path = root / dataset / subset_name / f"{dataset}_{subset_name}_windows.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Data subset not found: {path}. Run prepare_data_subsets.py first."
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"Data subset must contain a list of SensorSample objects: {path}")
    return samples, metadata


def transform_subset(
    dataset: str,
    subset_name: str,
    input_name: str,
    samples: list,
    subset_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    provider = build_input_provider(build_input_config(dataset, input_name, args))
    output_dir = Path(args.output_dir) / dataset / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset}_{provider.name}_{subset_name}_samples.pkl"
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Input subset cache exists: {path}. Use --overwrite.")

    started = time.perf_counter()
    llm_samples = provider.transform_all(samples)
    elapsed = round(time.perf_counter() - started, 3)
    metadata = {
        "dataset": dataset,
        "input_type": provider.name,
        "subset_name": subset_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_level": "fixed_preprocessed_data_subset",
        "source_data_subset_metadata": subset_metadata,
        "labels": subset_metadata.get("labels"),
        "subjects": subset_metadata.get("subjects"),
        "sample_count": len(llm_samples),
        "label_distribution": label_distribution(llm_samples),
        "subject_label_distribution": subject_label_distribution(llm_samples),
        "elapsed_time_sec": elapsed,
        "format": "LLMSample subset pickle",
    }
    if provider.name == "extra_knowledge":
        metadata["knowledge_file"] = args.knowledge_file
        metadata["knowledge_mode"] = args.knowledge_mode
        metadata["knowledge_text_provided"] = bool(args.knowledge_text)

    write_pickle_atomic(path, {"metadata": metadata, "samples": llm_samples})
    write_json_atomic(path.with_suffix(".json"), metadata)
    return {
        "dataset": dataset,
        "subset_name": subset_name,
        "input_type": provider.name,
        "path": str(path),
        "sample_count": len(llm_samples),
    }


def build_input_config(dataset: str, input_name: str, args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {"type": input_name, "dataset": dataset}
    if input_name == "extra_knowledge":
        config.update(
            {
                "knowledge_file": args.knowledge_file,
                "knowledge_text": args.knowledge_text,
                "knowledge_mode": args.knowledge_mode,
            }
        )
    return config


def label_distribution(samples: list) -> dict[str, int]:
    counts = Counter(int(sample.label) for sample in samples)
    return {str(label): int(counts[label]) for label in sorted(counts)}


def subject_label_distribution(samples: list) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        nested[str(sample.subject)][int(sample.label)] += 1
    return {
        subject: {str(label): int(count) for label, count in sorted(counts.items())}
        for subject, counts in sorted(nested.items())
    }


def write_pickle_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    main()
