from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from Dataset import DATASET_REGISTRY, build_dataset_loader, get_dataset_config
from Input import INPUT_REGISTRY, build_input_provider


OFFICIAL_INPUTS = [
    "raw_data",
    "feature_description",
    "encoded_time_series",
    "extra_knowledge",
]


def configure_text_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute input_text caches from processed SensorSample windows. "
            "This is the second cache layer for the 4x3x2 framework."
        )
    )
    parser.add_argument(
        "-dataset",
        required=True,
        choices=sorted(DATASET_REGISTRY),
        help="Dataset name.",
    )
    parser.add_argument(
        "-Input",
        default="all",
        choices=["all", *sorted(INPUT_REGISTRY)],
        help="Input representation to precompute. Use all for the four official inputs.",
    )
    parser.add_argument(
        "--processed-dir",
        default="Processed",
        help="Directory containing <DATASET>_binary_windows.pkl.",
    )
    parser.add_argument(
        "--processed-file",
        help="Optional explicit processed SensorSample .pkl file.",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help=(
            "Build input caches directly from raw dataset files, loading one subject "
            "at a time. Recommended for full WESAD because raw window caches are huge."
        ),
    )
    parser.add_argument(
        "--data-dir",
        help="Optional dataset directory override for --from-raw.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row limit for large CSV dataset loaders such as HHAR when using --from-raw.",
    )
    parser.add_argument(
        "--input-cache-dir",
        default="Processed",
        help="Output directory for <DATASET>_<INPUT>_samples.pkl files.",
    )
    parser.add_argument("--subjects", nargs="*", help="Optional subject IDs to precompute.")
    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Labels to keep. Defaults to 0 1.",
    )
    parser.add_argument("--knowledge-file", help="Optional external knowledge file for extra_knowledge.")
    parser.add_argument("--knowledge-text", default="", help="Optional inline knowledge for extra_knowledge.")
    parser.add_argument(
        "--knowledge-mode",
        choices=["default", "append", "replace"],
        help="How extra_knowledge uses external knowledge.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing input cache files.")
    return parser.parse_args()


def processed_path(args: argparse.Namespace) -> Path:
    if args.processed_file:
        return Path(args.processed_file)
    return Path(args.processed_dir) / f"{args.dataset}_binary_windows.pkl"


def processed_manifest_path(args: argparse.Namespace) -> Path:
    return Path(args.processed_dir) / f"{args.dataset}_binary_windows_manifest.json"


def load_processed_samples(args: argparse.Namespace) -> tuple[list, dict[str, Any]]:
    path = processed_path(args)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed cache not found: {path}. Run preprocess_datasets.py first."
        )
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except EOFError as exc:
        raise RuntimeError(
            f"Processed cache appears incomplete or corrupted: {path}. "
            "This usually happens when preprocessing was interrupted while writing the .pkl file. "
            "Delete this cache and rerun preprocess_datasets.py for the dataset."
        ) from exc
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"Processed cache must contain a list of SensorSample objects: {path}")
    return samples, metadata


def processed_sources(args: argparse.Namespace) -> tuple[list[Path], dict[str, Any]]:
    path = processed_path(args)
    if path.exists():
        return [path], {"source_type": "single_file", "path": str(path)}

    manifest_path = processed_manifest_path(args)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed cache not found: {path}. "
            f"Processed manifest not found: {manifest_path}. "
            "Run preprocess_datasets.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_paths = [Path(item["path"]) for item in manifest.get("shards", []) if item.get("path")]
    if not shard_paths:
        raise ValueError(f"Processed manifest contains no shard paths: {manifest_path}")
    missing = [str(shard_path) for shard_path in shard_paths if not shard_path.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed manifest references missing shard file(s): "
            + ", ".join(missing[:5])
        )
    manifest["source_type"] = "subject_shards"
    manifest["manifest_path"] = str(manifest_path)
    return shard_paths, manifest


def load_processed_samples_from_path(path: Path) -> tuple[list, dict[str, Any]]:
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except EOFError as exc:
        raise RuntimeError(
            f"Processed cache appears incomplete or corrupted: {path}. "
            "Delete this cache and rerun preprocess_datasets.py for the dataset."
        ) from exc
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"Processed cache must contain a list of SensorSample objects: {path}")
    return samples, metadata


def filter_samples(samples: list, subjects: list[str] | None, labels: list[int]) -> list:
    subject_filter = {str(subject) for subject in subjects} if subjects else None
    label_filter = {int(label) for label in labels}
    filtered = []
    for sample in samples:
        if subject_filter and str(sample.subject) not in subject_filter:
            continue
        if int(sample.label) not in label_filter:
            continue
        filtered.append(sample)
    return filtered


def input_names(args: argparse.Namespace) -> list[str]:
    if args.Input == "all":
        return OFFICIAL_INPUTS
    return [args.Input]


def build_input_config(args: argparse.Namespace, input_name: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "type": input_name,
        "dataset": args.dataset,
    }
    if input_name == "extra_knowledge":
        config.update(
            {
                "knowledge_file": args.knowledge_file,
                "knowledge_text": args.knowledge_text,
                "knowledge_mode": args.knowledge_mode,
            }
        )
    return config


def build_loader_config(args: argparse.Namespace) -> dict[str, Any]:
    dataset_defaults = get_dataset_config(args.dataset)
    config: dict[str, Any] = {
        "name": args.dataset,
        "data_dir": args.data_dir or dataset_defaults["data_dir"],
        "loader_kwargs": dict(dataset_defaults.get("loader_kwargs", {})),
    }
    if args.max_rows is not None:
        config["max_rows"] = args.max_rows
    return config


def output_path(args: argparse.Namespace, dataset: str, provider_name: str) -> Path:
    return Path(args.input_cache_dir) / f"{dataset}_{provider_name}_samples.pkl"


def label_distribution(samples: list) -> dict[int, int]:
    return dict(sorted(Counter(int(sample.label) for sample in samples).items()))


def subject_distribution(samples: list) -> dict[str, dict[int, int]]:
    nested: dict[str, Counter] = {}
    for sample in samples:
        nested.setdefault(str(sample.subject), Counter())[int(sample.label)] += 1
    return {
        subject: dict(sorted(counts.items()))
        for subject, counts in sorted(nested.items())
    }


def save_input_cache(
    args: argparse.Namespace,
    provider_name: str,
    llm_samples: list,
    source_metadata: dict,
    elapsed: float,
) -> dict:
    path = output_path(args, args.dataset, provider_name)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Input cache already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": args.dataset,
        "input_type": provider_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_processed_metadata": source_metadata,
        "subjects": args.subjects,
        "labels": [int(label) for label in args.labels],
        "sample_count": len(llm_samples),
        "label_distribution": label_distribution(llm_samples),
        "subject_label_distribution": subject_distribution(llm_samples),
        "elapsed_time_sec": elapsed,
        "format": "LLMSample list pickle",
    }
    if provider_name == "extra_knowledge":
        metadata["knowledge_file"] = args.knowledge_file
        metadata["knowledge_mode"] = args.knowledge_mode
        metadata["knowledge_text_provided"] = bool(args.knowledge_text)

    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump({"metadata": metadata, "samples": llm_samples}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)

    sidecar_path = path.with_suffix(".json")
    sidecar_tmp_path = sidecar_path.with_name(sidecar_path.name + ".tmp")
    sidecar_tmp_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    sidecar_tmp_path.replace(sidecar_path)

    print(f"Saved: {path}")
    print(f"Sample count: {len(llm_samples)}")
    print(f"Label distribution: {metadata['label_distribution']}")
    print(f"Elapsed: {elapsed} sec")
    return metadata


def precompute_input(args: argparse.Namespace, sensor_samples: list, input_name: str, source_metadata: dict) -> dict:
    provider = build_input_provider(build_input_config(args, input_name))

    started = time.perf_counter()
    print("\n" + "=" * 72)
    print(f"Precomputing input cache: {args.dataset} | {provider.name}")
    llm_samples = provider.transform_all(sensor_samples)
    elapsed = round(time.perf_counter() - started, 3)
    return save_input_cache(args, provider.name, llm_samples, source_metadata, elapsed)


def precompute_input_from_sources(
    args: argparse.Namespace,
    source_paths: list[Path],
    input_name: str,
    source_metadata: dict,
) -> dict:
    provider = build_input_provider(build_input_config(args, input_name))
    path = output_path(args, args.dataset, provider.name)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Input cache already exists: {path}. Use --overwrite to replace it.")

    started = time.perf_counter()
    print("\n" + "=" * 72)
    print(f"Precomputing input cache: {args.dataset} | {provider.name}")
    if source_metadata.get("source_type") == "subject_shards":
        print(f"Reading processed subject shards: {len(source_paths)}")

    llm_samples = []
    for source_path in source_paths:
        sensor_samples, shard_metadata = load_processed_samples_from_path(source_path)
        sensor_samples = filter_samples(sensor_samples, args.subjects, args.labels)
        if not sensor_samples:
            print(f"[skip] No selected samples in {source_path}")
            continue
        llm_samples.extend(provider.transform_all(sensor_samples))
        print(
            f"Processed source: {source_path.name} "
            f"selected={len(sensor_samples)} total_cached={len(llm_samples)}"
        )
        del sensor_samples
        del shard_metadata
        gc.collect()

    if not llm_samples:
        raise RuntimeError("No processed samples selected for input-cache preprocessing.")

    elapsed = round(time.perf_counter() - started, 3)
    return save_input_cache(args, provider.name, llm_samples, source_metadata, elapsed)


def precompute_inputs_from_raw(args: argparse.Namespace) -> list[dict]:
    loader_config = build_loader_config(args)
    loader = build_dataset_loader(loader_config)
    subjects = list(args.subjects or _discover_subjects_for_loader(loader))
    if not subjects:
        raise RuntimeError(
            f"Cannot determine subjects for {args.dataset}. Pass --subjects explicitly."
        )
    providers = [
        build_input_provider(build_input_config(args, input_name))
        for input_name in input_names(args)
    ]
    by_provider = {provider.name: [] for provider in providers}

    started = time.perf_counter()
    print("\n" + "=" * 72)
    print(f"Precomputing input cache directly from raw data: {args.dataset}")
    print(f"Data dir: {loader_config.get('data_dir')}")
    print(f"Subjects: {subjects}")
    print(f"Inputs: {', '.join(provider.name for provider in providers)}")

    for subject in subjects:
        sensor_samples = loader.load([subject], args.labels)
        sensor_samples = filter_samples(sensor_samples, [subject], args.labels)
        if not sensor_samples:
            print(f"[skip] No selected samples for subject {subject}")
            continue
        print(
            f"Loaded subject {subject}: {len(sensor_samples)} "
            f"labels={label_distribution(sensor_samples)}"
        )
        for provider in providers:
            by_provider[provider.name].extend(provider.transform_all(sensor_samples))
            print(
                f"  transformed {provider.name}: "
                f"total_cached={len(by_provider[provider.name])}"
            )
        del sensor_samples
        gc.collect()

    elapsed = round(time.perf_counter() - started, 3)
    source_metadata = {
        "source_type": "raw_subject_stream",
        "dataset": args.dataset,
        "data_dir": str(loader_config.get("data_dir")),
        "loader_kwargs": loader_config.get("loader_kwargs", {}),
        "subjects": subjects,
    }
    reports = []
    for provider in providers:
        samples = by_provider[provider.name]
        if not samples:
            raise RuntimeError(f"No samples generated for input cache: {provider.name}")
        reports.append(save_input_cache(args, provider.name, samples, source_metadata, elapsed))
    return reports


def _discover_subjects_for_loader(loader) -> list[str]:
    discover = getattr(loader, "_discover_subjects", None)
    if callable(discover):
        return list(discover())
    return []


def main() -> None:
    configure_text_output()
    args = parse_args()
    if args.from_raw:
        reports = precompute_inputs_from_raw(args)
        print("\nInput-cache preprocessing complete.")
        for report in reports:
            print(
                f"- {report['dataset']} {report['input_type']}: "
                f"{report['sample_count']} samples"
            )
        return

    source_paths, source_metadata = processed_sources(args)
    if len(source_paths) == 1:
        samples, source_metadata = load_processed_samples_from_path(source_paths[0])
        samples = filter_samples(samples, args.subjects, args.labels)
        if not samples:
            raise RuntimeError("No processed samples selected for input-cache preprocessing.")
        print(
            f"Loaded processed samples: {len(samples)} "
            f"labels={label_distribution(samples)}"
        )
        reports = [precompute_input(args, samples, name, source_metadata) for name in input_names(args)]
    else:
        print(
            f"Loaded processed manifest: {source_metadata.get('manifest_path')} "
            f"shards={len(source_paths)}"
        )
        reports = [
            precompute_input_from_sources(args, source_paths, name, source_metadata)
            for name in input_names(args)
        ]
    print("\nInput-cache preprocessing complete.")
    for report in reports:
        print(
            f"- {report['dataset']} {report['input_type']}: "
            f"{report['sample_count']} samples"
        )


if __name__ == "__main__":
    main()
