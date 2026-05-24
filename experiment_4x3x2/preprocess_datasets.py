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


def configure_text_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess dataset windows once and save them as reusable "
            "SensorSample caches for the 4x3x2 experiment framework."
        )
    )
    parser.add_argument(
        "-dataset",
        default="all",
        choices=["all", *sorted(DATASET_REGISTRY)],
        help="Dataset to preprocess. Use all to preprocess WESAD, HHAR, and DREAMT.",
    )
    parser.add_argument("--data-dir", help="Dataset directory override for a single -dataset run.")
    parser.add_argument("--wesad-data-dir", help="WESAD directory override for -dataset all.")
    parser.add_argument("--hhar-data-dir", help="HHAR directory override for -dataset all.")
    parser.add_argument("--dreamt-data-dir", help="DREAMT directory override for -dataset all.")
    parser.add_argument("--subjects", nargs="*", help="Optional subject IDs to preprocess.")
    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Binary labels to keep. Defaults to 0 1.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row limit for large CSV dataset loaders such as HHAR.",
    )
    parser.add_argument(
        "--processed-dir",
        default="Processed",
        help="Output directory for processed cache files.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional explicit output .pkl path. Only valid when -dataset is not all.",
    )
    parser.add_argument(
        "--shard-by-subject",
        action="store_true",
        help=(
            "Save one processed cache file per subject plus a manifest. "
            "Recommended for full WESAD preprocessing because raw windows are large."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing processed cache file.",
    )
    return parser.parse_args()


def build_loader_config(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    dataset_defaults = get_dataset_config(dataset)
    config: dict[str, Any] = {
        "name": dataset,
        "data_dir": dataset_defaults["data_dir"],
        "loader_kwargs": dict(dataset_defaults.get("loader_kwargs", {})),
    }
    if args.data_dir and args.dataset != "all":
        config["data_dir"] = args.data_dir
    elif dataset == "WESAD" and args.wesad_data_dir:
        config["data_dir"] = args.wesad_data_dir
    elif dataset == "HHAR" and args.hhar_data_dir:
        config["data_dir"] = args.hhar_data_dir
    elif dataset == "DREAMT" and args.dreamt_data_dir:
        config["data_dir"] = args.dreamt_data_dir

    if dataset == "HHAR" and args.max_rows is not None:
        config["max_rows"] = args.max_rows
    return config


def output_path_for(dataset: str, args: argparse.Namespace) -> Path:
    if args.output_file:
        if args.dataset == "all":
            raise ValueError("--output-file can only be used with one dataset.")
        return Path(args.output_file)
    return Path(args.processed_dir) / f"{dataset}_binary_windows.pkl"


def shard_path_for(dataset: str, subject: str, args: argparse.Namespace) -> Path:
    return Path(args.processed_dir) / f"{dataset}_binary_windows_{subject}.pkl"


def manifest_path_for(dataset: str, args: argparse.Namespace) -> Path:
    return Path(args.processed_dir) / f"{dataset}_binary_windows_manifest.json"


def label_distribution(samples) -> dict[int, int]:
    counts = Counter(int(sample.label) for sample in samples)
    return dict(sorted(counts.items()))


def subject_distribution(samples) -> dict[str, dict[int, int]]:
    nested: dict[str, Counter] = {}
    for sample in samples:
        subject = str(sample.subject)
        nested.setdefault(subject, Counter())[int(sample.label)] += 1
    return {
        subject: dict(sorted(counts.items()))
        for subject, counts in sorted(nested.items())
    }


def preprocess_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.shard_by_subject:
        return preprocess_dataset_sharded(dataset, args)

    output_path = output_path_for(dataset, args)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Processed cache already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    config = build_loader_config(dataset, args)
    loader = build_dataset_loader(config)
    subjects = args.subjects if args.dataset != "all" else None
    started = time.perf_counter()
    print("\n" + "=" * 72)
    print(f"Preprocessing {dataset}")
    print(f"Data dir: {config.get('data_dir')}")
    print(f"Subjects: {subjects or 'all/default'}")

    samples = loader.load(subjects, args.labels)
    elapsed = round(time.perf_counter() - started, 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": dataset,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "labels": [int(label) for label in args.labels],
        "data_dir": str(config.get("data_dir")),
        "loader_kwargs": config.get("loader_kwargs", {}),
        "subjects": subjects,
        "sample_count": len(samples),
        "label_distribution": label_distribution(samples),
        "subject_label_distribution": subject_distribution(samples),
        "elapsed_time_sec": elapsed,
        "format": "SensorSample list pickle",
    }
    payload = {
        "metadata": metadata,
        "samples": samples,
    }
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(output_path)

    sidecar_path = output_path.with_suffix(".json")
    sidecar_tmp_path = sidecar_path.with_name(sidecar_path.name + ".tmp")
    sidecar_tmp_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    sidecar_tmp_path.replace(sidecar_path)

    print(f"Saved: {output_path}")
    print(f"Metadata: {sidecar_path}")
    print(f"Sample count: {len(samples)}")
    print(f"Label distribution: {metadata['label_distribution']}")
    print(f"Elapsed: {elapsed} sec")
    return metadata


def preprocess_dataset_sharded(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.output_file:
        raise ValueError("--output-file cannot be used with --shard-by-subject.")

    config = build_loader_config(dataset, args)
    loader = build_dataset_loader(config)
    subjects = list(args.subjects or _discover_subjects_for_loader(loader))
    if not subjects:
        raise RuntimeError(
            f"Cannot determine subjects for sharded preprocessing of {dataset}. "
            "Pass --subjects explicitly."
        )

    output_dir = Path(args.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(dataset, args)
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Processed manifest already exists: {manifest_path}. "
            "Use --overwrite to replace it."
        )

    print("\n" + "=" * 72)
    print(f"Preprocessing {dataset} by subject shards")
    print(f"Data dir: {config.get('data_dir')}")
    print(f"Subjects: {subjects}")

    started_all = time.perf_counter()
    shard_reports = []
    total_counts: Counter = Counter()
    total_samples = 0

    for subject in subjects:
        shard_path = shard_path_for(dataset, subject, args)
        if shard_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Processed shard already exists: {shard_path}. "
                "Use --overwrite to replace it."
            )

        started = time.perf_counter()
        samples = loader.load([subject], args.labels)
        elapsed = round(time.perf_counter() - started, 3)
        counts = label_distribution(samples)
        total_counts.update({int(label): int(count) for label, count in counts.items()})
        total_samples += len(samples)

        metadata = {
            "dataset": dataset,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "labels": [int(label) for label in args.labels],
            "data_dir": str(config.get("data_dir")),
            "loader_kwargs": config.get("loader_kwargs", {}),
            "subjects": [subject],
            "sample_count": len(samples),
            "label_distribution": counts,
            "subject_label_distribution": subject_distribution(samples),
            "elapsed_time_sec": elapsed,
            "format": "SensorSample list pickle",
            "shard_subject": subject,
        }
        _save_pickle_payload_atomic(shard_path, {"metadata": metadata, "samples": samples})
        _save_json_atomic(shard_path.with_suffix(".json"), metadata)

        shard_reports.append(
            {
                "subject": subject,
                "path": str(shard_path),
                "metadata_path": str(shard_path.with_suffix(".json")),
                "sample_count": len(samples),
                "label_distribution": counts,
                "elapsed_time_sec": elapsed,
            }
        )
        print(
            f"Saved shard: {shard_path} "
            f"samples={len(samples)} labels={counts} elapsed={elapsed} sec"
        )
        del samples
        gc.collect()

    elapsed_all = round(time.perf_counter() - started_all, 3)
    manifest = {
        "dataset": dataset,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "labels": [int(label) for label in args.labels],
        "data_dir": str(config.get("data_dir")),
        "loader_kwargs": config.get("loader_kwargs", {}),
        "subjects": subjects,
        "sample_count": total_samples,
        "label_distribution": dict(sorted(total_counts.items())),
        "elapsed_time_sec": elapsed_all,
        "format": "SensorSample subject-shard manifest",
        "shards": shard_reports,
    }
    _save_json_atomic(manifest_path, manifest)

    print(f"Manifest: {manifest_path}")
    print(f"Sample count: {total_samples}")
    print(f"Label distribution: {manifest['label_distribution']}")
    print(f"Elapsed: {elapsed_all} sec")
    return manifest


def _discover_subjects_for_loader(loader) -> list[str]:
    discover = getattr(loader, "_discover_subjects", None)
    if callable(discover):
        return list(discover())
    return []


def _save_pickle_payload_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    configure_text_output()
    args = parse_args()
    datasets = sorted(DATASET_REGISTRY) if args.dataset == "all" else [args.dataset]
    reports = [preprocess_dataset(dataset, args) for dataset in datasets]
    print("\nPreprocessing complete.")
    for report in reports:
        print(
            f"- {report['dataset']}: {report['sample_count']} samples, "
            f"labels={report['label_distribution']}"
        )


if __name__ == "__main__":
    main()
