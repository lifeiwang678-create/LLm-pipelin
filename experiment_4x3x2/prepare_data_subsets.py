from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from Dataset import DATASET_REGISTRY


SUBSET_SPECS = {
    "WESAD": {
        "strategy": "label_balanced",
        "per_label": 160,
        "subject_source": "all",
        "description": "160 samples per class from preprocessed WESAD windows.",
    },
    "HHAR": {
        "strategy": "subject_label_balanced",
        "subject_count": 9,
        "per_subject_per_label": 50,
        "subject_source": "all",
        "description": "9 users, 50 samples per user per class from preprocessed HHAR windows.",
    },
    "DREAMT": {
        "strategy": "subject_label_balanced",
        "subject_count": 100,
        "per_subject_per_label": 5,
        "subject_source": "all",
        "description": "100 subjects, 5 samples per subject per class from preprocessed DREAMT windows.",
    },
}

SUBSET_LEVEL_SPECS = {
    "debug": {
        "strategy": "label_balanced",
        "per_label": 3,
        "description": "3 samples per class for smoke/debug runs.",
    },
    "pilot": {
        "strategy": "label_balanced",
        "per_label": 50,
        "description": "50 samples per class for pilot/cost profiling runs.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample fixed, reproducible main SensorSample data subsets "
            "from preprocessed dataset windows."
        )
    )
    parser.add_argument(
        "-dataset",
        default="all",
        choices=["all", *sorted(DATASET_REGISTRY)],
    )
    parser.add_argument("--processed-dir", default="Processed")
    parser.add_argument(
        "--processed-file",
        help="Optional explicit processed .pkl file for one dataset.",
    )
    parser.add_argument("--output-dir", default="Processed/DataSubsets")
    parser.add_argument("--labels", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allow-shortage",
        action="store_true",
        help="Clip to available data instead of failing when the requested subset is unavailable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.processed_file and args.dataset == "all":
        raise ValueError("--processed-file can only be used with one -dataset.")

    datasets = sorted(DATASET_REGISTRY) if args.dataset == "all" else [args.dataset]

    reports = []
    for dataset in datasets:
        manifest_path = Path(args.processed_dir) / f"{dataset}_binary_windows_manifest.json"
        single_path = Path(args.processed_dir) / f"{dataset}_binary_windows.pkl"

        if not args.processed_file and manifest_path.exists() and not single_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            reports.extend(create_dataset_subsets_from_shards(dataset, manifest, args))
            continue

        samples, source_metadata = load_processed_samples(dataset, args)
        reports.extend(create_dataset_subsets(dataset, samples, source_metadata, args))

    print("\nData subset generation complete.")
    for report in reports:
        print(
            f"- {report['dataset']} {report['subset_name']}: "
            f"{report['sample_count']} samples labels={report['label_distribution']} "
            f"subjects={report['subject_count']} path={report['path']}"
        )


def load_processed_samples(
    dataset: str,
    args: argparse.Namespace,
) -> tuple[list, dict[str, Any]]:
    if args.processed_file:
        return load_processed_payload(Path(args.processed_file))

    processed_dir = Path(args.processed_dir)
    single_path = processed_dir / f"{dataset}_binary_windows.pkl"

    if single_path.exists():
        return load_processed_payload(single_path)

    manifest_path = processed_dir / f"{dataset}_binary_windows_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed cache not found: {single_path}. "
            f"Processed manifest not found: {manifest_path}. "
            f"Run preprocess_datasets.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    samples = []
    for item in manifest.get("shards", []):
        shard_path = Path(item["path"])
        shard_samples, _ = load_processed_payload(shard_path)
        samples.extend(shard_samples)

    if not samples:
        raise RuntimeError(f"Processed manifest contains no samples: {manifest_path}")

    manifest["source_type"] = "processed_subject_shards"
    manifest["manifest_path"] = str(manifest_path)

    return samples, manifest


def load_processed_payload(path: Path) -> tuple[list, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed cache not found: {path}")

    with path.open("rb") as f:
        payload = pickle.load(f)

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload

    if not isinstance(samples, list):
        raise ValueError(
            f"Processed cache must contain a list of SensorSample objects: {path}"
        )

    return samples, metadata


def create_dataset_subsets(
    dataset: str,
    samples: list,
    source_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    labels = [int(label) for label in args.labels]

    filtered = [
        sample
        for sample in samples
        if int(getattr(sample, "label")) in labels
    ]

    if not filtered:
        raise RuntimeError(f"No samples with labels {labels} found for {dataset}.")

    specs = subset_specs_for_dataset(dataset)

    reports = []
    subjects = subjects_with_all_labels(filtered, labels)

    for subset_name, spec in specs.items():
        subset, subset_subjects = build_subset(
            samples=filtered,
            labels=labels,
            candidate_subjects=subjects,
            spec=spec,
            random_state=int(args.random_state),
            allow_shortage=bool(args.allow_shortage),
        )

        second_subset, second_subjects = build_subset(
            samples=filtered,
            labels=labels,
            candidate_subjects=subjects,
            spec=spec,
            random_state=int(args.random_state),
            allow_shortage=bool(args.allow_shortage),
        )

        reproducibility = reproducibility_report(
            first_samples=subset,
            second_samples=second_subset,
            first_subjects=subset_subjects,
            second_subjects=second_subjects,
            seed=int(args.random_state),
        )

        if not reproducibility["passed"]:
            raise RuntimeError(
                f"Sampling reproducibility failed for {dataset} {subset_name}."
            )

        reports.append(
            save_data_subset(
                dataset=dataset,
                subset_name=subset_name,
                samples=subset,
                subset_subjects=subset_subjects,
                subset_spec=spec,
                reproducibility=reproducibility,
                source_metadata=source_metadata,
                args=args,
            )
        )

    return reports


def create_dataset_subsets_from_shards(
    dataset: str,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    labels = [int(label) for label in args.labels]

    shard_paths = [
        Path(item["path"])
        for item in manifest.get("shards", [])
        if item.get("path")
    ]

    if not shard_paths:
        raise RuntimeError(f"Processed manifest contains no shard paths for {dataset}.")

    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed manifest references missing shards: "
            + ", ".join(missing[:5])
        )

    source_metadata = dict(manifest)
    source_metadata["source_type"] = "processed_subject_shards"

    specs = subset_specs_for_dataset(dataset)

    reports = []
    samples = []
    for shard_path in sorted(shard_paths, key=lambda path: path.name):
        shard_samples, _ = load_processed_payload(shard_path)
        samples.extend(shard_samples)

    if not samples:
        raise RuntimeError(f"Processed manifest contains no samples for {dataset}.")

    for subset_name, spec in specs.items():
        filtered = [
            sample
            for sample in samples
            if int(getattr(sample, "label")) in labels
        ]
        subjects = subjects_with_all_labels(filtered, labels)

        subset, subset_subjects = build_subset(
            samples=filtered,
            labels=labels,
            candidate_subjects=subjects,
            spec=spec,
            random_state=int(args.random_state),
            allow_shortage=bool(args.allow_shortage),
        )

        second_subset, second_subjects = build_subset(
            samples=filtered,
            labels=labels,
            candidate_subjects=subjects,
            spec=spec,
            random_state=int(args.random_state),
            allow_shortage=bool(args.allow_shortage),
        )

        reproducibility = reproducibility_report(
            first_samples=subset,
            second_samples=second_subset,
            first_subjects=subset_subjects,
            second_subjects=second_subjects,
            seed=int(args.random_state),
        )

        if not reproducibility["passed"]:
            raise RuntimeError(
                f"Sampling reproducibility failed for {dataset} {subset_name}."
            )

        reports.append(
            save_data_subset(
                dataset=dataset,
                subset_name=subset_name,
                samples=subset,
                subset_subjects=subset_subjects,
                subset_spec=spec,
                reproducibility=reproducibility,
                source_metadata=source_metadata,
                args=args,
            )
        )

    return reports


def subset_specs_for_dataset(dataset: str) -> dict[str, dict[str, Any]]:
    specs = {
        name: dict(spec)
        for name, spec in SUBSET_LEVEL_SPECS.items()
    }
    specs["main"] = dict(SUBSET_SPECS[dataset])
    return specs


def build_subset(
    *,
    samples: list,
    labels: list[int],
    candidate_subjects: list[str],
    spec: dict[str, Any],
    random_state: int,
    allow_shortage: bool,
) -> tuple[list, list[str]]:
    strategy = str(spec.get("strategy") or "").strip().lower()

    if strategy == "subject_label_balanced":
        per_subject = int(spec["per_subject_per_label"])

        selected_subjects = select_subjects(
            samples=samples,
            labels=labels,
            candidate_subjects=candidate_subjects,
            per_subject_per_label=per_subject,
            subject_count=int(spec["subject_count"]),
            random_state=random_state,
            allow_shortage=allow_shortage,
        )

        subset = sample_subject_label_balanced(
            samples=samples,
            labels=labels,
            subjects=selected_subjects,
            per_subject_per_label=per_subject,
            random_state=random_state,
            allow_shortage=allow_shortage,
        )

        return subset, selected_subjects

    subset = sample_label_balanced(
        samples=samples,
        labels=labels,
        per_label=int(spec["per_label"]),
        random_state=random_state,
        allow_shortage=allow_shortage,
    )

    return subset, sorted(
        {str(sample.subject) for sample in subset},
        key=subject_sort_key,
    )


def select_subjects(
    *,
    samples: list,
    labels: list[int],
    candidate_subjects: list[str],
    per_subject_per_label: int,
    subject_count: int,
    random_state: int,
    allow_shortage: bool,
) -> list[str]:
    groups = group_by_subject_label(samples)
    minimum = 1 if allow_shortage else per_subject_per_label

    eligible = [
        subject
        for subject in candidate_subjects
        if all(
            len(groups.get(subject, {}).get(int(label), [])) >= minimum
            for label in labels
        )
    ]

    if len(eligible) < subject_count and not allow_shortage:
        raise RuntimeError(
            f"Need {subject_count} subjects with at least "
            f"{per_subject_per_label} samples per label, "
            f"but only {len(eligible)} are available."
        )

    eligible = sorted(
        eligible,
        key=lambda subject: stable_digest(
            random_state,
            "subject",
            subject,
        ),
    )

    return sorted(
        eligible[: min(subject_count, len(eligible))],
        key=subject_sort_key,
    )


def sample_label_balanced(
    *,
    samples: list,
    labels: list[int],
    per_label: int,
    random_state: int,
    allow_shortage: bool,
) -> list:
    groups = group_by_label(samples)

    available = {
        int(label): len(groups.get(int(label), []))
        for label in labels
    }

    if min(available.values()) < per_label and not allow_shortage:
        raise RuntimeError(
            f"Cannot build exact label-balanced subset with {per_label} per label. "
            f"Available: {available}. Use --allow-shortage to clip."
        )

    effective = min(per_label, min(available.values()))

    selected = []

    for label in labels:
        group = list(groups[int(label)])
        group = stable_sample_order(group, random_state, "label", int(label))
        selected.extend(group[:effective])

    selected = stable_sample_order(selected, random_state, "output")

    return selected


def sample_subject_label_balanced(
    *,
    samples: list,
    labels: list[int],
    subjects: list[str],
    per_subject_per_label: int,
    random_state: int,
    allow_shortage: bool,
) -> list:
    groups = group_by_subject_label(samples)

    availability = [
        len(groups.get(subject, {}).get(int(label), []))
        for subject in subjects
        for label in labels
    ]

    if not availability or min(availability) < 1:
        raise RuntimeError(
            "Cannot build subject-label balanced subset; at least one group is empty."
        )

    if min(availability) < per_subject_per_label and not allow_shortage:
        raise RuntimeError(
            f"Cannot build exact subject-label subset with "
            f"{per_subject_per_label} per subject per label."
        )

    effective = min(per_subject_per_label, min(availability))

    selected = []

    for subject in subjects:
        for label in labels:
            group = list(groups[subject][int(label)])
            group = stable_sample_order(
                group,
                random_state,
                "subject-label",
                subject,
                int(label),
            )
            selected.extend(group[:effective])

    selected = stable_sample_order(selected, random_state, "output")

    return selected


def save_data_subset(
    *,
    dataset: str,
    subset_name: str,
    samples: list,
    subset_subjects: list[str],
    subset_spec: dict[str, Any],
    reproducibility: dict[str, Any],
    source_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir) / dataset / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{dataset}_{subset_name}_windows.pkl"

    if path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Data subset already exists: {path}. Use --overwrite."
        )

    metadata = {
        "dataset": dataset,
        "subset_name": subset_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_level": "preprocessed_sensor_windows",
        "random_state": int(args.random_state),
        "labels": [int(label) for label in args.labels],
        "sample_count": len(samples),
        "label_distribution": label_distribution(samples),
        "subject_label_distribution": subject_label_distribution(samples),
        "subjects": subset_subjects,
        "subset_spec": subset_spec,
        "reproducibility_check": reproducibility,
        "source_processed_metadata": source_metadata,
        "format": "SensorSample data subset pickle",
    }

    write_pickle_atomic(path, {"metadata": metadata, "samples": samples})
    write_json_atomic(path.with_suffix(".json"), metadata)

    return {
        "dataset": dataset,
        "subset_name": subset_name,
        "path": str(path),
        "sample_count": len(samples),
        "label_distribution": metadata["label_distribution"],
        "subject_count": len(subset_subjects),
    }


def group_by_label(samples: list) -> dict[int, list]:
    groups = defaultdict(list)

    for sample in samples:
        groups[int(sample.label)].append(sample)

    return groups


def group_by_subject_label(samples: list) -> dict[str, dict[int, list]]:
    groups = defaultdict(lambda: defaultdict(list))

    for sample in samples:
        groups[str(sample.subject)][int(sample.label)].append(sample)

    return groups


def subjects_with_all_labels(samples: list, labels: list[int]) -> list[str]:
    groups = group_by_subject_label(samples)

    return sorted(
        [
            subject
            for subject, by_label in groups.items()
            if all(
                len(by_label.get(int(label), [])) > 0
                for label in labels
            )
        ],
        key=subject_sort_key,
    )


def label_distribution(samples: list) -> dict[str, int]:
    counts = Counter(int(sample.label) for sample in samples)

    return {
        str(label): int(counts[label])
        for label in sorted(counts)
    }


def subject_label_distribution(samples: list) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter] = defaultdict(Counter)

    for sample in samples:
        nested[str(sample.subject)][int(sample.label)] += 1

    return {
        subject: {
            str(label): int(count)
            for label, count in sorted(counts.items())
        }
        for subject, counts in sorted(
            nested.items(),
            key=lambda item: subject_sort_key(item[0]),
        )
    }


def reproducibility_report(
    *,
    first_samples: list,
    second_samples: list,
    first_subjects: list[str],
    second_subjects: list[str],
    seed: int,
) -> dict[str, Any]:
    first_ids = [sample_fingerprint(sample) for sample in first_samples]
    second_ids = [sample_fingerprint(sample) for sample in second_samples]

    digest = hashlib.sha256(
        "\n".join(first_ids).encode("utf-8")
    ).hexdigest()

    return {
        "seed": int(seed),
        "passed": first_ids == second_ids and first_subjects == second_subjects,
        "sample_count": len(first_ids),
        "sample_ids_sha256": digest,
        "subject_count": len(first_subjects),
        "subjects": first_subjects,
    }


def sample_fingerprint(sample) -> str:
    meta = dict(getattr(sample, "meta", {}) or {})

    keys = (
        "sample_id",
        "data_index",
        "epoch_id",
        "local_index",
        "start_index",
        "end_index",
        "window_start",
        "window_end",
        "window_start_sec",
        "window_end_sec",
    )

    parts = {
        "dataset": getattr(sample, "dataset", ""),
        "subject": str(getattr(sample, "subject", "")),
        "label": int(getattr(sample, "label")),
    }

    for key in keys:
        if key in meta:
            parts[key] = meta[key]

    if len(parts) <= 3:
        parts["signals_sha256"] = hashlib.sha256(
            repr(getattr(sample, "signals", {})).encode("utf-8")
        ).hexdigest()

    return json.dumps(
        parts,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def subject_sort_key(subject: str) -> list[object]:
    parts = re.split(r"(\d+)", str(subject))

    return [
        int(part) if part.isdigit() else part.lower()
        for part in parts
    ]


def stable_seed(random_state: int, *parts: object) -> int:
    payload = ":".join(
        [str(int(random_state)), *(str(part) for part in parts)]
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    )


def stable_digest(random_state: int, *parts: object) -> str:
    payload = ":".join(
        [str(int(random_state)), *(str(part) for part in parts)]
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def stable_sample_order(samples: list, random_state: int, *parts: object) -> list:
    return sorted(
        samples,
        key=lambda sample: stable_digest(
            random_state,
            *parts,
            sample_fingerprint(sample),
        ),
    )


def write_pickle_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")

    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")

    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tmp_path.replace(path)


if __name__ == "__main__":
    main()
