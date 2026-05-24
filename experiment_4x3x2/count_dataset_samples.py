from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Dataset import DATASET_REGISTRY, build_dataset_loader
from Dataset.dreamt_loader import mode_label
from Dataset.hhar_loader import split_by_time_gap


def configure_text_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count binary window samples for the 4x3x2 datasets without "
            "building prompts or calling the LLM."
        )
    )
    parser.add_argument(
        "-dataset",
        default="all",
        choices=["all", *sorted(DATASET_REGISTRY)],
        help="Dataset to count. Use all to count WESAD, HHAR, and DREAMT.",
    )
    parser.add_argument("--data-dir", help="Dataset directory override for a single -dataset run.")
    parser.add_argument("--wesad-data-dir", help="WESAD directory override for -dataset all.")
    parser.add_argument("--hhar-data-dir", help="HHAR directory override for -dataset all.")
    parser.add_argument("--dreamt-data-dir", help="DREAMT directory override for -dataset all.")
    parser.add_argument("--subjects", nargs="*", help="Optional subject IDs to count.")
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
        help="Optional row limit for large CSV loaders such as HHAR.",
    )
    parser.add_argument(
        "--save-json",
        help="Optional path for saving the count report as JSON.",
    )
    return parser.parse_args()


def label_distribution(labels: list[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(label) for label in labels).items()))


def nested_int_dict(counter: dict[Any, Counter]) -> dict[str, dict[int, int]]:
    return {
        str(key): dict(sorted({int(label): int(count) for label, count in counts.items()}.items()))
        for key, counts in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def build_loader_config(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "name": dataset,
        "loader_kwargs": {},
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


def count_wesad(loader, subjects: list[str] | None, labels: list[int]) -> dict[str, Any]:
    subject_list = list(subjects or loader._discover_subjects())
    labels_to_keep = tuple(int(label) for label in labels)
    step_samples_chest = int(loader.stride_sec * 700)
    if step_samples_chest <= 0:
        raise ValueError("WESAD stride_sec is too small; step_samples_chest must be positive.")

    half_phys_chest = int(loader.physiology_window_sec * 700 / 2.0)
    by_subject: dict[str, Counter] = defaultdict(Counter)
    all_labels: list[int] = []
    skipped_subjects: list[str] = []

    for subject in subject_list:
        pkl_path = loader.data_dir / subject / f"{subject}.pkl"
        if not pkl_path.exists():
            skipped_subjects.append(subject)
            continue

        with pkl_path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        original_labels = np.asarray(data["label"]).ravel()
        mapped_labels = loader._map_label_series(original_labels)

        print(f"WESAD label distribution before binary mapping ({subject}): {loader._label_distribution(original_labels)}")
        print(
            f"WESAD label distribution after binary mapping ({subject}): "
            f"{loader._label_distribution(mapped_labels[mapped_labels >= 0])}"
        )

        for seg_start, seg_end, label_value in loader._iter_contiguous_label_segments(
            mapped_labels,
            valid_labels=labels_to_keep,
        ):
            centers = range(
                seg_start + half_phys_chest,
                seg_end - half_phys_chest,
                step_samples_chest,
            )
            count = len(centers)
            if count <= 0:
                continue
            by_subject[subject][int(label_value)] += count
            all_labels.extend([int(label_value)] * count)

    return {
        "dataset": "WESAD",
        "subjects_requested": subject_list,
        "subjects_counted": sorted(by_subject),
        "skipped_subjects": skipped_subjects,
        "total_windows": len(all_labels),
        "label_distribution": label_distribution(all_labels),
        "subject_label_distribution": nested_int_dict(by_subject),
        "windowing": {
            "physiology_window_sec": loader.physiology_window_sec,
            "acc_window_sec": loader.acc_window_sec,
            "stride_sec": loader.stride_sec,
        },
    }


def count_hhar(loader, subjects: list[str] | None, labels: list[int]) -> dict[str, Any]:
    df = loader._load_clean_accelerometer()
    subject_filter = {str(subject) for subject in subjects} if subjects else None
    labels_to_keep = {int(label) for label in labels}
    by_subject: dict[str, Counter] = defaultdict(Counter)
    all_labels: list[int] = []

    group_cols = ["user_id", "model", "device", "activity_label"]
    for _, group in df.groupby(group_cols, sort=True):
        subject = str(group["user_id"].iloc[0])
        if subject_filter and subject not in subject_filter:
            continue

        group = split_by_time_gap(group, max_gap_sec=loader.max_gap_sec)
        for _, continuous_group in group.groupby("continuous_segment_id", sort=True):
            continuous_group = continuous_group.sort_values("time_sec").reset_index(drop=True)
            if continuous_group.empty:
                continue

            current_start = float(continuous_group["time_sec"].min())
            end_time = float(continuous_group["time_sec"].max())
            while current_start + loader.window_sec <= end_time:
                current_end = current_start + loader.window_sec
                window_df = continuous_group[
                    (continuous_group["time_sec"] >= current_start)
                    & (continuous_group["time_sec"] < current_end)
                ]
                if len(window_df) >= loader.min_samples_per_window:
                    label = int(window_df["label_int"].mode().iloc[0])
                    if label in labels_to_keep:
                        by_subject[subject][label] += 1
                        all_labels.append(label)
                current_start += loader.stride_sec

    return {
        "dataset": "HHAR",
        "subjects_requested": list(subjects or []),
        "subjects_counted": sorted(by_subject),
        "total_windows": len(all_labels),
        "label_distribution": label_distribution(all_labels),
        "subject_label_distribution": nested_int_dict(by_subject),
        "rows_after_binary_filter": int(len(df)),
        "windowing": {
            "window_size": loader.window_size,
            "stride_size": loader.stride_size,
            "sampling_rate": loader.sampling_rate,
            "window_sec": loader.window_sec,
            "stride_sec": loader.stride_sec,
            "max_rows": loader.max_rows,
        },
    }


def count_dreamt(loader, subjects: list[str] | None, labels: list[int]) -> dict[str, Any]:
    files = loader._select_files(subjects)
    if not files:
        raise FileNotFoundError(
            f"No DREAMT raw CSV files found under {loader.data_dir}. "
            "Expected files like data_64Hz/S099_whole_df.csv."
        )

    labels_to_keep = {int(label) for label in labels}
    by_subject: dict[str, Counter] = defaultdict(Counter)
    all_labels: list[int] = []
    skipped_files: list[str] = []
    window_sec = float(loader.window_size) / float(loader.sampling_rate)
    stride_sec = float(loader.stride_size) / float(loader.sampling_rate)
    if stride_sec <= 0:
        raise ValueError("DREAMT stride_sec must be positive.")

    for file_path in files:
        subject = file_path.stem.split("_")[0].upper()
        df = pd.read_csv(file_path)
        cols = loader._detect_columns(df)
        if cols["label"] is None:
            skipped_files.append(str(file_path))
            continue

        raw_distribution = df[cols["label"]].value_counts(dropna=False).to_dict()
        df = loader._prepare_dataframe(df, cols)
        mapped_distribution = (
            df["label_mapped"].dropna().astype(int).value_counts().sort_index().to_dict()
        )
        print(f"DREAMT label distribution before binary mapping ({subject}): {raw_distribution}")
        print(f"DREAMT label distribution after binary mapping ({subject}): {mapped_distribution}")

        df = df.sort_values("time_sec").reset_index(drop=True)
        if df.empty:
            continue

        current_start = float(df["time_sec"].iloc[0])
        end_time = float(df["time_sec"].iloc[-1])
        while current_start + window_sec <= end_time + 1e-9:
            current_end = current_start + window_sec
            group = df.loc[
                (df["time_sec"] >= current_start)
                & (df["time_sec"] < current_end)
            ]
            if len(group) >= loader.min_samples:
                label = mode_label(group["label_mapped"])
                if not pd.isna(label):
                    label = int(label)
                    if label in labels_to_keep:
                        artifact = loader._artifact_flag(group)
                        if not (loader.skip_artifact_epochs and artifact):
                            signals = loader._signals_from_epoch(group)
                            if signals:
                                by_subject[subject][label] += 1
                                all_labels.append(label)
            current_start += stride_sec

    return {
        "dataset": "DREAMT",
        "subjects_requested": list(subjects or []),
        "subjects_counted": sorted(by_subject),
        "skipped_files": skipped_files,
        "total_windows": len(all_labels),
        "label_distribution": label_distribution(all_labels),
        "subject_label_distribution": nested_int_dict(by_subject),
        "files_counted": len(files) - len(skipped_files),
        "windowing": {
            "sampling_rate": loader.sampling_rate,
            "window_size": loader.window_size,
            "stride_size": loader.stride_size,
            "epoch_seconds": loader.window_size / loader.sampling_rate,
            "stride_seconds": loader.stride_size / loader.sampling_rate,
            "min_epoch_fraction": loader.min_samples / loader.window_size,
            "skip_artifact_epochs": loader.skip_artifact_epochs,
        },
    }


def count_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    config = build_loader_config(dataset, args)
    loader = build_dataset_loader(config)
    subjects = args.subjects if args.dataset != "all" else None
    started = time.perf_counter()

    if dataset == "WESAD":
        report = count_wesad(loader, subjects, args.labels)
    elif dataset == "HHAR":
        report = count_hhar(loader, subjects, args.labels)
    elif dataset == "DREAMT":
        report = count_dreamt(loader, subjects, args.labels)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    report["elapsed_time_sec"] = round(time.perf_counter() - started, 3)
    return report


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"Dataset: {report['dataset']}")
    print(f"Total windows: {report['total_windows']}")
    print(f"Label distribution: {report['label_distribution']}")
    print(f"Subjects counted: {len(report.get('subjects_counted', []))}")
    if report.get("subjects_counted"):
        print(f"Subject IDs: {', '.join(report['subjects_counted'])}")
    if report.get("skipped_subjects"):
        print(f"Skipped subjects: {report['skipped_subjects']}")
    if report.get("skipped_files"):
        print(f"Skipped files: {len(report['skipped_files'])}")
    print(f"Windowing: {report.get('windowing', {})}")
    print(f"Elapsed: {report['elapsed_time_sec']} sec")

    print("Subject-level label distribution:")
    for subject, distribution in report.get("subject_label_distribution", {}).items():
        print(f"  {subject}: {distribution}")


def main() -> None:
    configure_text_output()
    args = parse_args()
    datasets = sorted(DATASET_REGISTRY) if args.dataset == "all" else [args.dataset]
    reports = []
    for dataset in datasets:
        reports.append(count_dataset(dataset, args))
        print_report(reports[-1])

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"labels": args.labels, "reports": reports}
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON report: {output_path}")


if __name__ == "__main__":
    main()
