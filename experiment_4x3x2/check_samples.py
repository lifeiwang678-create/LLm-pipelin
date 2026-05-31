from pathlib import Path
from collections import Counter
import pickle
import math
import os
import re

import pandas as pd


processed_root = Path(
    os.environ.get(
        "PROCESSED_ROOT",
        r"D:\DATA\experiment_4x3x2\Processed"
    )
)

targets = [
    processed_root / "DataSubsets",
    processed_root / "LLMSubsets",
]


def is_missing(value):
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    return False


def get_from_dict(d, keys):
    for key in keys:
        if key in d and not is_missing(d[key]):
            return d[key]
    return None


def get_from_object(obj, keys):
    for key in keys:
        if hasattr(obj, key):
            value = getattr(obj, key)
            if not is_missing(value):
                return value
    return None


def get_label(x):
    label_keys = [
        "label",
        "y",
        "target",
        "state",
        "class",
        "activity_label",
        "sleep_label",
        "stress_label",
    ]

    if isinstance(x, dict):
        return get_from_dict(x, label_keys)

    return get_from_object(x, label_keys)


def get_subject(x):
    subject_keys = [
        "subject",
        "subject_id",
        "user",
        "user_id",
        "participant",
        "participant_id",
    ]

    if isinstance(x, dict):
        return get_from_dict(x, subject_keys)

    return get_from_object(x, subject_keys)


def get_dataset_name(path):
    for name in ["WESAD", "HHAR", "DREAMT"]:
        if name in path.parts:
            return name
    return "UNKNOWN"


def get_level_name(path):
    for name in ["debug", "pilot", "main"]:
        if name in path.parts:
            return name
    return "UNKNOWN"


def get_subset_type(path):
    if "DataSubsets" in path.parts:
        return "DataSubsets"
    if "LLMSubsets" in path.parts:
        return "LLMSubsets"
    return "UNKNOWN"


def get_input_type(path):
    name = path.name

    input_types = [
        "raw_data",
        "feature_description",
        "encoded_time_series",
        "extra_knowledge",
    ]

    for input_type in input_types:
        if input_type in name:
            return input_type

    return "-"


def unwrap_pickle_object(obj):
    if isinstance(obj, dict):
        for key in ["samples", "data", "items", "windows", "records"]:
            if key in obj:
                return obj[key]

    return obj


def count_dataframe(path, df):
    labels = None
    subjects = None

    for col in ["label", "y", "target"]:
        if col in df.columns:
            labels = df[col].value_counts(dropna=False).to_dict()
            break

    for col in ["subject", "subject_id", "user", "user_id"]:
        if col in df.columns:
            subjects = df[col].nunique(dropna=True)
            break

    print(f"\n{path}")
    print(f"rows = {len(df)}")

    if labels is not None:
        print(f"label counts = {labels}")

    if subjects is not None:
        print(f"subjects = {subjects}")

    return {
        "subset_type": get_subset_type(path),
        "dataset": get_dataset_name(path),
        "level": get_level_name(path),
        "input_type": get_input_type(path),
        "file": path.name,
        "samples": len(df),
        "label_counts": labels,
        "subjects": subjects,
        "path": str(path),
    }


def count_sequence(path, seq):
    labels = Counter()
    subjects = Counter()

    for x in seq:
        labels[get_label(x)] += 1
        subjects[get_subject(x)] += 1

    non_missing_subjects = [s for s in subjects if s is not None]

    print(f"\n{path}")
    print(f"samples = {len(seq)}")
    print(f"label counts = {dict(labels)}")
    print(f"subjects = {len(non_missing_subjects)}")

    return {
        "subset_type": get_subset_type(path),
        "dataset": get_dataset_name(path),
        "level": get_level_name(path),
        "input_type": get_input_type(path),
        "file": path.name,
        "samples": len(seq),
        "label_counts": dict(labels),
        "subjects": len(non_missing_subjects),
        "path": str(path),
    }


def count_file(path):
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return count_dataframe(path, df)

    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)

        obj = unwrap_pickle_object(obj)

        if isinstance(obj, pd.DataFrame):
            return count_dataframe(path, obj)

        if isinstance(obj, (list, tuple)):
            return count_sequence(path, obj)

        print(f"\n{path}")
        print(f"type = {type(obj)}")

        try:
            sample_count = len(obj)
            print(f"len = {sample_count}")
        except Exception:
            sample_count = None
            print("len unavailable")

        return {
            "subset_type": get_subset_type(path),
            "dataset": get_dataset_name(path),
            "level": get_level_name(path),
            "input_type": get_input_type(path),
            "file": path.name,
            "samples": sample_count,
            "label_counts": None,
            "subjects": None,
            "path": str(path),
        }

    return None


def scan_folder(folder):
    print(f"\n==== {folder} ====")

    if not folder.exists():
        print("folder not found")
        return []

    results = []

    for path in sorted(folder.rglob("*")):
        if path.suffix in [".pkl", ".csv"]:
            info = count_file(path)
            if info is not None:
                results.append(info)

    return results


def print_summary_table(results):
    df = pd.DataFrame(results)

    if df.empty:
        print("\nNo files found.")
        return

    print("\n==== Summary: all subset files ====")
    summary_cols = [
        "subset_type",
        "dataset",
        "level",
        "input_type",
        "samples",
        "subjects",
        "file",
    ]

    print(
        df[summary_cols]
        .sort_values(["subset_type", "dataset", "level", "input_type", "file"])
        .to_string(index=False)
    )

    print("\n==== DataSubsets main unique sample counts ====")
    data_main = df[
        (df["subset_type"] == "DataSubsets")
        & (df["level"] == "main")
    ]

    print(
        data_main[["dataset", "samples", "subjects", "file"]]
        .sort_values(["dataset"])
        .to_string(index=False)
    )

    print("\nDataSubsets main unique total =", int(data_main["samples"].sum()))

    print("\n==== LLMSubsets main counts by input type ====")
    llm_main = df[
        (df["subset_type"] == "LLMSubsets")
        & (df["level"] == "main")
    ]

    print(
        llm_main[["dataset", "input_type", "samples", "subjects", "file"]]
        .sort_values(["dataset", "input_type"])
        .to_string(index=False)
    )

    print("\nLLMSubsets main rows across 4 input files =", int(llm_main["samples"].sum()))

    print("\n==== Expected main prediction rows ====")
    unique_main_samples = int(data_main["samples"].sum())
    input_count = 4
    lm_usage_count = 3
    output_count = 2
    combo_count = input_count * lm_usage_count * output_count

    print(f"unique main samples = {unique_main_samples}")
    print(f"combinations = {input_count} inputs × {lm_usage_count} LM usages × {output_count} outputs = {combo_count}")
    print(f"expected prediction rows = {unique_main_samples} × {combo_count} = {unique_main_samples * combo_count}")


all_results = []

for folder in targets:
    all_results.extend(scan_folder(folder))

print_summary_table(all_results)