from __future__ import annotations

import subprocess
import sys

import numpy as np

from core.runner import _resolve_dataset_config
from core.schema import SensorSample
from Input import build_input_provider
from Input.dreamt_feature_description import DreaMTFeatureDescriptionInput
from Input.feature_description import build_feature_description_input
from Input.hhar_feature_description import HHARFeatureDescriptionInput
from Input.wesad_feature_description import WESADFeatureDescriptionInput


def assert_nonempty_no_label_text(text: str) -> None:
    assert isinstance(text, str)
    assert text.strip()
    lowered = text.lower()
    for forbidden in ["label", "baseline", "stress", "amusement"]:
        assert forbidden not in lowered, f"output leaked label token: {forbidden}"


def test_factory_returns_dataset_specific_classes() -> None:
    assert isinstance(build_feature_description_input("WESAD"), WESADFeatureDescriptionInput)
    assert isinstance(build_feature_description_input("HHAR"), HHARFeatureDescriptionInput)
    assert isinstance(build_feature_description_input("DreaMT"), DreaMTFeatureDescriptionInput)
    assert isinstance(build_input_provider({"type": "feature_description", "dataset": "WESAD"}), WESADFeatureDescriptionInput)

    try:
        build_feature_description_input("UNKNOWN")
    except ValueError as exc:
        assert "Unknown feature description dataset" in str(exc)
    else:
        raise AssertionError("Unknown dataset should raise ValueError")


def test_runner_dataset_injection_rules() -> None:
    config = {"dataset": {"name": "WESAD"}, "input": {"type": "feature_description", "dataset": "HHAR"}}
    dataset_config = _resolve_dataset_config(config, dataset_name=None)
    input_config = dict(config["input"])
    input_config.setdefault("dataset", dataset_config.get("name"))
    assert input_config["dataset"] == "HHAR"

    config = {"dataset": {"name": "WESAD"}, "input": {"type": "feature_description"}}
    dataset_config = _resolve_dataset_config(config, dataset_name=None)
    input_config = dict(config["input"])
    input_config.setdefault("dataset", dataset_config.get("name"))
    assert input_config["dataset"] == "WESAD"

    config = {"input": {"type": "feature_description", "dataset": "HHAR"}}
    assert _resolve_dataset_config(config, dataset_name=None)["name"] == "HHAR"

    try:
        _resolve_dataset_config({"input": {"type": "feature_description"}}, dataset_name=None)
    except ValueError as exc:
        assert "Dataset name is required" in str(exc)
    else:
        raise AssertionError("Missing dataset should raise a clear ValueError")


def test_wesad_feature_description_mock_inputs() -> None:
    provider = WESADFeatureDescriptionInput()
    sample = _mock_wesad_sample()
    text = provider.build_input(sample)
    assert_nonempty_no_label_text(text)
    relabeled = SensorSample(
        dataset=sample.dataset,
        subject=sample.subject,
        label=1,
        signals=sample.signals,
    )
    assert provider.build_input(relabeled) == text
    for expected in ["Chest ACC", "Chest ECG/HRV", "Chest EDA/SCR", "Chest EMG", "Chest RESP", "Chest TEMP", "Wrist ACC", "Wrist BVP/HRV", "Wrist EDA/SCR", "Wrist TEMP"]:
        assert expected in text

    missing = SensorSample(
        dataset="WESAD",
        subject="mock_missing",
        label=2,
        signals={"chest_acc": np.ones((20, 3), dtype=float)},
    )
    missing_text = provider.build_input(missing)
    assert_nonempty_no_label_text(missing_text)
    assert "Chest ACC" in missing_text
    assert "Chest EDA/SCR" not in missing_text
    assert "Wrist BVP/HRV" not in missing_text

    short = SensorSample(
        dataset="WESAD",
        subject="mock_short",
        label=2,
        signals={
            "chest_acc": np.ones((2, 3), dtype=float),
            "chest_eda": np.array([0.1, 0.2]),
            "chest_temp": np.array([32.0]),
            "wrist_bvp": np.array([0.1, 0.2]),
        },
    )
    assert_nonempty_no_label_text(provider.build_input(short))


def test_hhar_feature_description_motion_only() -> None:
    provider = HHARFeatureDescriptionInput()
    sample = SensorSample(
        dataset="HHAR",
        subject="mock_hhar",
        label=1,
        signals={
            "acc_x": np.linspace(0, 1, 32),
            "acc_y": np.linspace(1, 0, 32),
            "acc_z": np.sin(np.linspace(0, 2, 32)),
            "gyro_x": np.cos(np.linspace(0, 2, 32)),
            "eda": np.linspace(0.1, 0.2, 32),
            "resp": np.linspace(0.1, 0.2, 32),
            "ecg": np.linspace(0.1, 0.2, 32),
        },
    )
    text = provider.build_input(sample)
    assert_nonempty_no_label_text(text)
    assert "Accelerometer" in text
    assert "Gyroscope" in text
    for forbidden in ["EDA", "SCR", "RESP", "ECG", "chest_", "wrist_"]:
        assert forbidden not in text


def test_dreamt_feature_description_dataset_fields() -> None:
    provider = DreaMTFeatureDescriptionInput()
    sample = SensorSample(
        dataset="DREAMT",
        subject="mock_dreamt",
        label=3,
        signals={
            "eda": np.linspace(0.1, 0.3, 32),
            "bvp": np.sin(np.linspace(0, 4, 32)),
            "heart_rate": np.linspace(65, 72, 32),
            "skin_temp": np.linspace(32.0, 33.0, 32),
            "actigraphy_x": np.cos(np.linspace(0, 2, 32)),
            "ecg": np.linspace(0.1, 0.2, 32),
            "resp": np.linspace(0.1, 0.2, 32),
            "scr_num": np.linspace(0.1, 0.2, 32),
        },
    )
    text = provider.build_input(sample)
    assert_nonempty_no_label_text(text)
    for expected in ["Electrodermal activity", "Pulse and heart-rate signals", "Temperature", "Activity and movement"]:
        assert expected in text
    for forbidden in ["SCR", "ECG", "RESP", "chest_", "wrist_"]:
        assert forbidden not in text


def test_dataset_feature_description_modules_can_run() -> None:
    for module_name in [
        "Input.wesad_feature_description",
        "Input.hhar_feature_description",
        "Input.dreamt_feature_description",
    ]:
        completed = subprocess.run(
            [sys.executable, "-m", module_name],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip()


def _mock_wesad_sample() -> SensorSample:
    rng = np.random.default_rng(7)
    sec = 12
    chest_fs = 700
    bvp_fs = 64
    wrist_acc_fs = 32
    wrist_slow_fs = 4

    chest_t = np.arange(sec * chest_fs) / chest_fs
    bvp_t = np.arange(sec * bvp_fs) / bvp_fs
    wrist_acc_t = np.arange(sec * wrist_acc_fs) / wrist_acc_fs
    wrist_slow_t = np.arange(sec * wrist_slow_fs) / wrist_slow_fs

    return SensorSample(
        dataset="WESAD",
        subject="mock_wesad",
        label=2,
        signals={
            "chest_acc": np.column_stack([
                np.sin(chest_t),
                np.cos(chest_t),
                np.sin(chest_t * 0.5),
            ]),
            "chest_ecg": np.sin(2 * np.pi * 1.2 * chest_t) + 0.02 * rng.normal(size=len(chest_t)),
            "chest_eda": 0.2 + 0.01 * np.sin(2 * np.pi * 0.05 * chest_t),
            "chest_emg": 0.05 * rng.normal(size=len(chest_t)),
            "chest_resp": np.sin(2 * np.pi * 0.2 * chest_t),
            "chest_temp": 32.0 + 0.01 * chest_t,
            "wrist_acc": np.column_stack([
                np.sin(wrist_acc_t),
                np.cos(wrist_acc_t),
                np.sin(wrist_acc_t * 0.5),
            ]),
            "wrist_bvp": np.sin(2 * np.pi * 1.1 * bvp_t) + 0.02 * rng.normal(size=len(bvp_t)),
            "wrist_eda": 0.1 + 0.01 * np.sin(2 * np.pi * 0.05 * wrist_slow_t),
            "wrist_temp": 31.5 + 0.01 * wrist_slow_t,
        },
    )


def main() -> None:
    tests = [
        test_factory_returns_dataset_specific_classes,
        test_runner_dataset_injection_rules,
        test_wesad_feature_description_mock_inputs,
        test_hhar_feature_description_motion_only,
        test_dreamt_feature_description_dataset_fields,
        test_dataset_feature_description_modules_can_run,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
