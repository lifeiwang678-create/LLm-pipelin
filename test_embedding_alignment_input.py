from __future__ import annotations

import json
from tempfile import TemporaryDirectory

import numpy as np

from core.schema import SensorSample
from Input import build_input_provider
from Input.embedding_alignment import EmbeddingAlignmentInput


def test_factory_aliases() -> None:
    embedding_provider = build_input_provider({"type": "embedding_alignment", "dataset": "WESAD"})
    encoded_provider = build_input_provider({"type": "encoded_time_series", "dataset": "WESAD"})

    assert isinstance(embedding_provider, EmbeddingAlignmentInput)
    assert isinstance(encoded_provider, EmbeddingAlignmentInput)
    assert embedding_provider.name == "embedding_alignment"
    assert encoded_provider.name == "encoded_time_series"


def test_encoded_time_series_prompt() -> None:
    sample = _mock_wesad_sample()
    provider = EmbeddingAlignmentInput(dataset="WESAD", max_points=64, num_segments=4)
    prompt = provider.build_input(sample)
    assert "Input representation: Encoded Time-series" in prompt
    assert "<WRIST_EDA_CHANNEL>" in prompt
    assert "<WRIST_BVP_CHANNEL>" in prompt
    assert "<WRIST_ACC_X_CHANNEL>" in prompt
    assert "Encoded temporal pattern:" in prompt
    assert "Dynamic cue:" in prompt
    assert "Supporting statistics:" in prompt
    lowered = prompt.lower()
    for forbidden in ["baseline", "stress", "amusement"]:
        assert forbidden not in lowered


def test_missing_and_invalid_channels_do_not_fail() -> None:
    sample = SensorSample(
        dataset="WESAD",
        subject="mock",
        label=1,
        signals={
            "wrist_eda": [0.1, "bad", 0.2, None, 0.3],
            "missing_like_unknown": ["bad", None],
        },
    )
    prompt = EmbeddingAlignmentInput(dataset="WESAD").build_input(sample)
    assert "<WRIST_EDA_CHANNEL>" in prompt
    assert "missing_like_unknown" not in prompt


def test_custom_metadata_is_used_for_matching_dataset() -> None:
    sample = SensorSample(
        dataset="WESAD",
        subject="mock",
        label=1,
        signals={"wrist_eda": [0.1, 0.2, 0.3, 0.4]},
    )
    provider = EmbeddingAlignmentInput(
        dataset="WESAD",
        channel_metadata={
            "wrist_eda": {
                "channel": "EDA_CUSTOM",
                "sensor_type": "custom electrodermal channel",
                "body_location": "custom wrist",
                "sampling_rate": 9,
            }
        },
    )
    prompt = provider.build_input(sample)
    assert "<CUSTOM_WRIST_EDA_CUSTOM_CHANNEL>" in prompt
    assert "Sensor type: custom electrodermal channel" in prompt
    assert "Sampling rate: 9 Hz" in prompt


def test_nested_wesad_signals_and_acc_orientation() -> None:
    sample = SensorSample(
        dataset="WESAD",
        subject="mock",
        label=1,
        signals={
            "chest": {
                "ECG": np.linspace(0.0, 1.0, 12),
                "EDA": np.linspace(0.2, 0.4, 12),
                "ACC": np.vstack([
                    np.linspace(0.0, 1.0, 12),
                    np.linspace(1.0, 0.0, 12),
                    np.ones(12),
                ]),
            },
            "wrist": {
                "BVP": np.sin(np.linspace(0, 2 * np.pi, 12)),
                "ACC": np.column_stack([
                    np.linspace(0.0, 1.0, 12),
                    np.linspace(1.0, 0.0, 12),
                    np.ones(12),
                ]),
            },
        },
    )
    prompt = EmbeddingAlignmentInput(dataset="WESAD").build_input(sample)
    assert "<CHEST_ECG_CHANNEL>" in prompt
    assert "<CHEST_EDA_CHANNEL>" in prompt
    assert "<CHEST_ACC_X_CHANNEL>" in prompt
    assert "<WRIST_BVP_CHANNEL>" in prompt
    assert "<WRIST_ACC_Z_CHANNEL>" in prompt


def test_strict_raises_when_no_numeric_channels() -> None:
    sample = SensorSample(dataset="WESAD", subject="mock", label=1, signals={"unknown": ["bad"]})
    try:
        EmbeddingAlignmentInput(dataset="WESAD", strict=True).build_input(sample)
    except ValueError as exc:
        assert "No valid numeric sensor channels" in str(exc)
    else:
        raise AssertionError("strict=True should raise when no valid numeric channels are found")


def test_legacy_load_uses_explicit_label_not_qa_answer() -> None:
    with TemporaryDirectory() as tmp:
        qa_path = f"{tmp}/qa.json"
        payload = {
            "dataset": [
                {
                    "subject": "S2",
                    "index": 0,
                    "label": 1,
                    "qa_pair": {"Q": "What is the class?", "A": "Stress"},
                    "signals": {"wrist_eda": [0.1, 0.2, 0.3]},
                }
            ]
        }
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        samples = EmbeddingAlignmentInput(dataset="WESAD", qa_path=qa_path).load(["S2"], [1, 2])
    assert len(samples) == 1
    assert samples[0].label == 1


def test_legacy_load_requires_explicit_label() -> None:
    with TemporaryDirectory() as tmp:
        qa_path = f"{tmp}/qa.json"
        payload = {
            "dataset": [
                {
                    "subject": "S2",
                    "index": 0,
                    "qa_pair": {"Q": "What is the class?", "A": "Stress"},
                    "signals": {"wrist_eda": [0.1, 0.2, 0.3]},
                }
            ]
        }
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        try:
            EmbeddingAlignmentInput(dataset="WESAD", qa_path=qa_path).load(["S2"], [1, 2])
        except ValueError as exc:
            assert "explicit numeric label" in str(exc)
        else:
            raise AssertionError("legacy load should reject QA items without explicit labels")


def test_unsupported_dataset_raises() -> None:
    try:
        EmbeddingAlignmentInput(dataset="HHAR")
    except ValueError as exc:
        assert "Unsupported dataset" in str(exc)
    else:
        raise AssertionError("Unsupported dataset should raise ValueError")


def _mock_wesad_sample() -> SensorSample:
    t = np.linspace(0, 1, 80)
    return SensorSample(
        dataset="WESAD",
        subject="mock",
        label=2,
        signals={
            "wrist_eda": np.linspace(0.1, 0.6, 80),
            "wrist_bvp": np.sin(2 * np.pi * 5 * t),
            "wrist_acc": np.column_stack([np.sin(t), np.cos(t), np.linspace(0, 1, 80)]),
            "chest_resp": np.sin(2 * np.pi * 2 * t),
            "chest_temp": np.linspace(32.0, 32.3, 80),
        },
    )


def main() -> None:
    tests = [
        test_factory_aliases,
        test_encoded_time_series_prompt,
        test_missing_and_invalid_channels_do_not_fail,
        test_custom_metadata_is_used_for_matching_dataset,
        test_nested_wesad_signals_and_acc_orientation,
        test_strict_raises_when_no_numeric_channels,
        test_legacy_load_uses_explicit_label_not_qa_answer,
        test_legacy_load_requires_explicit_label,
        test_unsupported_dataset_raises,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
