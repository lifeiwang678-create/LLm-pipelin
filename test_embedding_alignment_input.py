from __future__ import annotations

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
        test_unsupported_dataset_raises,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
