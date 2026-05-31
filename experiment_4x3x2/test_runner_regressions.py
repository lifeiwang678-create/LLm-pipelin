from __future__ import annotations

from pathlib import Path

import pytest

from core.lm_client import build_lm_client
from core.runner import (
    _cache_loader_metadata,
    _can_share_lm_usage,
    _default_output_max_tokens,
    _normalize_run_config,
    _run_eval_samples,
    _validate_balanced_per_label_counts,
    _validate_subject_config_semantics,
)
from core.schema import LLMSample
from LM import build_lm_usage
from LM.few_shot import FewShotUsage


def _sample(idx: int, subject: str = "S1", label: int = 0) -> LLMSample:
    return LLMSample(
        dataset="WESAD",
        subject=subject,
        label=label,
        input_text=f"sample {idx}",
        meta={"sample_id": idx},
    )


def test_few_shot_usage_is_not_shared_between_workers() -> None:
    usage = FewShotUsage(
        labels=[0, 1],
        input_name="feature_description",
        output_instructions='Return JSON with "predicted_state".',
        examples=[_sample(1, "S1", 0), _sample(2, "S1", 1)],
        example_selection="class_balanced",
        n_per_class=1,
    )

    assert _can_share_lm_usage("few_shot", usage) is False


def test_default_lm_client_stays_openai_compatible() -> None:
    config = _normalize_run_config(
        {
            "dataset": {"name": "WESAD"},
            "input": {"type": "feature_description"},
            "lm_usage": {"type": "direct"},
            "output": {"type": "label_only"},
        },
        dataset_name="WESAD",
    )

    assert config["lm_client"]["provider"] == "openai_compatible"
    assert config["lm_client"]["api_key"] == "lm-studio"
    assert config["lm_client"]["model"] == "qwen2.5-14b-instruct"


def test_gemini_lm_client_uses_environment_key_by_default() -> None:
    config = _normalize_run_config(
        {
            "dataset": {"name": "WESAD"},
            "input": {"type": "feature_description"},
            "lm_usage": {"type": "direct"},
            "output": {"type": "label_only"},
            "lm_client": {"provider": "gemini"},
        },
        dataset_name="WESAD",
    )

    assert config["lm_client"]["provider"] == "gemini"
    assert config["lm_client"]["model"] == "gemini-3.5-flash"
    assert config["lm_client"]["max_tokens"] == 384
    assert "api_key" not in config["lm_client"]


def test_gemini_cli_defaults_use_larger_output_limits() -> None:
    assert _default_output_max_tokens("label_only", "gemini") == 384
    assert _default_output_max_tokens("label_explanation", "gemini") == 768


def test_gemini_label_explanation_config_uses_larger_output_limit() -> None:
    config = _normalize_run_config(
        {
            "dataset": {"name": "WESAD"},
            "input": {"type": "feature_description"},
            "lm_usage": {"type": "direct"},
            "output": {"type": "label_explanation"},
            "lm_client": {"provider": "gemini"},
        },
        dataset_name="WESAD",
    )

    assert config["lm_client"]["max_tokens"] == 768


def test_openai_compatible_default_output_limits_stay_unchanged() -> None:
    assert _default_output_max_tokens("label_only", "openai_compatible") == 128
    assert _default_output_max_tokens("label_explanation", "openai_compatible") == 256


def test_gemini_lm_client_reports_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Set GEMINI_API_KEY"):
        build_lm_client({"provider": "gemini"})


def test_lm_client_accepts_legacy_max_tokens_config() -> None:
    client = build_lm_client(
        {
            "provider": "openai_compatible",
            "model": "qwen2.5-14b-instruct",
            "max_tokens": 321,
        }
    )

    assert client.max_completion_tokens == 321


def test_parallel_evaluation_reports_multiple_failures(tmp_path: Path) -> None:
    class FailingUsage:
        def build_prompt(self, sample):
            raise ValueError(f"boom {sample.meta['sample_id']}")

    class UnusedOutputHandler:
        pass

    try:
        _run_eval_samples(
            eval_samples=[_sample(1), _sample(2)],
            config={
                "lm_client": {},
                "lm_usage": {"type": "direct"},
                "output": {"type": "label_only"},
            },
            labels=[0, 1],
            input_name="feature_description",
            train_samples=[],
            output_instructions="",
            dataset="WESAD",
            output_handler=UnusedOutputHandler(),
            usage_type="direct",
            shared_lm_usage=FailingUsage(),
            sample_meta_safe_keys=set(),
            trace_path=tmp_path / "trace.jsonl",
            concurrency=2,
            log_every=1,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "Evaluation failed for 2 sample(s)" in message
        assert "sample 1/2: ValueError: boom 1" in message
        assert "sample 2/2: ValueError: boom 2" in message
    else:
        raise AssertionError("Expected aggregated parallel evaluation failure.")


def test_balanced_per_label_error_distinguishes_shortage_and_excess() -> None:
    try:
        _validate_balanced_per_label_counts({0: 1, 1: 3}, labels=[0, 1], expected=2)
    except RuntimeError as exc:
        message = str(exc)
        assert "insufficient samples {0: 1}" in message
        assert "too many samples {1: 3}" in message
    else:
        raise AssertionError("Expected balanced_per_label validation error.")


def test_cache_loader_metadata_prefers_clear_source_layers() -> None:
    metadata = {
        "loader_kwargs": {"window_size": 1},
        "source_data_subset_metadata": {
            "source_processed_metadata": {
                "loader_kwargs": {"window_size": 2},
            }
        },
    }

    assert _cache_loader_metadata(metadata)["loader_kwargs"] == {"window_size": 2}


def test_few_shot_subjects_requires_explicit_train_or_test_semantics() -> None:
    config = {
        "data": {
            "subjects": ["S1", "S2"],
            "subjects_explicit": True,
        }
    }

    try:
        _validate_subject_config_semantics(config, "few_shot")
    except ValueError as exc:
        assert "data.subjects/--subjects is ambiguous" in str(exc)
        assert "train_subjects" in str(exc)
        assert "test_subjects" in str(exc)
    else:
        raise AssertionError("Expected few_shot subject semantics error.")


def test_leave_one_subject_out_rejects_conflicting_n_per_class() -> None:
    try:
        build_lm_usage(
            {
                "type": "few_shot",
                "example_selection": "leave_one_subject_out",
                "n_per_class": 2,
                "examples_per_subject_per_label": 1,
            },
            labels=[0, 1],
            input_name="feature_description",
            train_samples=[_sample(1, "S1", 0), _sample(2, "S1", 1)],
            output_instructions='Return JSON with "predicted_state".',
            dataset="WESAD",
        )
    except ValueError as exc:
        assert "n_per_class is only used by class_balanced" in str(exc)
    else:
        raise AssertionError("Expected conflicting few-shot config error.")

