from __future__ import annotations

from core.schema import LLMSample
from LM.few_shot import FewShotUsage


def _sample(subject: str, label: int) -> LLMSample:
    return LLMSample(
        dataset="WESAD",
        subject=subject,
        label=label,
        input_text=f"subject={subject} label-hidden sample={label}",
        meta={"sample_id": f"{subject}-{label}"},
    )


def test_leave_one_subject_out_fewshot_excludes_eval_subject_and_is_reproducible() -> None:
    examples = [_sample(subject, label) for subject in ["S1", "S2", "S3", "S4", "S5", "S6"] for label in [0, 1]]
    eval_sample = _sample("S3", 0)

    usage_a = FewShotUsage(
        labels=[0, 1],
        input_name="feature_description",
        output_instructions='Return JSON with "predicted_state".',
        examples=examples,
        example_selection="leave_one_subject_out",
        example_subjects=3,
        examples_per_subject_per_label=1,
        random_state=42,
        dataset="WESAD",
    )
    prompt_a = usage_a.build_prompt(eval_sample)

    usage_b = FewShotUsage(
        labels=[0, 1],
        input_name="feature_description",
        output_instructions='Return JSON with "predicted_state".',
        examples=examples,
        example_selection="leave_one_subject_out",
        example_subjects=3,
        examples_per_subject_per_label=1,
        random_state=42,
        dataset="WESAD",
    )
    prompt_b = usage_b.build_prompt(eval_sample)

    assert usage_a.last_example_count == 6
    assert len(usage_a.last_example_subjects) == 3
    assert "S3" not in usage_a.last_example_subjects
    assert usage_a.last_example_subjects == usage_b.last_example_subjects
    assert prompt_a == prompt_b


def test_leave_one_subject_out_fewshot_requires_enough_other_subjects() -> None:
    examples = [_sample(subject, label) for subject in ["S1", "S2", "S3"] for label in [0, 1]]
    usage = FewShotUsage(
        labels=[0, 1],
        input_name="feature_description",
        output_instructions='Return JSON with "predicted_state".',
        examples=examples,
        example_selection="leave_one_subject_out",
        example_subjects=3,
        examples_per_subject_per_label=1,
        random_state=42,
        dataset="WESAD",
    )

    try:
        usage.build_prompt(_sample("S1", 0))
    except ValueError as exc:
        assert "Insufficient few-shot example subjects" in str(exc)
    else:
        raise AssertionError("Expected insufficient-subject error.")
