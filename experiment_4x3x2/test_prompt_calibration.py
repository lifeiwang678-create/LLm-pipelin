from __future__ import annotations

from core.schema import LLMSample, decision_guidance_block
from LM.direct import DirectUsage
from LM.few_shot import FewShotUsage
from LM.multi_agent import MultiAgentUsage


def _sample(subject: str = "S1", label: int = 0) -> LLMSample:
    return LLMSample(
        dataset="WESAD",
        subject=subject,
        label=label,
        input_text="feature summary with mixed evidence",
        meta={"sample_id": f"{subject}-{label}"},
    )


def test_decision_guidance_is_symmetric_and_protects_label_one_recall() -> None:
    guidance = decision_guidance_block("WESAD")

    assert "neither label 0 nor label 1 is a default" in guidance
    assert "Choose label 1 only" in guidance
    assert "Choose label 0 when" in guidance
    assert "one isolated extreme value" in guidance


def test_direct_prompt_uses_symmetric_decision_calibration() -> None:
    usage = DirectUsage(
        labels=[0, 1],
        input_name="feature_description",
        output_instructions='Return {"predicted_state": 0 or 1}.',
        dataset="WESAD",
    )

    prompt = usage.build_prompt(_sample())

    assert "Decision calibration:" in prompt
    assert "neither label 0 nor label 1 is a default" in prompt
    assert "Choose label 1 only" in prompt
    assert "Do not predict label 1 or the positive class" not in prompt


def test_few_shot_prompt_uses_symmetric_decision_calibration() -> None:
    examples = [_sample("S1", 0), _sample("S2", 1)]
    usage = FewShotUsage(
        labels=[0, 1],
        input_name="feature_description",
        output_instructions='Return {"predicted_state": 0 or 1}.',
        examples=examples,
        n_per_class=1,
        example_selection="class_balanced",
        dataset="WESAD",
    )

    prompt = usage.build_prompt(_sample("S3", 0))

    assert "Decision calibration:" in prompt
    assert "neither label 0 nor label 1 is a default" in prompt
    assert "Choose label 1 only" in prompt
    assert "Do not predict label 1 or the positive class" not in prompt


def test_multi_agent_prompts_use_symmetric_decision_calibration() -> None:
    usage = MultiAgentUsage(
        labels=[0, 1],
        input_name="feature_description",
        output_instructions='Return {"predicted_state": 0 or 1}.',
        dataset="WESAD",
    )
    sample = _sample()

    prompts = [
        usage.build_prompt(sample),
        usage.build_agent_prompt(sample, "signal_pattern_agent"),
        usage.build_judge_prompt(
            sample,
            agent_outputs=[
                {
                    "agent": "signal_pattern_agent",
                    "vote": 1,
                    "response": '{"predicted_label": 1, "supporting_evidence": ["multi-cue pattern"]}',
                }
            ],
            majority_vote=1,
        ),
    ]

    for prompt in prompts:
        assert "Decision calibration:" in prompt
        assert "neither label 0 nor label 1 is a default" in prompt
        assert "Choose label 1 only" in prompt
        assert "Do not predict label 1 or the positive class" not in prompt
