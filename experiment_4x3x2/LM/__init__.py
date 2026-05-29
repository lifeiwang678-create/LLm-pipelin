from __future__ import annotations

from .direct import DirectUsage
from .few_shot import FewShotUsage
from .multi_agent import MultiAgentUsage


LM_REGISTRY = {
    "direct": DirectUsage,
    "few_shot": FewShotUsage,
    "multi_agent": MultiAgentUsage,
}


def build_lm_usage(
    config: dict,
    labels: list[int],
    input_name: str,
    train_samples: list,
    output_instructions: str,
    dataset: str | None = None,
):
    kind = str(config.get("type", "direct")).strip().lower()

    if kind == "direct":
        return DirectUsage(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            dataset=dataset,
        )

    if kind == "few_shot":
        return FewShotUsage(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            examples=train_samples,
            n_per_class=int(config.get("n_per_class", 2)),
            random_state=int(config.get("random_state", 42)),
            example_max_chars=config.get("example_max_chars"),
            dataset=dataset,
            example_selection=config.get("example_selection", "class_balanced"),
            example_subjects=int(config.get("example_subjects") or 3),
            examples_per_subject_per_label=int(config.get("examples_per_subject_per_label") or 1),
            exclude_eval_subject=bool(config.get("exclude_eval_subject", True)),
        )

    if kind == "multi_agent":
        # intermediate_max_tokens 由 runner.build_experiment_config 注入 (默认 1024)。
        # 这里仅在显式提供时透传,缺省走 MultiAgentUsage 的类内默认,保证旧配置仍能跑。
        intermediate_max_tokens = config.get("intermediate_max_tokens")
        kwargs = dict(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            agents=config.get("agents"),
            final_decider=config.get("final_decider", "final_decision_agent"),
            dataset=dataset,
        )
        if intermediate_max_tokens is not None:
            kwargs["intermediate_max_tokens"] = int(intermediate_max_tokens)
        return MultiAgentUsage(**kwargs)

    raise ValueError(f"Unknown LM usage type: {kind}")


__all__ = [
    "LM_REGISTRY",
    "DirectUsage",
    "FewShotUsage",
    "MultiAgentUsage",
    "build_lm_usage",
]
