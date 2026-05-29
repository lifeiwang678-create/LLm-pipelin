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
        example_selection = config.get("example_selection", "class_balanced")
        normalized_selection = str(example_selection or "class_balanced").strip().lower().replace("-", "_")
        n_per_class = config.get("n_per_class")
        examples_per_subject_per_label = config.get("examples_per_subject_per_label")
        if normalized_selection in {"leave_one_subject_out", "loo", "subject_loo"}:
            if n_per_class is not None and examples_per_subject_per_label is not None:
                if int(n_per_class) != int(examples_per_subject_per_label):
                    raise ValueError(
                        "few_shot n_per_class is only used by class_balanced sampling. "
                        "For leave_one_subject_out, set examples_per_subject_per_label "
                        "instead, or remove n_per_class."
                    )
            if examples_per_subject_per_label is None:
                examples_per_subject_per_label = n_per_class if n_per_class is not None else 1
            n_per_class = 2
        else:
            n_per_class = 2 if n_per_class is None else n_per_class
            examples_per_subject_per_label = (
                1 if examples_per_subject_per_label is None else examples_per_subject_per_label
            )
        return FewShotUsage(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            examples=train_samples,
            n_per_class=int(n_per_class),
            random_state=int(config.get("random_state", 42)),
            example_max_chars=config.get("example_max_chars"),
            dataset=dataset,
            example_selection=example_selection,
            example_subjects=int(config.get("example_subjects") or 3),
            examples_per_subject_per_label=int(examples_per_subject_per_label),
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
