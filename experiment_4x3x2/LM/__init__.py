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
        )

    if kind == "multi_agent":
        return MultiAgentUsage(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            agents=config.get("agents"),
            final_decider=config.get("final_decider", "final_decision_agent"),
            dataset=dataset,
        )

    raise ValueError(f"Unknown LM usage type: {kind}")


__all__ = [
    "LM_REGISTRY",
    "DirectUsage",
    "FewShotUsage",
    "MultiAgentUsage",
    "build_lm_usage",
]
