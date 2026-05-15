from .direct import DirectUsage
from .few_shot import FewShotUsage
from .multi_agent import MultiAgentUsage


def build_lm_usage(
    config: dict,
    labels: list[int],
    input_name: str,
    train_samples,
    output_instructions: str,
):
    kind = str(config.get("type", "direct")).lower()

    if kind == "direct":
        return DirectUsage(labels=labels, input_name=input_name, output_instructions=output_instructions)

    if kind in {"fewshot", "few_shot", "few-shot"}:
        return FewShotUsage(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            examples=train_samples,
            n_per_class=int(config.get("n_per_class", 2)),
            random_state=int(config.get("random_state", 42)),
            example_max_chars=config.get("example_max_chars"),
        )

    if kind in {"multiagent", "multi_agent", "multi-agent"}:
        return MultiAgentUsage(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            agents=config.get("agents"),
            final_decider=config.get("final_decider", "final_decision_agent"),
        )

    raise ValueError(f"Unknown LM usage type: {kind}")


__all__ = [
    "DirectUsage",
    "FewShotUsage",
    "MultiAgentUsage",
    "build_lm_usage",
]
