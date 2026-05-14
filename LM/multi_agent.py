from __future__ import annotations

from core.schema import Sample, label_block


class MultiAgentUsage:
    name = "multi_agent"

    def __init__(
        self,
        labels: list[int],
        input_name: str,
        output_instructions: str,
        agents: list[str] | None = None,
        final_decider: str = "decision_maker",
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.agents = agents or ["signal_pattern_agent", "feature_trend_agent", "consistency_agent"]
        self.final_decider = final_decider

    def build_prompt(self, sample: Sample) -> str:
        agent_lines = "\n".join(f"- {agent}" for agent in self.agents)
        return f"""You are coordinating a multi-agent classification discussion for one physiological sample.

Task:
Classify the state of this sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels)}

Agents:
{agent_lines}
- {self.final_decider}: compare the agents' conclusions and choose the final label.

Important constraints:
- The agents must use only the provided sample content.
- Do not use external medical knowledge.
- Keep any reasoning internal unless the selected output format asks for explanation.
- The final answer must follow the requested JSON schema.

{sample.input_text}

{self.output_instructions}"""


__all__ = ["MultiAgentUsage"]
