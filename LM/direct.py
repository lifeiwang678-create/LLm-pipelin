from __future__ import annotations

from core.schema import Sample, label_block


class DirectUsage:
    name = "direct"

    def __init__(self, labels: list[int], input_name: str, output_instructions: str) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions

    def build_prompt(self, sample: Sample) -> str:
        return f"""You are given one physiological sample for classification.

Task:
Classify the state of this sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels)}

Important constraints:
- Use only the provided sample content.
- Do not use external medical knowledge.
- Do not guess randomly.
- Do not add extra explanation outside JSON.
- Process this sample independently.

{sample.input_text}

{self.output_instructions}"""


__all__ = ["DirectUsage"]
