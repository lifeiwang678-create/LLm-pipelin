from __future__ import annotations

from core.schema import Sample, label_block


class DirectUsage:
    name = "direct"

    def __init__(
        self,
        labels: list[int],
        input_name: str,
        output_instructions: str,
        dataset: str | None = None,
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.dataset = dataset

    def build_prompt(self, sample: Sample) -> str:
        return f"""You are given one time-series sample for classification.

Task:
Classify the state of this sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels, self.dataset)}

Important constraints:
- Use only the information provided in this prompt.
- Do not use knowledge outside the provided prompt.
- Do not guess randomly.
- Do not add extra explanation outside JSON.
- Process this sample independently.

{sample.input_text}

{self.output_instructions}"""


__all__ = ["DirectUsage"]
