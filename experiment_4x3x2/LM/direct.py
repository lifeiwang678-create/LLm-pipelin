from __future__ import annotations

from core.schema import Sample, label_block, label_rules_block


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

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Important constraints:
- Use only the information provided in this prompt.
- Do not use knowledge outside the provided prompt.
- Apply the dataset-specific label rules exactly.
- Treat all allowed labels symmetrically; neither label 0 nor label 1 is a default or safer answer.
- Compare the strongest evidence for label 0 against the strongest evidence for label 1 before deciding.
- Choose label 0 when label 0 has stronger overall support.
- Choose label 1 when label 1 has stronger overall support.
- If evidence is mixed, choose the label with the stronger overall support; do not choose a label because it appears earlier or feels safer.
- Do not add extra explanation outside JSON.
- Process this sample independently.

{sample.input_text}

{self.output_instructions}"""


__all__ = ["DirectUsage"]
