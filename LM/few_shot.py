from __future__ import annotations

import random

from core.schema import Sample, label_block


class FewShotUsage:
    name = "few_shot"

    def __init__(
        self,
        labels: list[int],
        input_name: str,
        output_instructions: str,
        examples: list[Sample],
        n_per_class: int = 2,
        random_state: int = 42,
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.examples = self._sample_examples(examples, n_per_class, random_state)

    def _sample_examples(self, examples: list[Sample], n_per_class: int, random_state: int) -> list[Sample]:
        rng = random.Random(random_state)
        picked = []
        for label in self.labels:
            class_examples = [sample for sample in examples if sample.label == label]
            rng.shuffle(class_examples)
            picked.extend(class_examples[: min(n_per_class, len(class_examples))])
        return picked

    def build_prompt(self, sample: Sample) -> str:
        example_blocks = []
        for idx, example in enumerate(self.examples, 1):
            example_blocks.append(
                f"""Example {idx}
{example.input_text}

Correct label:
- {example.label}"""
            )

        return f"""You are given physiological samples for classification.

Task:
Classify the state of the final sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels)}

Important constraints:
- Use only the provided sample content and few-shot examples.
- Do not use external medical knowledge.
- Do not add extra explanation outside JSON.
- Process the final sample independently based on the examples.

Few-shot examples:
{chr(10).join(example_blocks)}

Now classify the following sample.

{sample.input_text}

{self.output_instructions}"""


__all__ = ["FewShotUsage"]
