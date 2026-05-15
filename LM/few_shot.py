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
        example_max_chars: int | None = None,
        dataset: str | None = None,
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.example_max_chars = example_max_chars
        self.dataset = dataset
        # The caller should pass examples from the training split only.
        self.examples = self._sample_examples(examples, n_per_class, random_state)

    def _sample_examples(self, examples: list[Sample], n_per_class: int, random_state: int) -> list[Sample]:
        if n_per_class < 1:
            raise ValueError("few_shot n_per_class must be at least 1.")

        rng = random.Random(random_state)
        picked = []
        for label in self.labels:
            class_examples = [sample for sample in examples if sample.label == label]
            if len(class_examples) < n_per_class:
                raise ValueError(
                    "Insufficient few-shot examples for label "
                    f"{label}: required {n_per_class}, got {len(class_examples)}. "
                    "Use more training subjects, reduce n_per_class, or adjust the label set."
                )
            rng.shuffle(class_examples)
            picked.extend(class_examples[:n_per_class])
        rng.shuffle(picked)
        return picked

    def build_prompt(self, sample: Sample) -> str:
        # This is prompt-level few-shot in-context learning, not fine-tuning or prompt tuning.
        example_blocks = []
        for idx, example in enumerate(self.examples, 1):
            example_blocks.append(
                f"""Example {idx}
{self._format_example_input(example.input_text)}

Correct label:
- {example.label}"""
            )

        return f"""You are given time-series samples for classification.

Task:
Classify the state of the final sample using the selected input representation: {self.input_name}.
The examples and the final sample use the same input representation.

Labels:
{label_block(self.labels, self.dataset)}

Important constraints:
- Use only the information provided in this prompt.
- Use the examples only as label-format and decision-reference demonstrations.
- Do not use knowledge outside the provided prompt.
- Do not add extra explanation outside JSON.
- Predict only the label of the final sample.

Few-shot examples:
{chr(10).join(example_blocks)}

Now classify the following final sample.

{sample.input_text}

{self.output_instructions}"""

    def _format_example_input(self, input_text: str) -> str:
        if self.example_max_chars is None or len(input_text) <= self.example_max_chars:
            return input_text
        return (
            input_text[: self.example_max_chars].rstrip()
            + "\n\n[Example input truncated for few-shot context length.]"
        )


__all__ = ["FewShotUsage"]
