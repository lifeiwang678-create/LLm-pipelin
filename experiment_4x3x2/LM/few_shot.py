from __future__ import annotations

import json
import hashlib
import random
import re
from collections import defaultdict

from core.schema import Sample, label_block, label_names_for_dataset, label_rules_block


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
        example_selection: str = "class_balanced",
        example_subjects: int = 5,
        examples_per_subject_per_label: int = 1,
        exclude_eval_subject: bool = True,
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.example_max_chars = example_max_chars
        self.dataset = dataset
        self.random_state = int(random_state)
        self.example_selection = _normalize_example_selection(example_selection)
        self.example_subjects = int(example_subjects)
        self.examples_per_subject_per_label = int(examples_per_subject_per_label)
        self.exclude_eval_subject = bool(exclude_eval_subject)
        self.last_example_subjects: list[str] = []
        self.last_example_count = 0

        if self.example_selection == "leave_one_subject_out":
            if self.example_subjects < 1:
                raise ValueError("few_shot example_subjects must be at least 1.")
            if self.examples_per_subject_per_label < 1:
                raise ValueError("few_shot examples_per_subject_per_label must be at least 1.")
            self.examples = list(examples)
            self._examples_by_subject_label = self._group_by_subject_label(self.examples)
        else:
            # Legacy prompt-level few-shot: the caller should pass training-split examples only.
            self.examples = self._sample_examples(examples, n_per_class, self.random_state)
            self._examples_by_subject_label = {}

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
        examples = self._examples_for_sample(sample)
        example_blocks = []
        for idx, example in enumerate(examples, 1):
            example_blocks.append(
                f"""Example {idx}
{self._format_example_input(example.input_text)}

Correct answer JSON:
{self._format_example_answer(example.label)}"""
            )

        return f"""You are given time-series samples for classification.

Task:
Classify the state of the final sample using the selected input representation: {self.input_name}.
The examples and the final sample use the same input representation.

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Important constraints:
- Use only the information provided in this prompt.
- Use the examples only as label-format and decision-reference demonstrations.
- Do not use knowledge outside the provided prompt.
- Apply the dataset-specific label rules exactly.
- Do not predict label 1 or the positive class from one high absolute sensor value alone.
- Consider evidence for both labels before choosing the final label.
- Do not add extra explanation outside JSON.
- Predict only the label of the final sample.

Few-shot examples:
{chr(10).join(example_blocks)}

Now classify the following final sample.

{sample.input_text}

{self.output_instructions}"""

    def _examples_for_sample(self, sample: Sample) -> list[Sample]:
        if self.example_selection != "leave_one_subject_out":
            self.last_example_subjects = sorted({str(example.subject) for example in self.examples})
            self.last_example_count = len(self.examples)
            return self.examples

        eval_subject = str(getattr(sample, "subject", ""))
        candidate_subjects = []
        for subject, by_label in self._examples_by_subject_label.items():
            if self.exclude_eval_subject and subject == eval_subject:
                continue
            if all(
                len(by_label.get(int(label), [])) >= self.examples_per_subject_per_label
                for label in self.labels
            ):
                candidate_subjects.append(subject)
        candidate_subjects = sorted(candidate_subjects, key=_subject_sort_key)
        if len(candidate_subjects) < self.example_subjects:
            raise ValueError(
                "Insufficient few-shot example subjects for leave-one-subject-out sampling. "
                f"Eval subject={eval_subject!r}, required={self.example_subjects}, "
                f"available={len(candidate_subjects)}. Build/use a train input cache with "
                "more subjects or reduce --few-shot-example-subjects."
            )

        rng = random.Random(_stable_subject_seed(self.random_state, eval_subject))
        shuffled_subjects = list(candidate_subjects)
        rng.shuffle(shuffled_subjects)
        selected_subjects = shuffled_subjects[: self.example_subjects]

        picked = []
        for subject in selected_subjects:
            by_label = self._examples_by_subject_label[subject]
            for label in self.labels:
                examples = list(by_label[int(label)])
                rng.shuffle(examples)
                picked.extend(examples[: self.examples_per_subject_per_label])
        rng.shuffle(picked)
        self.last_example_subjects = list(selected_subjects)
        self.last_example_count = len(picked)
        return picked

    def _group_by_subject_label(self, examples: list[Sample]) -> dict[str, dict[int, list[Sample]]]:
        groups: dict[str, dict[int, list[Sample]]] = defaultdict(lambda: defaultdict(list))
        for example in examples:
            groups[str(example.subject)][int(example.label)].append(example)
        return {
            subject: {label: list(items) for label, items in by_label.items()}
            for subject, by_label in groups.items()
        }

    def _format_example_input(self, input_text: str) -> str:
        if self.example_max_chars is None or len(input_text) <= self.example_max_chars:
            return input_text
        return (
            input_text[: self.example_max_chars].rstrip()
            + "\n\n[Example input truncated for few-shot context length.]"
        )

    def _format_example_answer(self, label: int) -> str:
        names = label_names_for_dataset(self.dataset)
        label_name = names.get(int(label), str(label))
        answer = {"predicted_state": int(label)}
        if '"explanation"' in self.output_instructions:
            answer["explanation"] = f"Example label: {label_name}."
        return json.dumps(answer, ensure_ascii=False)


def _normalize_example_selection(value: str | None) -> str:
    normalized = str(value or "class_balanced").strip().lower().replace("-", "_")
    if normalized in {"leave_one_subject_out", "loo", "subject_loo"}:
        return "leave_one_subject_out"
    if normalized in {"class_balanced", "label_balanced", "legacy"}:
        return "class_balanced"
    raise ValueError(
        f"Unknown few-shot example_selection={value!r}. "
        "Use 'leave_one_subject_out' or 'class_balanced'."
    )


def _stable_subject_seed(random_state: int, subject: str) -> int:
    payload = f"{int(random_state)}:{subject}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _subject_sort_key(subject: str) -> list[object]:
    parts = re.split(r"(\d+)", str(subject))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


__all__ = ["FewShotUsage"]
