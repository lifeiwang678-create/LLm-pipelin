from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict

from core.schema import (
    Sample,
    decision_guidance_block,
    label_block,
    label_names_for_dataset,
    label_rules_block,
)


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
        example_subjects: int = 3,
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

        picked = []
        for label in self.labels:
            class_examples = [sample for sample in examples if sample.label == label]
            if len(class_examples) < n_per_class:
                raise ValueError(
                    "Insufficient few-shot examples for label "
                    f"{label}: required {n_per_class}, got {len(class_examples)}. "
                    "Use more training subjects, reduce n_per_class, or adjust the label set."
                )
            class_examples = _stable_sample_order(
                class_examples,
                random_state,
                "class-balanced",
                int(label),
            )
            picked.extend(class_examples[:n_per_class])
        return _stable_sample_order(picked, random_state, "class-balanced-output")

    def build_prompt(self, sample: Sample) -> str:
        # This is prompt-level few-shot in-context learning, not fine-tuning or prompt tuning.
        prompt, subjects, count = self.build_prompt_with_metadata(sample)
        self.last_example_subjects = subjects
        self.last_example_count = count
        return prompt

    def build_prompt_with_metadata(self, sample: Sample) -> tuple[str, list[str], int]:
        examples, subjects = self._examples_for_sample(sample)
        example_blocks = []
        for idx, example in enumerate(examples, 1):
            example_blocks.append(
                f"""Example {idx}
{self._format_example_input(example.input_text)}

Correct answer JSON:
{self._format_example_answer(example.label)}"""
            )

        prompt = f"""You are given time-series samples for classification.

Task:
Classify the state of the final sample using the selected input representation: {self.input_name}.
The examples and the final sample use the same input representation.

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Decision calibration:
{decision_guidance_block(self.dataset)}

Important constraints:
- Use only the information provided in this prompt.
- Use the examples only as label-format and decision-reference demonstrations.
- Do not use knowledge outside the provided prompt.
- Apply the dataset-specific label rules exactly.
- Consider evidence for both labels before choosing the final label.
- Do not add extra explanation outside JSON.
- Predict only the label of the final sample.

Few-shot examples:
{chr(10).join(example_blocks)}

Now classify the following final sample.

{sample.input_text}

{self.output_instructions}"""
        return prompt, subjects, len(examples)

    def _examples_for_sample(self, sample: Sample) -> tuple[list[Sample], list[str]]:
        if self.example_selection != "leave_one_subject_out":
            subjects = sorted({str(example.subject) for example in self.examples})
            return self.examples, subjects

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

        ranked_subjects = sorted(
            candidate_subjects,
            key=lambda subject: _stable_digest(
                self.random_state,
                "few-shot-subject",
                eval_subject,
                subject,
            ),
        )
        selected_subjects = ranked_subjects[: self.example_subjects]

        picked = []
        for subject in selected_subjects:
            by_label = self._examples_by_subject_label[subject]
            for label in self.labels:
                examples = list(by_label[int(label)])
                examples = _stable_sample_order(
                    examples,
                    self.random_state,
                    "few-shot-sample",
                    eval_subject,
                    subject,
                    int(label),
                )
                picked.extend(examples[: self.examples_per_subject_per_label])
        return picked, list(selected_subjects)

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


def _stable_digest(random_state: int, *parts: object) -> str:
    payload = ":".join(
        [str(int(random_state)), *(str(part) for part in parts)]
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def _stable_sample_order(samples: list[Sample], random_state: int, *parts: object) -> list[Sample]:
    return sorted(
        samples,
        key=lambda sample: _stable_digest(
            random_state,
            *parts,
            _sample_fingerprint(sample),
        ),
    )


def _sample_fingerprint(sample: Sample) -> str:
    meta = dict(getattr(sample, "meta", {}) or {})
    parts = {
        "dataset": getattr(sample, "dataset", ""),
        "subject": str(getattr(sample, "subject", "")),
        "label": int(getattr(sample, "label")),
        "input_sha256": hashlib.sha256(
            str(getattr(sample, "input_text", "")).encode("utf-8")
        ).hexdigest(),
    }
    for key in ("sample_id", "data_index", "epoch_id", "local_index"):
        if key in meta:
            parts[key] = meta[key]
    return json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)


def _subject_sort_key(subject: str) -> list[object]:
    parts = re.split(r"(\d+)", str(subject))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


__all__ = ["FewShotUsage"]
