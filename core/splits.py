from __future__ import annotations


def validate_fewshot_split(train_subjects: list[str] | None, test_subjects: list[str] | None) -> None:
    if not train_subjects or not test_subjects:
        raise ValueError("few_shot requires explicit train_subjects and test_subjects.")

    overlap = sorted(set(train_subjects) & set(test_subjects))
    if overlap:
        raise ValueError(f"few_shot leakage: test subjects also appear in examples: {overlap}")

