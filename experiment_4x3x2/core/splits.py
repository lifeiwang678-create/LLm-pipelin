from __future__ import annotations


def normalize_subjects(subjects: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if subjects is None:
        return None
    normalized = [str(subject) for subject in subjects if str(subject).strip()]
    return normalized or None


def validate_subject_independent_split(
    train_subjects: list[str] | None,
    test_subjects: list[str] | None,
) -> None:
    if not train_subjects or not test_subjects:
        raise ValueError("subject_independent split requires train_subjects and test_subjects.")

    overlap = sorted(set(train_subjects) & set(test_subjects))
    if overlap:
        raise ValueError(f"subject leakage: train and test subjects overlap: {overlap}")


def validate_fewshot_split(train_subjects: list[str] | None, test_subjects: list[str] | None) -> None:
    if not train_subjects or not test_subjects:
        raise ValueError("few_shot requires explicit train_subjects and test_subjects.")

    validate_subject_independent_split(train_subjects, test_subjects)
