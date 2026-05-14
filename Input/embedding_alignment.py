from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.schema import Sample


class EmbeddingAlignmentInput:
    name = "embedding_alignment"

    def __init__(
        self,
        data_path: str | Path,
        qa_path: str | Path,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.qa_path = Path(qa_path)
        self.label_map = label_map or {
            "Non-stress": 1,
            "Stress": 2,
            "Baseline": 1,
            "Amusement": 3,
        }

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[Sample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Cannot find embedding/alignment data file: {self.data_path}")
        if not self.qa_path.exists():
            raise FileNotFoundError(f"Cannot find embedding/alignment QA file: {self.qa_path}")

        subject_filter = set(subjects or [])
        with self.qa_path.open("r", encoding="utf-8") as f:
            qa_items = json.load(f)["dataset"]

        samples = []
        for item in qa_items:
            subject = str(item.get("subject", ""))
            if subject_filter and subject not in subject_filter:
                continue

            qa_pair = item.get("qa_pair", {})
            label_text = qa_pair.get("A", item.get("binary_label", item.get("original_3class_label", "")))
            label = int(self.label_map.get(label_text, item.get("majority_label_original", 0)))
            if label not in labels:
                continue

            samples.append(
                Sample(
                    subject=subject,
                    label=label,
                    input_text="SensorLLM embedding/alignment input",
                    meta={
                        "data_path": str(self.data_path),
                        "qa_path": str(self.qa_path),
                        "data_index": int(item["index"]),
                        "label_text": label_text,
                        "question": qa_pair.get("Q", ""),
                        "source": str(self.qa_path),
                        "local_index": item.get("local_index"),
                        "start_index": item.get("start_index"),
                        "end_index": item.get("end_index"),
                    },
                )
            )
        return samples


__all__ = ["EmbeddingAlignmentInput"]
