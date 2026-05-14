from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.schema import Sample

from .feature_description import FeatureDescriptionInput


class ExtraKnowledgeInput:
    name = "extra_knowledge"

    def __init__(
        self,
        data_dir: str | Path,
        pattern: str = "*_features_paperstyle.csv",
        knowledge_file: str | Path | None = None,
        knowledge_text: str = "",
    ) -> None:
        self.base_input = FeatureDescriptionInput(data_dir=data_dir, pattern=pattern)
        self.knowledge_file = Path(knowledge_file) if knowledge_file else None
        self.knowledge_text = knowledge_text

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[Sample]:
        samples = self.base_input.load(subjects, labels)
        knowledge = self._load_knowledge()
        if not knowledge:
            return samples

        for sample in samples:
            sample.input_text = f"""{sample.input_text}

Extra knowledge:
{knowledge}"""
            sample.meta["knowledge_source"] = str(self.knowledge_file) if self.knowledge_file else "inline"
        return samples

    def _load_knowledge(self) -> str:
        if self.knowledge_text:
            return self.knowledge_text.strip()
        if self.knowledge_file and self.knowledge_file.exists():
            return self.knowledge_file.read_text(encoding="utf-8").strip()
        return ""


__all__ = ["ExtraKnowledgeInput"]
