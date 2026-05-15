from __future__ import annotations

from pathlib import Path

from core.schema import LLMSample, SensorSample

from .feature_description import get_feature_description_builder


class ExtraKnowledgeInput:
    name = "extra_knowledge"

    def __init__(
        self,
        dataset: str | None = None,
        knowledge_file: str | Path | None = None,
        knowledge_text: str = "",
    ) -> None:
        self.base_input = get_feature_description_builder(dataset)
        self.knowledge_file = Path(knowledge_file) if knowledge_file else None
        self.knowledge_text = knowledge_text

    def transform(self, sample: SensorSample) -> LLMSample:
        llm_sample = self.base_input.transform(sample)
        llm_sample.meta["base_input_type"] = llm_sample.meta.get("input_type", self.base_input.name)
        llm_sample.meta["input_type"] = self.name
        knowledge = self._load_knowledge()
        if not knowledge:
            return llm_sample

        llm_sample.input_text = f"""{llm_sample.input_text}

Extra knowledge:
{knowledge}"""
        llm_sample.meta["knowledge_source"] = str(self.knowledge_file) if self.knowledge_file else "inline"
        return llm_sample

    def transform_all(self, samples: list[SensorSample]) -> list[LLMSample]:
        return [self.transform(sample) for sample in samples]

    def _load_knowledge(self) -> str:
        if self.knowledge_text:
            return self.knowledge_text.strip()
        if self.knowledge_file and self.knowledge_file.exists():
            return self.knowledge_file.read_text(encoding="utf-8").strip()
        return ""


__all__ = ["ExtraKnowledgeInput"]
