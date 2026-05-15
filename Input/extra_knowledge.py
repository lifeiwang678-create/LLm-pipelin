from __future__ import annotations

from pathlib import Path
from typing import Any

from core.schema import LLMSample, SensorSample

from .feature_description import get_feature_description_builder
from .feature_description.basic_feature_description import BasicFeatureDescriptionInput


DEFAULT_DATASET_KNOWLEDGE = {
    "WESAD": {
        "context": "This dataset contains physiological time-series signals for stress or affective-state classification.",
        "channel_knowledge": [
            "EDA can reflect sympathetic nervous system arousal.",
            "ECG or BVP-related features can reflect heart-related physiological changes.",
            "Respiration features may change under stress or affective load.",
            "Temperature changes are usually slower and should be interpreted together with other channels.",
            "ACC mainly reflects body movement and can indicate motion artifacts.",
        ],
        "decision_guidance": [
            "Do not classify the state from one physiological feature alone.",
            "Consider whether movement-related features may affect physiological signals.",
            "Use cross-channel consistency as supporting evidence.",
            "Treat the dataset knowledge as supporting context, not as a deterministic rule.",
        ],
    },
    "HHAR": {
        "context": "This dataset contains wearable or smartphone motion time-series for human activity recognition.",
        "channel_knowledge": [
            "Accelerometer channels describe linear acceleration and motion intensity.",
            "Gyroscope channels describe rotational movement.",
            "Activity patterns should be inferred from temporal movement patterns, not from a single timestamp.",
            "Device placement, user differences, and orientation may affect signal magnitude.",
        ],
        "decision_guidance": [
            "Motion intensity alone may not uniquely identify the activity.",
            "Interpret accelerometer and gyroscope information jointly.",
            "Prefer temporal pattern evidence over isolated values.",
            "Consider user and device variation when interpreting signal magnitude.",
        ],
    },
    "DREAMT": {
        "context": "This dataset contains healthcare or sleep-related time-series signals.",
        "channel_knowledge": [
            "Sleep-related or healthcare-related states may depend on gradual temporal patterns.",
            "Physiological or wearable features should be interpreted over the full time window.",
            "Single noisy changes should not dominate the decision.",
            "Individual variation and sensor noise should be considered.",
        ],
        "decision_guidance": [
            "Use the whole temporal window as evidence.",
            "Avoid overinterpreting one abnormal value.",
            "Interpret available channels jointly.",
            "Treat the dataset knowledge as supporting context, not as a deterministic rule.",
        ],
    },
}


FALLBACK_DATASET_KNOWLEDGE = {
    "context": "This dataset contains time-series samples for classification.",
    "channel_knowledge": [
        "Time-series features should be interpreted jointly across available channels.",
        "Mean, variance, range, trend, and fluctuation patterns may provide useful evidence.",
        "Sensor noise and artifacts should be considered.",
    ],
    "decision_guidance": [
        "Use the current sample features as the primary evidence.",
        "Use the provided knowledge only as supporting context.",
        "Do not infer the label from one isolated feature.",
        "Predict only one label from the allowed label set provided by the main prompt.",
    ],
}


BASE_DECISION_GUIDANCE = [
    "Use the current sample features as the primary evidence.",
    "Use the dataset knowledge only as supporting context.",
    "Interpret multiple channels jointly.",
    "Do not infer the label from one isolated feature.",
    "Predict only one label from the allowed label set provided by the main prompt.",
]


class ExtraKnowledgeInput:
    """Feature-description input augmented with lightweight dataset knowledge.

    This is a ZARA-inspired prompt-compatible input only. It does not implement
    retrieval, embeddings, RRF, pair-wise activity knowledge bases, fine-tuning,
    or any LLM forward-process changes.
    """

    name = "extra_knowledge"

    def __init__(
        self,
        dataset: str | None = None,
        knowledge_file: str | Path | None = None,
        knowledge_text: str = "",
        knowledge_mode: str | None = None,
    ) -> None:
        self.dataset = dataset
        self._validate_dataset_name(dataset)
        self._base_input_cache = {}
        self.knowledge_file = Path(knowledge_file) if knowledge_file else None
        self.knowledge_text = knowledge_text.strip() if knowledge_text else ""
        has_external_knowledge = bool(self.knowledge_text) or self.knowledge_file is not None
        self.knowledge_mode = self._normalize_knowledge_mode(knowledge_mode, has_external_knowledge)

        if self.knowledge_file is not None and not self.knowledge_file.exists():
            raise FileNotFoundError(f"Cannot find extra knowledge file: {self.knowledge_file}")

        self.external_knowledge = self._load_external_knowledge()
        if self.knowledge_mode == "replace" and not self.external_knowledge:
            raise ValueError("knowledge_mode='replace' requires knowledge_text or knowledge_file.")

    def transform(self, sample: SensorSample) -> LLMSample:
        dataset = self._resolve_dataset(sample)
        base_input = self._base_input_for_dataset(dataset)
        base_sample = base_input.transform(sample)
        retrieved_examples = self._get_retrieved_examples(sample, base_sample)

        meta = dict(base_sample.meta)
        meta["base_input_type"] = meta.get("input_type", base_input.name)
        meta["input_type"] = self.name
        meta["knowledge_dataset"] = dataset
        meta["knowledge_mode"] = self.knowledge_mode
        meta["knowledge_source"] = self._knowledge_source()
        if retrieved_examples:
            meta["retrieved_examples_used"] = len(retrieved_examples)

        return LLMSample(
            dataset=base_sample.dataset,
            subject=base_sample.subject,
            label=base_sample.label,
            input_text=self._build_input_text(base_sample.input_text, dataset, retrieved_examples),
            meta=meta,
        )

    def transform_all(self, samples: list[SensorSample]) -> list[LLMSample]:
        return [self.transform(sample) for sample in samples]

    def _base_input_for_dataset(self, dataset: str | None):
        normalized = self._normalize_dataset_name(dataset)
        cache_key = normalized or "UNKNOWN"
        if cache_key in self._base_input_cache:
            return self._base_input_cache[cache_key]
        try:
            if normalized in {"", "UNKNOWN"}:
                base_input = BasicFeatureDescriptionInput()
            else:
                base_input = get_feature_description_builder(dataset)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported dataset for extra_knowledge input: {dataset}. "
                "Expected WESAD, HHAR, DREAMT, None, or UNKNOWN."
            ) from exc
        self._base_input_cache[cache_key] = base_input
        return base_input

    def _build_input_text(
        self,
        feature_description: str,
        dataset: str,
        retrieved_examples: list[Any],
    ) -> str:
        knowledge = self._knowledge_for_dataset(dataset)
        context = knowledge["context"]
        channel_knowledge = list(knowledge["channel_knowledge"])
        decision_guidance = self._combined_decision_guidance(knowledge["decision_guidance"])

        sections = [
            "[Current Sample Feature Description]",
            feature_description.strip(),
            "",
            "[Dataset Context]",
            "External knowledge supplied for this run." if self.knowledge_mode == "replace" else context,
        ]
        if self.knowledge_mode != "replace":
            sections.extend([
                "",
                "[Dataset / Channel Knowledge]",
                self._format_bullets(channel_knowledge),
            ])
        if self.knowledge_mode in {"append", "replace"} and self.external_knowledge:
            sections.extend([
                "",
                "[External Knowledge]",
                self.external_knowledge,
            ])
        sections.extend([
            "",
            "[Decision Guidance]",
            self._format_bullets(BASE_DECISION_GUIDANCE if self.knowledge_mode == "replace" else decision_guidance),
        ])
        if retrieved_examples:
            sections.extend([
                "",
                "[Optional Retrieved Evidence]",
                self._format_retrieved_examples(retrieved_examples),
            ])
        return "\n".join(sections).strip()

    def _knowledge_for_dataset(self, dataset: str) -> dict[str, Any]:
        return DEFAULT_DATASET_KNOWLEDGE.get(self._normalize_dataset_name(dataset), FALLBACK_DATASET_KNOWLEDGE)

    def _combined_decision_guidance(self, dataset_guidance: list[str]) -> list[str]:
        combined = list(BASE_DECISION_GUIDANCE)
        for item in dataset_guidance:
            if item not in combined:
                combined.append(item)
        return combined

    def _format_bullets(self, items: list[str]) -> str:
        if not items:
            return "- No additional knowledge provided."
        return "\n".join(f"- {item}" for item in items)

    def _format_retrieved_examples(self, retrieved_examples: list[Any]) -> str:
        lines = []
        for idx, example in enumerate(retrieved_examples, 1):
            if isinstance(example, dict):
                parts = []
                if "label" in example:
                    parts.append(f"label={example['label']}")
                if "feature_summary" in example:
                    parts.append(f"feature_summary={example['feature_summary']}")
                if "similarity_score" in example:
                    parts.append(f"similarity_score={example['similarity_score']}")
                text = "; ".join(parts) if parts else "retrieved example metadata unavailable"
            else:
                text = str(example)
            lines.append(f"- Retrieved example {idx}: {text}")
        return "\n".join(lines)

    def _get_retrieved_examples(self, sample: SensorSample, llm_sample: LLMSample) -> list[Any]:
        sample_examples = sample.meta.get("retrieved_examples") if isinstance(sample.meta, dict) else None
        llm_examples = llm_sample.meta.get("retrieved_examples") if isinstance(llm_sample.meta, dict) else None
        examples = sample_examples if sample_examples is not None else llm_examples
        if examples is None:
            return []
        if isinstance(examples, list):
            return examples
        return [examples]

    def _load_external_knowledge(self) -> str:
        chunks = []
        if self.knowledge_file is not None:
            chunks.append(self.knowledge_file.read_text(encoding="utf-8").strip())
        if self.knowledge_text:
            chunks.append(self.knowledge_text)
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def _knowledge_source(self) -> str:
        if self.knowledge_mode == "default" or not self.external_knowledge:
            return "default"
        if self.knowledge_file is not None:
            return str(self.knowledge_file)
        return "inline"

    def _resolve_dataset(self, sample: SensorSample) -> str:
        sample_dataset = getattr(sample, "dataset", None)
        if self._normalize_dataset_name(sample_dataset) not in {"", "UNKNOWN"}:
            return str(sample_dataset)
        if self._normalize_dataset_name(self.dataset) not in {"", "UNKNOWN"}:
            return str(self.dataset)
        return "UNKNOWN"

    def _validate_dataset_name(self, dataset: str | None) -> None:
        normalized = self._normalize_dataset_name(dataset)
        if normalized in {"", "UNKNOWN"}:
            return
        if normalized not in DEFAULT_DATASET_KNOWLEDGE:
            raise ValueError(
                f"Unsupported dataset for extra_knowledge input: {dataset}. "
                "Expected WESAD, HHAR, DREAMT, None, or UNKNOWN."
            )

    def _normalize_dataset_name(self, dataset: str | None) -> str:
        return str(dataset or "").replace("-", "").replace("_", "").strip().upper()

    def _normalize_knowledge_mode(self, knowledge_mode: str | None, has_external_knowledge: bool) -> str:
        if knowledge_mode is None:
            return "append" if has_external_knowledge else "default"
        mode = str(knowledge_mode).strip().lower()
        if mode not in {"default", "append", "replace"}:
            raise ValueError("knowledge_mode must be one of: default, append, replace.")
        return mode


__all__ = ["DEFAULT_DATASET_KNOWLEDGE", "ExtraKnowledgeInput"]
