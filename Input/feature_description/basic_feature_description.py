from __future__ import annotations

from core.schema import LLMSample, SensorSample

from .feature_functions import FeatureDict, extract_signal_features, format_feature_block


class BaseFeatureDescriptionInput:
    name = "feature_description"
    dataset_name = "generic"
    signal_order: list[str] | None = None
    title = "Input feature description:"
    sections: dict[str, list[str]] | None = None

    def build_input(self, sample: SensorSample) -> str:
        features = self.extract_features(sample)
        return self.format_features(features)

    def transform(self, sample: SensorSample) -> LLMSample:
        meta = dict(sample.meta)
        meta["input_type"] = self.name
        meta["feature_description_dataset"] = self.dataset_name
        return LLMSample(
            dataset=sample.dataset,
            subject=sample.subject,
            label=sample.label,
            input_text=self.build_input(sample),
            meta=meta,
        )

    def transform_all(self, samples: list[SensorSample]) -> list[LLMSample]:
        return [self.transform(sample) for sample in samples]

    def extract_features(self, sample: SensorSample) -> FeatureDict:
        return extract_signal_features(sample.signals, self.signal_order)

    def format_features(self, features: FeatureDict) -> str:
        return format_feature_block(features, title=self.title, sections=self.sections)


class BasicFeatureDescriptionInput(BaseFeatureDescriptionInput):
    dataset_name = "basic"


# Backward-compatible name for code that imports FeatureDescriptionInput directly.
FeatureDescriptionInput = BasicFeatureDescriptionInput


# Compatibility alias for older local imports/tests.
extract_feature_dict = extract_signal_features


__all__ = [
    "BaseFeatureDescriptionInput",
    "BasicFeatureDescriptionInput",
    "FeatureDescriptionInput",
    "extract_feature_dict",
    "format_feature_block",
]
