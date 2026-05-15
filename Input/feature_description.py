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


class GenericFeatureDescriptionInput(BaseFeatureDescriptionInput):
    dataset_name = "generic"


# Backward-compatible name for code that imports FeatureDescriptionInput directly.
FeatureDescriptionInput = GenericFeatureDescriptionInput


def build_feature_description_input(dataset: str | None = None) -> BaseFeatureDescriptionInput:
    if dataset is None or str(dataset).strip() == "":
        raise ValueError("Feature description input requires config['dataset'] or config['input']['dataset'].")

    key = _normalize_dataset_name(dataset)
    if key == "WESAD":
        from .wesad_feature_description import WESADFeatureDescriptionInput

        return WESADFeatureDescriptionInput()
    if key == "HHAR":
        from .hhar_feature_description import HHARFeatureDescriptionInput

        return HHARFeatureDescriptionInput()
    if key == "DREAMT":
        from .dreamt_feature_description import DreaMTFeatureDescriptionInput

        return DreaMTFeatureDescriptionInput()
    raise ValueError(f"Unknown feature description dataset: {dataset}")


def _normalize_dataset_name(dataset: str) -> str:
    return str(dataset).replace("-", "").replace("_", "").strip().upper()


# Compatibility aliases for older local imports/tests.
extract_feature_dict = extract_signal_features


__all__ = [
    "BaseFeatureDescriptionInput",
    "FeatureDescriptionInput",
    "GenericFeatureDescriptionInput",
    "build_feature_description_input",
    "extract_feature_dict",
    "format_feature_block",
]
