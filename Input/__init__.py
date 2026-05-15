from .embedding_alignment import EmbeddingAlignmentInput
from .extra_knowledge import ExtraKnowledgeInput
from .feature_description import FeatureDescriptionInput, get_feature_description_builder
from .raw_data import RawDataInput


def build_input_provider(config: dict):
    kind = str(config.get("type", "feature_description")).strip().lower()
    dataset = config.get("dataset")

    if kind in {"raw", "raw_data"}:
        return RawDataInput()

    if kind in {"feature", "feature_description", "description"}:
        return get_feature_description_builder(dataset)

    if kind in {
        "embedding",
        "alignment",
        "embedding_alignment",
        "embedding / alignment",
        "encoded_time_series",
        "encoded-time-series",
        "encoded time series",
    }:
        return EmbeddingAlignmentInput(
            dataset=dataset or config.get("dataset_name", "WESAD"),
            data_path=config.get("data_path"),
            qa_path=config.get("qa_path"),
            label_map=config.get("label_map"),
            max_points=int(config.get("max_points", 256)),
            num_segments=int(config.get("num_segments", 4)),
            trend_threshold=float(config.get("trend_threshold", 0.25)),
            fluctuation_low_threshold=float(config.get("fluctuation_low_threshold", 0.08)),
            fluctuation_high_threshold=float(config.get("fluctuation_high_threshold", 0.35)),
            periodicity_threshold=float(config.get("periodicity_threshold", 0.35)),
            strong_periodicity_threshold=float(config.get("strong_periodicity_threshold", 0.60)),
            peak_z_threshold=float(config.get("peak_z_threshold", 2.5)),
            sudden_change_z_threshold=float(config.get("sudden_change_z_threshold", 2.5)),
            include_qa=bool(config.get("include_qa", True)),
            include_supporting_stats=bool(config.get("include_supporting_stats", True)),
            name=kind,
            strict=bool(config.get("strict", False)),
        )

    if kind in {"extra_knowledge", "knowledge", "extra knowledge"}:
        return ExtraKnowledgeInput(
            dataset=dataset,
            knowledge_file=config.get("knowledge_file"),
            knowledge_text=config.get("knowledge_text", ""),
        )

    raise ValueError(f"Unknown input type: {kind}")


__all__ = [
    "RawDataInput",
    "FeatureDescriptionInput",
    "EmbeddingAlignmentInput",
    "ExtraKnowledgeInput",
    "get_feature_description_builder",
    "build_input_provider",
]
