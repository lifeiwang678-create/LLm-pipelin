from .embedding_alignment import EmbeddingAlignmentInput
from .extra_knowledge import ExtraKnowledgeInput
from .feature_description import FeatureDescriptionInput, get_feature_description_builder
from .raw_data import RawDataInput


def build_input_provider(config: dict):
    kind = str(config.get("type", "feature_description")).lower()
    dataset = config.get("dataset")

    if kind in {"raw", "raw_data"}:
        return RawDataInput()

    if kind in {"feature", "feature_description", "description"}:
        return get_feature_description_builder(dataset)

    if kind in {"embedding", "alignment", "embedding_alignment", "embedding / alignment"}:
        return EmbeddingAlignmentInput(
            data_path=config.get("data_path", "sensorllm_wesad_binary_loso/fold_S2/eval_data.pkl"),
            qa_path=config.get("qa_path", "sensorllm_wesad_binary_loso/fold_S2/eval_qa.json"),
            label_map=config.get("label_map"),
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
