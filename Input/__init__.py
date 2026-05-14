from .embedding_alignment import EmbeddingAlignmentInput
from .extra_knowledge import ExtraKnowledgeInput
from .feature_description import FeatureDescriptionInput
from .raw_data import RawDataInput


def build_input_provider(config: dict):
    kind = str(config.get("type", "feature_description")).lower()
    data_dir = config.get("data_dir", ".")

    if kind in {"raw", "raw_data"}:
        return RawDataInput(
            data_dir=data_dir,
            window_sec=float(config.get("window_sec", 10.0)),
            stride_sec=float(config.get("stride_sec", 15.0)),
        )

    if kind in {"feature", "feature_description", "description"}:
        return FeatureDescriptionInput(
            data_dir=data_dir,
            pattern=config.get("pattern", "*_features_paperstyle.csv"),
        )

    if kind in {"embedding", "alignment", "embedding_alignment", "embedding / alignment"}:
        return EmbeddingAlignmentInput(
            data_path=config.get("data_path", "sensorllm_wesad_binary_loso/fold_S2/eval_data.pkl"),
            qa_path=config.get("qa_path", "sensorllm_wesad_binary_loso/fold_S2/eval_qa.json"),
            label_map=config.get("label_map"),
        )

    if kind in {"extra_knowledge", "knowledge", "extra knowledge"}:
        return ExtraKnowledgeInput(
            data_dir=data_dir,
            pattern=config.get("pattern", "*_features_paperstyle.csv"),
            knowledge_file=config.get("knowledge_file"),
            knowledge_text=config.get("knowledge_text", ""),
        )

    raise ValueError(f"Unknown input type: {kind}")


__all__ = [
    "RawDataInput",
    "FeatureDescriptionInput",
    "EmbeddingAlignmentInput",
    "ExtraKnowledgeInput",
    "build_input_provider",
]

