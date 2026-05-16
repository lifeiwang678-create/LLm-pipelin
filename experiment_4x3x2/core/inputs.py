"""Compatibility forwarding module for input providers.

The official input implementations live in the top-level `Input/` package.
This module intentionally keeps no independent input logic.
"""

from Input import (
    EmbeddingAlignmentInput,
    ExtraKnowledgeInput,
    FeatureDescriptionInput,
    RawDataInput,
    build_input_provider,
    get_feature_description_builder,
)

__all__ = [
    "RawDataInput",
    "FeatureDescriptionInput",
    "EmbeddingAlignmentInput",
    "ExtraKnowledgeInput",
    "get_feature_description_builder",
    "build_input_provider",
]
