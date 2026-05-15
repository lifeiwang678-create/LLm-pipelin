"""Compatibility forwarding module for output handlers.

The official output implementations live in the top-level `Output/` package.
This module intentionally keeps no independent parser logic.
"""

from Output import LabelExplanationOutput, LabelOnlyOutput, build_output_handler

__all__ = ["LabelOnlyOutput", "LabelExplanationOutput", "build_output_handler"]
