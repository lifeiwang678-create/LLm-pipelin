"""Compatibility forwarding module for LM usage strategies.

The official LM usage implementations live in the top-level `LM/` package.
This module intentionally keeps no independent prompt or inference logic.
"""

from LM import DirectUsage, FewShotUsage, MultiAgentUsage, build_lm_usage

__all__ = ["DirectUsage", "FewShotUsage", "MultiAgentUsage", "build_lm_usage"]
