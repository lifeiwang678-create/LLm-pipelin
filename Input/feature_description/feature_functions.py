from __future__ import annotations

import numpy as np

from core.signal_utils import describe_1d


FeatureStats = dict[str, float | str]
FeatureDict = dict[str, FeatureStats]


def ordered_signal_names(signals: dict, preferred_order: list[str] | None = None) -> list[str]:
    preferred_order = preferred_order or []
    ordered = [name for name in preferred_order if name in signals]
    ordered.extend(sorted(name for name in signals if name not in set(ordered)))
    return ordered


def extract_signal_features(signals: dict, signal_order: list[str] | None = None) -> FeatureDict:
    features: FeatureDict = {}
    for name in ordered_signal_names(signals, signal_order):
        arr = np.asarray(signals[name])
        features.update(extract_one_signal_features(name, arr))
    return features


def extract_one_signal_features(name: str, arr: np.ndarray) -> FeatureDict:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return {
            f"{name}_x": describe_1d(arr[:, 0]),
            f"{name}_y": describe_1d(arr[:, 1]),
            f"{name}_z": describe_1d(arr[:, 2]),
            f"{name}_magnitude": describe_1d(np.linalg.norm(arr, axis=1)),
        }
    if arr.ndim == 2:
        return {
            f"{name}_{idx}": describe_1d(arr[:, idx])
            for idx in range(arr.shape[1])
        }
    return {name: describe_1d(arr)}


def format_feature_block(
    features: FeatureDict,
    title: str = "Input feature description:",
    sections: dict[str, list[str]] | None = None,
) -> str:
    if not features:
        return f"{title}\n- no numeric signals available"

    if not sections:
        return "\n".join([title, *[_format_feature_line(name, stats) for name, stats in features.items()]])

    lines = [title]
    emitted: set[str] = set()
    for section, prefixes in sections.items():
        section_lines = []
        for name, stats in features.items():
            if name in emitted or not _matches_prefix(name, prefixes):
                continue
            section_lines.append(_format_feature_line(name, stats))
            emitted.add(name)
        if section_lines:
            lines.append("")
            lines.append(f"{section}:")
            lines.extend(section_lines)

    remaining = [name for name in features if name not in emitted]
    if remaining:
        lines.append("")
        lines.append("Other signals:")
        lines.extend(_format_feature_line(name, features[name]) for name in remaining)
    return "\n".join(lines)


def _matches_prefix(name: str, prefixes: list[str]) -> bool:
    name_lower = name.lower()
    return any(name_lower == prefix.lower() or name_lower.startswith(f"{prefix.lower()}_") for prefix in prefixes)


def _format_feature_line(name: str, stats: FeatureStats) -> str:
    return (
        "- "
        f"{name}: "
        f"mean={_fmt(stats['mean'])}, "
        f"std={_fmt(stats['std'])}, "
        f"min={_fmt(stats['min'])}, "
        f"max={_fmt(stats['max'])}, "
        f"p25={_fmt(stats['p25'])}, "
        f"p75={_fmt(stats['p75'])}, "
        f"trend={stats['trend']}"
    )


def _fmt(value: float | str) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "FeatureDict",
    "FeatureStats",
    "extract_one_signal_features",
    "extract_signal_features",
    "format_feature_block",
    "ordered_signal_names",
]
