from __future__ import annotations

import argparse

from Dataset import DATASET_REGISTRY
from Input import INPUT_REGISTRY
from LM import LM_REGISTRY
from Output import OUTPUT_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular LLM experiments.")

    parser.add_argument(
        "-dataset",
        required=True,
        choices=sorted(DATASET_REGISTRY),
        help="Dataset name.",
    )

    parser.add_argument(
        "-Input",
        required=True,
        choices=sorted(INPUT_REGISTRY),
        help="Input type.",
    )

    parser.add_argument(
        "-LM",
        required=True,
        choices=sorted(LM_REGISTRY),
        help="LM usage.",
    )

    parser.add_argument(
        "-output",
        required=True,
        choices=sorted(OUTPUT_REGISTRY),
        help="Output type.",
    )

    parser.add_argument("-llm", default="qwen2.5-14b-instruct", help="LM Studio model name.")
    parser.add_argument("--api-url", default="http://127.0.0.1:1234/v1", help="LM Studio API URL.")
    parser.add_argument("--api-key", default="lm-studio", help="LM Studio API key.")

    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Label IDs to evaluate.",
    )

    parser.add_argument("--subjects", nargs="*", help="Evaluation subjects for direct or multi_agent runs.")
    parser.add_argument("--train-subjects", nargs="*", help="Few-shot training subjects.")
    parser.add_argument("--test-subjects", nargs="*", help="Few-shot testing subjects.")
    parser.add_argument("--few-shot-n-per-class", type=int, help="Few-shot examples per label.")

    parser.add_argument(
        "--few-shot-example-max-chars",
        type=int,
        help="Maximum characters kept from each few-shot example input.",
    )

    parser.add_argument(
        "--knowledge-file",
        help="Optional external knowledge file for -Input extra_knowledge.",
    )
    parser.add_argument(
        "--knowledge-text",
        default="",
        help="Optional inline external knowledge text for -Input extra_knowledge.",
    )
    parser.add_argument(
        "--knowledge-mode",
        choices=["default", "append", "replace"],
        help="How extra_knowledge uses external knowledge.",
    )

    parser.add_argument(
        "--balanced-per-label",
        type=int,
        default=None,
        help="Optional balanced debug samples per label.",
    )

    parser.add_argument("--log-every", type=int, default=10, help="Progress print interval.")

    return parser.parse_args()
