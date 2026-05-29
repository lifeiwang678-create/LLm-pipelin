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

    parser.add_argument("-llm", default="qwen2.5-14b-instruct", help="OpenAI-compatible model name.")
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:1234/v1",
        help="OpenAI-compatible API base URL, such as vLLM or LM Studio.",
    )
    parser.add_argument(
        "--api-key",
        default="lm-studio",
        help="API key for the OpenAI-compatible server. Local vLLM usually ignores this unless configured.",
    )
    parser.add_argument(
        "--data-dir",
        help="Optional dataset directory override. For DREAMT, use the folder containing data_64Hz or data_64Hz itself.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row limit for large CSV dataset loaders such as HHAR. Use only for debugging.",
    )
    parser.add_argument(
        "--use-processed",
        action="store_true",
        help="Load preprocessed SensorSample windows from Processed/ instead of reprocessing raw data.",
    )
    parser.add_argument(
        "--processed-dir",
        default="Processed",
        help="Directory containing preprocessed dataset window files.",
    )
    parser.add_argument(
        "--processed-file",
        help="Optional explicit preprocessed .pkl file for this run.",
    )
    parser.add_argument(
        "--use-input-cache",
        action="store_true",
        help="Load precomputed LLMSample input_text from cache instead of transforming SensorSamples.",
    )
    parser.add_argument(
        "--input-cache-dir",
        default="Processed",
        help="Directory containing precomputed input cache files.",
    )
    parser.add_argument(
        "--input-cache-file",
        help="Optional explicit input cache .pkl file for this run.",
    )
    parser.add_argument(
        "--train-input-cache-file",
        help="Optional input cache .pkl file used only for few-shot training examples.",
    )
    parser.add_argument(
        "--eval-input-cache-file",
        help="Optional input cache .pkl file used only for evaluation samples.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Label IDs to evaluate. Defaults to the maintained binary labels: 0 1.",
    )

    parser.add_argument("--subjects", nargs="*", help="Evaluation subjects for direct or multi_agent runs.")
    parser.add_argument("--train-subjects", nargs="*", help="Few-shot training subjects.")
    parser.add_argument("--test-subjects", nargs="*", help="Few-shot testing subjects.")
    parser.add_argument(
        "--subject-split",
        choices=["subject_independent", "all"],
        default=None,
        help=(
            "Subject split policy. Defaults to subject_independent, which evaluates "
            "only held-out test subjects. Use all only for debugging or legacy runs."
        ),
    )
    parser.add_argument("--few-shot-n-per-class", type=int, help="Few-shot examples per label.")
    parser.add_argument(
        "--few-shot-example-selection",
        choices=["leave_one_subject_out", "class_balanced"],
        default="leave_one_subject_out",
        help=(
            "Few-shot sampling policy. Defaults to leave_one_subject_out: for each eval subject, "
            "exclude that subject and sample examples from other subjects."
        ),
    )
    parser.add_argument(
        "--few-shot-example-subjects",
        type=int,
        default=5,
        help="Number of non-evaluation subjects sampled for leave-one-subject-out few-shot examples.",
    )
    parser.add_argument(
        "--few-shot-examples-per-subject-per-label",
        type=int,
        default=1,
        help="Examples sampled from each selected subject for each label in leave-one-subject-out mode.",
    )

    parser.add_argument(
        "--few-shot-example-max-chars",
        type=int,
        help="Maximum characters kept from each few-shot example input.",
    )

    # multi_agent 的前两个 agent (evidence / candidate_evaluation) 输出结构化 JSON,
    # 用全局 max_tokens (128/384) 会被截断成残缺 JSON,导致后续 agent 看到脏输入。
    # 这里允许在 CLI 显式调大,默认 1024。
    parser.add_argument(
        "--multi-agent-intermediate-max-tokens",
        type=int,
        default=None,
        help="Max tokens for multi_agent intermediate steps (evidence/evaluation). "
             "Defaults to 512 for faster full-combination runs.",
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

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent LLM requests. Defaults to 1 for the original serial behavior.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Progress print interval.")

    return parser.parse_args()
