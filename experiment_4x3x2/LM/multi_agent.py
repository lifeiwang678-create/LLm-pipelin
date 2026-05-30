from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from core.schema import Sample, decision_guidance_block, label_block, label_rules_block


# Intermediate agents usually need more output tokens than the final classifier.
# If this is too small, JSON outputs from each agent may be truncated.
DEFAULT_INTERMEDIATE_MAX_TOKENS = 1024


class MultiAgentUsage:
    """Multi-view prompt-level multi-agent classification.

    This version assigns different input representations to different agents:

    - signal_pattern_agent:
      raw_data + encoded_time_series
    - feature_statistic_agent:
      feature_description
    - knowledge_agent:
      extra_knowledge
    - label_rule_agent:
      dataset label rules + safe sample metadata only
    - judge_agent:
      all available input views + independent agent outputs

    It still uses the same llm_client interface:
    run_agent_pipeline(sample, llm_client) -> str

    Required multi-view sample format:
    sample.metadata["input_views"] = {
        "raw_data": "...",
        "encoded_time_series": "...",
        "feature_description": "...",
        "extra_knowledge": "...",
    }

    Backward compatibility:
    If input_views is not available, the current sample.input_text is used as the
    only available input view under self.input_name.
    """

    name = "multi_agent"

    def __init__(
        self,
        labels: list[int],
        input_name: str,
        output_instructions: str,
        agents: list[str] | None = None,
        final_decider: str = "judge_agent",
        dataset: str | None = None,
        intermediate_max_tokens: int | None = DEFAULT_INTERMEDIATE_MAX_TOKENS,
        agent_temperatures: dict[str, float] | None = None,
        agent_views: dict[str, list[str]] | None = None,
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.dataset = dataset

        self.agents = agents or [
            "signal_pattern_agent",
            "feature_statistic_agent",
            "knowledge_agent",
            "label_rule_agent",
        ]
        self.final_decider = final_decider

        self.agent_views = agent_views or {
            "signal_pattern_agent": ["raw_data", "encoded_time_series"],
            "feature_statistic_agent": ["feature_description"],
            "knowledge_agent": ["extra_knowledge"],
            "label_rule_agent": ["metadata"],
        }

        self.intermediate_max_tokens = (
            int(intermediate_max_tokens) if intermediate_max_tokens is not None else None
        )

        # If llm_client supports temperature as a per-call argument, this gives
        # mild sampling diversity. If not supported, _call_llm falls back safely.
        self.agent_temperatures = agent_temperatures or {
            "signal_pattern_agent": 0.2,
            "feature_statistic_agent": 0.2,
            "knowledge_agent": 0.2,
            "label_rule_agent": 0.0,
        }

        # Runner can write this to a side JSONL file.
        self.last_trace: dict[str, Any] | None = None

    def build_prompt(self, sample: Sample) -> str:
        """Single-call fallback prompt kept for compatibility."""
        agent_lines = "\n".join(f"- {agent}: {self._agent_focus(agent)}" for agent in self.agents)
        return f"""You are coordinating a multi-view multi-agent classification discussion for one time-series sample.

Task:
Classify the state of this sample by comparing all available input representations.

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Decision calibration:
{decision_guidance_block(self.dataset)}

Agents:
{agent_lines}
- {self.final_decider}: compare independent agent votes and choose the final label.

Important constraints:
- Use only the information provided in this prompt.
- Do not use knowledge outside the provided prompt.
- Apply the dataset-specific label rules exactly.
- Do not use either label as a default fallback.
- The final answer must follow the requested JSON schema.

All available input views:
{self._build_all_views_text(sample)}

{self.output_instructions}"""

    def run_agent_pipeline(self, sample: Sample, llm_client) -> str:
        """Run multi-view independent agents, aggregate votes, then call a judge."""
        agent_outputs: list[dict[str, Any]] = []

        for agent_name in self.agents:
            response = self._call_llm(
                llm_client,
                self.build_agent_prompt(sample, agent_name),
                max_tokens=self.intermediate_max_tokens,
                temperature=self.agent_temperatures.get(agent_name),
            )
            vote = self._extract_label_vote(response)
            agent_outputs.append(
                {
                    "agent": agent_name,
                    "views": self.agent_views.get(agent_name, []),
                    "vote": vote,
                    "response": response,
                }
            )

        majority_vote = self._majority_vote([item["vote"] for item in agent_outputs])

        final_response = self._call_llm(
            llm_client,
            self.build_judge_prompt(sample, agent_outputs, majority_vote),
        )

        self.last_trace = {
            "mode": "multi_view_agents_with_judge",
            "agent_outputs": agent_outputs,
            "majority_vote": majority_vote,
            "final_response": final_response,
        }
        return final_response

    def build_agent_prompt(self, sample: Sample, agent_name: str) -> str:
        agent_focus = self._agent_focus(agent_name)
        label_values = ", ".join(str(label) for label in self.labels)
        source_text = self._build_agent_source_text(sample, agent_name)
        used_views = self.agent_views.get(agent_name, [])

        return f"""You are one independent agent in a multi-agent classification system.

Agent name:
{agent_name}

Agent focus:
{agent_focus}

Task:
Classify the current time-series sample from your assigned input view.

Assigned input view:
{source_text}

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Decision calibration:
{decision_guidance_block(self.dataset)}

Constraints:
- Work independently. Do not assume other agents' conclusions.
- Use only the input view assigned to you.
- Do not use knowledge outside the provided prompt.
- Do not invent sensor values, features, labels, or retrieved examples.
- Apply the dataset-specific label rules exactly.
- Consider both positive and negative evidence.
- Do not rely on one isolated high absolute sensor value for either label.
- Do not treat either label as a default fallback.
- Predict label 1 only when its overall support clearly outweighs label 0 support.
- Predict label 0 when label 0 support clearly outweighs label 1 support.
- If your assigned input view is missing or weak, report low confidence rather than inventing evidence.
- Output JSON only.

Required JSON output:
{{
  "agent": "{agent_name}",
  "predicted_label": <one of: {label_values}>,
  "confidence": "high | medium | low",
  "used_input_views": {json.dumps(used_views, ensure_ascii=False)},
  "supporting_evidence": [
    "..."
  ],
  "contradicting_or_weak_evidence": [
    "..."
  ],
  "uncertainties": [
    "..."
  ]
}}"""

    def build_judge_prompt(
        self,
        sample: Sample,
        agent_outputs: list[dict[str, Any]],
        majority_vote: int | None,
    ) -> str:
        compact_outputs = [
            {
                "agent": item["agent"],
                "views": item.get("views", []),
                "vote": item["vote"],
                "response": item["response"],
            }
            for item in agent_outputs
        ]
        agent_outputs_text = json.dumps(compact_outputs, ensure_ascii=False, indent=2)
        majority_text = "unknown" if majority_vote is None else str(majority_vote)

        return f"""You are the final judge in a multi-view multi-agent classification system.

Task:
Choose the final label for the current sample by comparing:
- all available input views
- independent agent predictions
- supporting and contradicting evidence from each agent
- the majority vote

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Decision calibration:
{decision_guidance_block(self.dataset)}

Critical constraints:
- Produce the final answer only.
- Follow the existing output instructions exactly.
- Do not output intermediate agent notes.
- Do not output anything outside the required JSON format.
- Consider only labels from the allowed label set.
- Do not use knowledge outside the provided prompt.
- Apply the dataset-specific label rules exactly.
- The majority vote is useful evidence, but it is not automatically correct.
- Override the majority vote only when the input views and label rules support the override.
- Do not use either label as a default fallback.
- Do not use label 1 as a shortcut because a single sensor feature is high.
- If an agent predicts label 1 with concrete multi-cue evidence, evaluate that evidence directly before choosing label 0.
- If an agent predicts label 0 with concrete counter-evidence, evaluate that evidence directly before choosing label 1.

All available input views:
{self._build_all_views_text(sample)}

Independent agent outputs:
{agent_outputs_text}

Parsed majority vote:
{majority_text}

{self.output_instructions}"""

    def _agent_focus(self, agent_name: str) -> str:
        focuses = {
            "signal_pattern_agent": (
                "Inspect raw temporal patterns and encoded time-series representations. "
                "Focus on trends, bursts, stability, movement-like patterns, artifacts, "
                "and temporal consistency."
            ),
            "feature_statistic_agent": (
                "Inspect statistical feature descriptions. Focus on mean, standard deviation, "
                "range, frequency-domain features, cross-channel consistency, and feature-level evidence."
            ),
            "knowledge_agent": (
                "Inspect extra knowledge provided for the sample. Use it only as supporting context, "
                "not as a replacement for the actual dataset label rules."
            ),
            "label_rule_agent": (
                "Apply the dataset-specific label definition strictly using only label rules and safe sample metadata. "
                "Do not infer from unavailable raw signals, feature values, or external knowledge."
            ),
            "consistency_agent": (
                "Check whether different channels and features support the same label or conflict "
                "with each other. Penalize decisions based on a single weak cue."
            ),
        }
        return focuses.get(
            agent_name,
            "Analyze the assigned input view independently and produce a label prediction with evidence.",
        )

    def _get_input_views(self, sample: Sample) -> dict[str, Any]:
        """Collect all available input representations for this sample.

        Preferred format:
        sample.metadata["input_views"] = {
            "raw_data": "...",
            "encoded_time_series": "...",
            "feature_description": "...",
            "extra_knowledge": "...",
        }

        Also supports sample.input_views / sample.inputs / sample.representations.
        """
        views: dict[str, Any] = {}

        metadata = getattr(sample, "metadata", None)
        if isinstance(metadata, dict):
            for key in ["input_views", "inputs", "representations", "input_texts"]:
                value = metadata.get(key)
                if isinstance(value, dict):
                    views.update(value)

            for key in [
                "raw_data",
                "encoded_time_series",
                "feature_description",
                "extra_knowledge",
            ]:
                if key in metadata:
                    views[key] = metadata[key]

                text_key = f"{key}_text"
                if text_key in metadata:
                    views[key] = metadata[text_key]

        for attr in [
            "input_views",
            "inputs",
            "representations",
            "input_texts",
        ]:
            value = getattr(sample, attr, None)
            if isinstance(value, dict):
                views.update(value)

        # Backward compatibility:
        # If the runner still passes only one input representation, keep it usable.
        if self.input_name and hasattr(sample, "input_text"):
            views.setdefault(self.input_name, getattr(sample, "input_text"))

        return views

    def _build_agent_source_text(self, sample: Sample, agent_name: str) -> str:
        requested_views = self.agent_views.get(agent_name, [])
        views = self._get_input_views(sample)

        if requested_views == ["metadata"]:
            return self._sample_metadata_text(sample)

        blocks: list[str] = []
        missing: list[str] = []

        for view_name in requested_views:
            value = views.get(view_name)
            if value is not None and self._to_text(value).strip():
                blocks.append(
                    f"### {view_name}\n{self._to_text(value)}"
                )
            else:
                missing.append(view_name)

        if missing:
            blocks.append(
                "### missing_input_views\n"
                + json.dumps(missing, ensure_ascii=False)
            )

        if not blocks:
            blocks.append(
                "### fallback_selected_input\n"
                + self._to_text(getattr(sample, "input_text", ""))
            )

        return "\n\n".join(blocks)

    def _build_all_views_text(self, sample: Sample) -> str:
        views = self._get_input_views(sample)
        blocks: list[str] = []

        for view_name in [
            "raw_data",
            "encoded_time_series",
            "feature_description",
            "extra_knowledge",
        ]:
            value = views.get(view_name)
            if value is not None and self._to_text(value).strip():
                blocks.append(
                    f"### {view_name}\n{self._to_text(value)}"
                )

        # Include any additional representation keys not covered above.
        known_keys = {
            "raw_data",
            "encoded_time_series",
            "feature_description",
            "extra_knowledge",
        }
        for view_name, value in views.items():
            if view_name in known_keys:
                continue
            if value is not None and self._to_text(value).strip():
                blocks.append(
                    f"### {view_name}\n{self._to_text(value)}"
                )

        blocks.append("### metadata\n" + self._sample_metadata_text(sample))

        if not blocks:
            return self._to_text(getattr(sample, "input_text", ""))

        return "\n\n".join(blocks)

    def _sample_metadata_text(self, sample: Sample) -> str:
        """Return safe metadata for prompts.

        Ground-truth labels must not be exposed to the model. Therefore this method
        removes label/target-like fields from sample metadata.
        """
        metadata = getattr(sample, "metadata", None)

        excluded_keys = {
            "input_views",
            "inputs",
            "representations",
            "input_texts",
            "raw_data",
            "encoded_time_series",
            "feature_description",
            "extra_knowledge",
            "raw_data_text",
            "encoded_time_series_text",
            "feature_description_text",
            "extra_knowledge_text",
        }

        target_like_patterns = [
            "label",
            "target",
            "ground_truth",
            "truth",
            "answer",
            "class",
            "y_true",
            "y",
        ]

        safe_metadata: dict[str, Any] = {}

        if isinstance(metadata, dict):
            for key, value in metadata.items():
                key_lower = str(key).lower()

                if key in excluded_keys:
                    continue

                if any(pattern == key_lower or pattern in key_lower for pattern in target_like_patterns):
                    continue

                safe_metadata[key] = value

        for attr in [
            "dataset",
            "subject_id",
            "subject",
            "sample_id",
            "window_id",
            "start_time",
            "end_time",
            "timestamp",
        ]:
            if hasattr(sample, attr):
                safe_metadata[attr] = getattr(sample, attr)

        if not safe_metadata:
            return "No safe sample metadata is available."

        return json.dumps(safe_metadata, ensure_ascii=False, indent=2)

    def _to_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

    def _extract_label_vote(self, response: str) -> int | None:
        """Extract predicted_label from an agent JSON response.

        Returns None if the output is invalid or the label is outside the allowed set.
        This keeps invalid intermediate agent outputs visible in last_trace without crashing
        the whole experiment.
        """
        label_set = set(self.labels)

        try:
            data = json.loads(response)
            value = data.get("predicted_label")
            label = self._coerce_label(value)
            return label if label in label_set else None
        except Exception:
            pass

        # Fallback for slightly malformed JSON.
        patterns = [
            r'"predicted_label"\s*:\s*"?(-?\d+)"?',
            r"predicted_label\s*[:=]\s*" + r'"?(-?\d+)"?',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                label = self._coerce_label(match.group(1))
                return label if label in label_set else None

        return None

    def _coerce_label(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if re.fullmatch(r"-?\d+", value):
                return int(value)
        return None

    def _majority_vote(self, votes: list[int | None]) -> int | None:
        valid_votes = [vote for vote in votes if vote in set(self.labels)]
        if not valid_votes:
            return None

        counts = Counter(valid_votes)
        top_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == top_count]

        # Tie: no majority. The judge must decide from evidence.
        if len(winners) != 1:
            return None
        return winners[0]

    def _call_llm(
        self,
        llm_client,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Call different LLM client implementations safely.

        Supports clients with complete(prompt), complete(prompt, max_tokens=...),
        complete(prompt, temperature=...), or complete(prompt, max_tokens=..., temperature=...).
        Falls back if a client does not support optional per-call arguments.
        """
        if hasattr(llm_client, "complete"):
            kwargs: dict[str, Any] = {}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            if kwargs:
                try:
                    return llm_client.complete(prompt, **kwargs)
                except TypeError:
                    # Try max_tokens only because the existing client may support this
                    # but not temperature.
                    if max_tokens is not None:
                        try:
                            return llm_client.complete(prompt, max_tokens=max_tokens)
                        except TypeError:
                            pass
                    return llm_client.complete(prompt)
            return llm_client.complete(prompt)

        if hasattr(llm_client, "generate"):
            return llm_client.generate(prompt)

        raise TypeError("LLM client must provide complete(prompt) or generate(prompt).")


__all__ = ["MultiAgentUsage"]
