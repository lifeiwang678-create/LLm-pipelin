from __future__ import annotations

from core.schema import Sample, label_block, label_rules_block


# multi_agent 前两步 (evidence / candidate_evaluation) 默认输出 token 上限。
# 旧实现复用 client 全局 max_tokens (128 / 384),会把结构化 JSON 截断成残缺串,
# 实测会让 encoded_time_series 的 multi_agent 直接 0% 准确率。
DEFAULT_INTERMEDIATE_MAX_TOKENS = 1024


class MultiAgentUsage:
    """Prompt-level 多步推理流水线 (依次 3 次 LLM 调用)。

    注意:名字叫 "multi_agent" 但实质上只是 multi-step prompting:
    三步用的是同一模型、同一温度、同一 prompt 风格,既没有多模型多样性,
    也没有 debate/voting。如果要变成真正的 multi-agent,需要至少引入
    不同模型、或不同温度采样、或最终阶段的投票/辩论机制。
    """

    name = "multi_agent"

    def __init__(
        self,
        labels: list[int],
        input_name: str,
        output_instructions: str,
        agents: list[str] | None = None,
        final_decider: str = "final_decision_agent",
        dataset: str | None = None,
        intermediate_max_tokens: int | None = DEFAULT_INTERMEDIATE_MAX_TOKENS,
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.dataset = dataset
        self.agents = agents or [
            "evidence_extraction_agent",
            "candidate_evaluation_agent",
        ]
        self.final_decider = final_decider
        # 前两步 (evidence / evaluation) 单独设上限。None 表示「按 client 默认」,
        # 一般不推荐传 None,除非已经把 client max_tokens 调到足够大。
        self.intermediate_max_tokens = (
            int(intermediate_max_tokens) if intermediate_max_tokens is not None else None
        )
        # 把中间步骤的原始输出暂存,由 runner 写到 JSONL 侧文件,
        # 不再像旧实现那样塞进 sample.meta (会污染 CSV 列、跨样本串台)。
        self.last_trace: dict | None = None

    def build_prompt(self, sample: Sample) -> str:
        """Single-call fallback prompt kept for compatibility."""
        agent_lines = "\n".join(f"- {agent}" for agent in self.agents)
        return f"""You are coordinating a multi-agent classification discussion for one time-series sample.

Task:
Classify the state of this sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Agents:
{agent_lines}
- {self.final_decider}: compare the agents' conclusions and choose the final label.

Important constraints:
- Use only the information provided in this prompt.
- Do not use knowledge outside the provided prompt.
- Apply the dataset-specific label rules exactly.
- Do not treat one high absolute sensor value as sufficient evidence for label 1 or the positive class.
- Keep any reasoning internal unless the selected output format asks for explanation.
- The final answer must follow the requested JSON schema.

{sample.input_text}

{self.output_instructions}"""

    def run_agent_pipeline(self, sample: Sample, llm_client) -> str:
        """三步 prompt 流水线。

        前两步 (evidence_extraction / candidate_evaluation) 要求结构化 JSON 输出,
        体量大,所以用 self.intermediate_max_tokens 显式放大;最终 decision 步走
        client 默认 max_tokens,与 direct / few_shot 保持一致。

        返回 final 步的回答,前两步原始输出存到 self.last_trace,由 runner 写到
        JSONL 侧文件,避免污染主 CSV 表格。
        """
        evidence_response = self._call_llm(
            llm_client,
            self.build_evidence_prompt(sample),
            max_tokens=self.intermediate_max_tokens,
        )
        evaluation_response = self._call_llm(
            llm_client,
            self.build_candidate_evaluation_prompt(sample, evidence_response),
            max_tokens=self.intermediate_max_tokens,
        )
        final_response = self._call_llm(
            llm_client,
            self.build_final_decision_prompt(sample, evidence_response, evaluation_response),
        )
        self.last_trace = {
            "evidence_extraction": evidence_response,
            "candidate_evaluation": evaluation_response,
        }
        return final_response

    def build_evidence_prompt(self, sample: Sample) -> str:
        return f"""You are given one time-series sample for classification.

Agent role:
Evidence Extraction Agent.

Task:
Extract relevant evidence for time-series classification.

Selected input representation:
{self.input_name}

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Allowed evidence sources:
- raw signal values
- statistical features
- temporal patterns
- dataset context
- dataset/channel knowledge
- external knowledge if included in the prompt
- retrieved evidence if included in the prompt

Constraints:
- Use only the information provided in this prompt.
- Do not use knowledge outside the provided prompt.
- Do not invent sensor values, features, labels, or retrieved examples.
- Current sample features are primary evidence.
- Dataset knowledge is supporting context only.
- Extract evidence for both allowed labels, including evidence against label 1 or the positive class.
- Treat high absolute sensor values cautiously when subject-specific baseline may differ.
- Output JSON only.

Sample content:
{sample.input_text}

Required JSON output:
{{
  "key_evidence": [
    {{
      "evidence": "...",
      "source": "current_sample | dataset_knowledge | external_knowledge | retrieved_evidence",
      "relevance": "..."
    }}
  ],
  "uncertainties": [
    "..."
  ]
}}"""

    def build_candidate_evaluation_prompt(self, sample: Sample, evidence_response: str) -> str:
        return f"""You are given one time-series sample for classification.

Agent role:
Candidate Evaluation Agent.

Task:
Evaluate each allowed label using the extracted evidence.

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Constraints:
- Consider only labels from the allowed label set.
- Do not invent new labels.
- Do not rely on one isolated feature alone.
- Down-weight weakly supported labels.
- If evidence is insufficient, state uncertainty.
- Use only the information provided in this prompt.
- Do not use knowledge outside the provided prompt.
- Apply the dataset-specific label rules exactly.
- Evaluate whether apparent label-1 evidence may also be compatible with label 0, movement, artifacts, or subject variation.
- Output JSON only.

Sample content:
{sample.input_text}

Agent 1 evidence response:
{evidence_response}

Required JSON output:
{{
  "label_evaluations": [
    {{
      "label": "<allowed label>",
      "supporting_evidence": ["..."],
      "contradicting_or_weak_evidence": ["..."],
      "assessment": "strong | moderate | weak | unclear"
    }}
  ],
  "candidate_ranking": [
    "<label1>",
    "<label2>"
  ],
  "remaining_uncertainties": [
    "..."
  ]
}}"""

    def build_final_decision_prompt(
        self,
        sample: Sample,
        evidence_response: str,
        evaluation_response: str,
    ) -> str:
        return f"""You are given one time-series sample for classification.

Agent role:
Final Decision Agent.

Task:
Choose the final label for the sample using the provided sample content and previous agent outputs.

Labels:
{label_block(self.labels, self.dataset)}

Dataset-specific label rules:
{label_rules_block(self.dataset)}

Critical constraints:
- Produce the final answer only.
- Follow the existing output instructions exactly.
- Do not output intermediate agent notes.
- Do not output anything outside the required JSON format.
- Consider only labels from the allowed label set.
- Do not use knowledge outside the provided prompt.
- Apply the dataset-specific label rules exactly.
- Do not default to label 1 or the positive class because a single sensor feature is high.
- Predict only the final label for the current sample.

Sample content:
{sample.input_text}

Agent 1 evidence response:
{evidence_response}

Agent 2 candidate evaluation response:
{evaluation_response}

{self.output_instructions}"""

    def _call_llm(self, llm_client, prompt: str, max_tokens: int | None = None) -> str:
        # 优先尝试给 client 传 per-call max_tokens (OpenAICompatibleClient 支持)。
        # 如果用户接了一个旧版/第三方 client 不支持该关键字,降级为不带覆盖再调一次,
        # 保证 multi_agent 仍然可用 (代价是中间步骤可能被截断,届时去看 last_trace 排查)。
        if hasattr(llm_client, "complete"):
            if max_tokens is not None:
                try:
                    return llm_client.complete(prompt, max_tokens=max_tokens)
                except TypeError:
                    return llm_client.complete(prompt)
            return llm_client.complete(prompt)
        if hasattr(llm_client, "generate"):
            return llm_client.generate(prompt)
        raise TypeError("LLM client must provide complete(prompt) or generate(prompt).")


__all__ = ["MultiAgentUsage"]
