from __future__ import annotations

from core.schema import Sample, label_block


class MultiAgentUsage:
    name = "multi_agent"

    def __init__(
        self,
        labels: list[int],
        input_name: str,
        output_instructions: str,
        agents: list[str] | None = None,
        final_decider: str = "final_decision_agent",
        dataset: str | None = None,
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

    def build_prompt(self, sample: Sample) -> str:
        """Single-call fallback prompt kept for compatibility."""
        agent_lines = "\n".join(f"- {agent}" for agent in self.agents)
        return f"""You are coordinating a multi-agent classification discussion for one time-series sample.

Task:
Classify the state of this sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels, self.dataset)}

Agents:
{agent_lines}
- {self.final_decider}: compare the agents' conclusions and choose the final label.

Important constraints:
- Use only the information provided in this prompt.
- Do not use knowledge outside the provided prompt.
- Keep any reasoning internal unless the selected output format asks for explanation.
- The final answer must follow the requested JSON schema.

{sample.input_text}

{self.output_instructions}"""

    def run_agent_pipeline(self, sample: Sample, llm_client) -> str:
        """Run true prompt-level multi-agent reasoning with three LLM calls."""
        evidence_response = self._call_llm(llm_client, self.build_evidence_prompt(sample))
        evaluation_response = self._call_llm(
            llm_client,
            self.build_candidate_evaluation_prompt(sample, evidence_response),
        )
        final_response = self._call_llm(
            llm_client,
            self.build_final_decision_prompt(sample, evidence_response, evaluation_response),
        )
        if hasattr(sample, "meta") and isinstance(sample.meta, dict):
            sample.meta["agent_trace"] = {
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

Constraints:
- Consider only labels from the allowed label set.
- Do not invent new labels.
- Do not rely on one isolated feature alone.
- Down-weight weakly supported labels.
- If evidence is insufficient, state uncertainty.
- Use only the information provided in this prompt.
- Do not use knowledge outside the provided prompt.
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

Critical constraints:
- Produce the final answer only.
- Follow the existing output instructions exactly.
- Do not output intermediate agent notes.
- Do not output anything outside the required JSON format.
- Consider only labels from the allowed label set.
- Do not use knowledge outside the provided prompt.
- Predict only the final label for the current sample.

Sample content:
{sample.input_text}

Agent 1 evidence response:
{evidence_response}

Agent 2 candidate evaluation response:
{evaluation_response}

{self.output_instructions}"""

    def _call_llm(self, llm_client, prompt: str) -> str:
        if hasattr(llm_client, "complete"):
            return llm_client.complete(prompt)
        if hasattr(llm_client, "generate"):
            return llm_client.generate(prompt)
        raise TypeError("LLM client must provide complete(prompt) or generate(prompt).")


__all__ = ["MultiAgentUsage"]
