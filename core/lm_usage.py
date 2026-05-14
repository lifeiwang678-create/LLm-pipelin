from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

from .schema import Sample, label_block


class LMUsage(ABC):
    """Base interface for prompt/use strategy."""

    name: str

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        """Build the prompt sent to the LLM for one sample."""


class DirectUsage(LMUsage):
    name = "direct"

    def __init__(self, labels: list[int], input_name: str, output_instructions: str) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions

    def build_prompt(self, sample: Sample) -> str:
        return f"""You are given one physiological sample for classification.

Task:
Classify the state of this sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels)}

Important constraints:
- Use only the provided sample content.
- Do not use external medical knowledge.
- Do not guess randomly.
- Do not add extra explanation outside JSON.
- Process this sample independently.

{sample.input_text}

{self.output_instructions}"""


class FewShotUsage(LMUsage):
    name = "few_shot"

    def __init__(
        self,
        labels: list[int],
        input_name: str,
        output_instructions: str,
        examples: list[Sample],
        n_per_class: int = 2,
        random_state: int = 42,
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.examples = self._sample_examples(examples, n_per_class, random_state)

    def _sample_examples(self, examples: list[Sample], n_per_class: int, random_state: int) -> list[Sample]:
        rng = random.Random(random_state)
        picked = []
        for label in self.labels:
            class_examples = [sample for sample in examples if sample.label == label]
            rng.shuffle(class_examples)
            picked.extend(class_examples[: min(n_per_class, len(class_examples))])
        return picked

    def build_prompt(self, sample: Sample) -> str:
        example_blocks = []
        for idx, example in enumerate(self.examples, 1):
            example_blocks.append(
                f"""Example {idx}
{example.input_text}

Correct label:
- {example.label}"""
            )

        return f"""You are given physiological samples for classification.

Task:
Classify the state of the final sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels)}

Important constraints:
- Use only the provided sample content and few-shot examples.
- Do not use external medical knowledge.
- Do not add extra explanation outside JSON.
- Process the final sample independently based on the examples.

Few-shot examples:
{chr(10).join(example_blocks)}

Now classify the following sample.

{sample.input_text}

{self.output_instructions}"""


class MultiAgentUsage(LMUsage):
    name = "multi_agent"

    def __init__(
        self,
        labels: list[int],
        input_name: str,
        output_instructions: str,
        agents: list[str] | None = None,
        final_decider: str = "decision_maker",
    ) -> None:
        self.labels = labels
        self.input_name = input_name
        self.output_instructions = output_instructions
        self.agents = agents or ["signal_pattern_agent", "feature_trend_agent", "consistency_agent"]
        self.final_decider = final_decider

    def build_prompt(self, sample: Sample) -> str:
        agent_lines = "\n".join(f"- {agent}" for agent in self.agents)
        return f"""You are coordinating a multi-agent classification discussion for one physiological sample.

Task:
Classify the state of this sample using the selected input representation: {self.input_name}.

Labels:
{label_block(self.labels)}

Agents:
{agent_lines}
- {self.final_decider}: compare the agents' conclusions and choose the final label.

Important constraints:
- The agents must use only the provided sample content.
- Do not use external medical knowledge.
- Keep any reasoning internal unless the selected output format asks for explanation.
- The final answer must follow the requested JSON schema.

{sample.input_text}

{self.output_instructions}"""


class SensorLLMCheckpointUsage(LMUsage):
    name = "sensorllm_checkpoint"

    def __init__(
        self,
        labels: list[int],
        checkpoint_path: str,
        pt_encoder_backbone_ckpt: str,
        dataset: str = "wesad_binary",
        tokenize_method: str = "StanNormalizeUniformBins",
        torch_dtype: str = "float32",
        device: str = "auto",
        model_max_length: int = 2048,
        preprocess_type: str = "smry+trend+Q",
        add_ts_special_token_text: bool = False,
        label_id_map: dict[str, int] | None = None,
    ) -> None:
        self.labels = labels
        self.checkpoint_path = str(checkpoint_path)
        self.pt_encoder_backbone_ckpt = str(pt_encoder_backbone_ckpt)
        self.dataset = dataset
        self.tokenize_method = tokenize_method
        self.torch_dtype = torch_dtype
        self.device = device
        self.model_max_length = model_max_length
        self.preprocess_type = preprocess_type
        self.add_ts_special_token_text = add_ts_special_token_text
        self.label_id_map = label_id_map or {"0": 1, "1": 2}
        self._loaded = False
        self._dataset_cache = {}

    def build_prompt(self, sample: Sample) -> str:
        raise RuntimeError("SensorLLM checkpoint usage predicts from embeddings directly, not text prompts.")

    def predict(self, sample: Sample) -> dict:
        self._ensure_loaded()
        dataset = self._get_dataset(sample)
        item = dataset[int(sample.meta["data_index"])]

        self.torch.cuda.empty_cache() if self.device_obj.type == "cuda" else None
        with self.torch.inference_mode():
            batch = self._collate_one(item)
            outputs = self.model(**batch)
            logits = outputs.logits
            pred_class_id = int(self.torch.argmax(logits, dim=-1).item())
            probs = self.torch.softmax(logits, dim=-1)
            confidence = float(probs[0, pred_class_id].detach().cpu().item())

        label = int(self.label_id_map.get(str(pred_class_id), pred_class_id))
        label_name = self.model.config.id2label.get(pred_class_id, str(pred_class_id))
        return {
            "label": label,
            "valid": True,
            "parse_error": "",
            "explanation": (
                f"SensorLLM embedding-alignment checkpoint predicted {label_name} "
                f"from the time-series logits with confidence {confidence:.4f}."
            ),
            "raw_response": (
                f"SensorLLM logits={logits.detach().cpu().tolist()} "
                f"pred_class_id={pred_class_id} confidence={confidence:.4f}"
            ),
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        import torch
        import transformers
        import yaml
        from transformers import AutoConfig

        from sensorllm.data.stage2_dataset import MultiChannelTimeSeriesCLSDatasetStage2
        from sensorllm.model import SensorLLMStage2LlamaForSequenceClassification
        from sensorllm.model.chronos_model import ChronosConfig

        self.torch = torch
        self.transformers = transformers
        self.MultiChannelTimeSeriesCLSDatasetStage2 = MultiChannelTimeSeriesCLSDatasetStage2

        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_mapping.get(self.torch_dtype, torch.float32)
        if self.device == "auto":
            self.device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_obj = torch.device(self.device)

        with open("./sensorllm/model/ts_backbone.yaml", "r", encoding="utf-8") as f:
            dataset_configs = yaml.safe_load(f)
        dataset_config = dataset_configs[self.dataset]
        id2label = dataset_config["id2label"]
        label2id = {v: k for k, v in id2label.items()}

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            model_max_length=self.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.model = SensorLLMStage2LlamaForSequenceClassification.from_pretrained(
            self.checkpoint_path,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=dtype,
        )
        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device_obj)
        self.model.eval()

        self.model.get_model().load_pt_encoder_backbone_checkpoint(
            self.pt_encoder_backbone_ckpt,
            tc=self.tokenize_method,
            torch_dtype=dtype,
        )
        pt_backbone_config = AutoConfig.from_pretrained(self.pt_encoder_backbone_ckpt)
        chronos_config = ChronosConfig(**pt_backbone_config.chronos_config)
        chronos_config.tokenizer_class = self.tokenize_method
        self.chronos_tokenizer = chronos_config.create_tokenizer()

        self.model.initialize_tokenizer_ts_backbone_config_wo_embedding(self.tokenizer, dataset=self.dataset)
        self.model.get_model().load_start_end_tokens(dataset=self.dataset)
        self.ts_backbone_config = self.model.get_model().ts_backbone_config
        self._loaded = True

    def _get_dataset(self, sample: Sample):
        key = (sample.meta["data_path"], sample.meta["qa_path"])
        if key in self._dataset_cache:
            return self._dataset_cache[key]

        data_args = SimpleNamespace(
            preprocess_type=self.preprocess_type,
            shuffle=False,
            dataset=self.dataset,
            ts_backbone_config=self.ts_backbone_config,
            add_ts_special_token_text=self.add_ts_special_token_text,
        )
        dataset_config = self.model.config.id2label
        label2id = {v: int(k) for k, v in dataset_config.items()}
        dataset = self.MultiChannelTimeSeriesCLSDatasetStage2(
            data_path=sample.meta["data_path"],
            qa_path=sample.meta["qa_path"],
            tokenizer=self.tokenizer,
            chronos_tokenizer=self.chronos_tokenizer,
            split="eval",
            label2id=label2id,
            data_args=data_args,
        )
        self._dataset_cache[key] = dataset
        return dataset

    def _collate_one(self, item: dict) -> dict:
        input_ids = item["input_ids"].unsqueeze(0).to(self.device_obj)
        mts_token_ids = item["mts_token_ids"].unsqueeze(0).to(self.device_obj)
        mts_attention_mask = item["mts_attention_mask"].unsqueeze(0).to(self.device_obj)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id).to(self.device_obj),
            "mts_token_ids": mts_token_ids,
            "mts_attention_mask": mts_attention_mask,
            "labels": self.torch.tensor([int(item["labels"])], device=self.device_obj),
        }


def build_lm_usage(
    config: dict,
    labels: list[int],
    input_name: str,
    train_samples: list[Sample],
    output_instructions: str,
):
    kind = str(config.get("type", "direct")).lower()
    if input_name == "embedding_alignment":
        return SensorLLMCheckpointUsage(
            labels=labels,
            checkpoint_path=config.get("checkpoint_path", "sensorllm_wesad_binary_output_formal/fold_S2"),
            pt_encoder_backbone_ckpt=config.get("pt_encoder_backbone_ckpt", "D:/models/chronos-t5-large"),
            dataset=config.get("dataset", "wesad_binary"),
            tokenize_method=config.get("tokenize_method", "StanNormalizeUniformBins"),
            torch_dtype=config.get("torch_dtype", "float32"),
            device=config.get("device", "auto"),
            model_max_length=int(config.get("model_max_length", 2048)),
            preprocess_type=config.get("preprocess_type", "smry+trend+Q"),
            add_ts_special_token_text=bool(config.get("add_ts_special_token_text", False)),
            label_id_map=config.get("label_id_map"),
        )
    if kind == "direct":
        return DirectUsage(labels=labels, input_name=input_name, output_instructions=output_instructions)
    if kind in {"fewshot", "few_shot", "few-shot"}:
        return FewShotUsage(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            examples=train_samples,
            n_per_class=int(config.get("n_per_class", 2)),
            random_state=int(config.get("random_state", 42)),
        )
    if kind in {"multiagent", "multi_agent", "multi-agent"}:
        return MultiAgentUsage(
            labels=labels,
            input_name=input_name,
            output_instructions=output_instructions,
            agents=config.get("agents"),
            final_decider=config.get("final_decider", "decision_maker"),
        )
    raise ValueError(f"Unknown LM usage type: {kind}")
