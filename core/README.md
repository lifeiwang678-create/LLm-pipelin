# Modular LLM Experiments

This folder contains shared orchestration code for the modular experiment framework:

- The official method logic lives in the top-level `Dataset/`, `Input/`, `LM/`, `Output/`, and `Evaluation/` folders.
- `runner.py` is the only experiment execution path.
- `main.py` parses command-line arguments and calls `core.runner`.
- `run_experiment.py` reads JSON/YAML configs, expands optional grids, and calls the same `core.runner`.
- `inputs.py`, `lm_usage.py`, and `outputs.py` are compatibility forwarding modules. They re-export the official top-level modules and keep no independent experiment logic.
- The legacy `experiment_pipeline/` path is removed from the maintained framework. Do not use it as an experiment entry.

Run an experiment with:

```powershell
python main.py -dataset WESAD -Input raw_data -LM direct -output label_only
```

By default, `main.py` does not sample or balance the evaluation set. Add `--balanced-per-label` only for debug runs.

Install dependencies from the repository root before running experiments:

```powershell
pip install -r requirements.txt
```

By default, `main.py` saves compact CSV files to `Results/`, for example:

```text
WESAD_raw_data_direct_label_only_20260512213815.csv
```

Config files can override the result folder with `output_dir`.

The config-based runner uses the same shared execution logic:

```powershell
python run_experiment.py --config configs/example_experiment.json
```

Change combinations by editing the config:

```json
{
  "input": {"type": "raw_data"},
  "lm_usage": {"type": "multi_agent"},
  "output": {"type": "label_explanation"}
}
```

For 3-class experiments, change `labels` to `[1, 2, 3]`.

Label names are selected by dataset:

| Dataset | Label 1 | Label 2 | Label 3 |
| -- | -- | -- | -- |
| `WESAD` | Baseline | Stress | Amusement |
| `HHAR` | Static activity | Dynamic activity | Stairs activity |
| `DREAMT` | Baseline/Neutral/Relax | Stress | Amusement/Happy |

These names are used in LM prompts and in `classification_report`; predictions still use numeric label IDs.

`embedding_alignment` / `encoded_time_series` is a SensorLLM-inspired prompt-compatible input.
It describes channel-aware temporal patterns in text and stays inside the official prompt-compatible 4 x 3 x 2 framework.
It does not train projectors, modify LLM embeddings, or replace time-series token embeddings.
`extra_knowledge` is implemented in the official `Input/extra_knowledge.py` module.
Its optional external-knowledge parameters are configured under `input`:

```json
{
  "input": {
    "type": "extra_knowledge",
    "knowledge_mode": "append",
    "knowledge_file": "local_knowledge.txt",
    "knowledge_text": ""
  }
}
```

Parsing failures are not converted to a default label. They are saved as invalid predictions, excluded from valid-only metrics, and counted as wrong in all-samples metrics.

Evaluation metrics include valid-only Accuracy / Macro-F1 / Weighted-F1 and all-samples Accuracy / Macro-F1 / Weighted-F1 where invalid predictions are counted as wrong. Both valid-only and all-samples confusion matrices are saved.
Metrics JSON also includes `usage_summary`, `cost_estimate`, and `scaling_estimate`; prediction CSV files include per-sample LLM call count, character counts, token counts when available, elapsed runtime, and per-sample estimated cost fields.

LM prompts use prompt-scoped knowledge constraints:

```text
Use only the information provided in this prompt.
Do not use knowledge outside the provided prompt.
```

This avoids conflicts when `extra_knowledge` includes dataset/channel knowledge inside the prompt.

For debug runs, use balanced sampling:

```json
{
  "evaluation": {
    "balanced_per_label": 10,
    "log_every": 10
  }
}
```

Few-shot runs must explicitly set both `data.train_subjects` and `data.test_subjects`; overlapping subjects are rejected.
Few-shot also requires at least `n_per_class` training examples for every label in `labels`; insufficient examples raise a clear error instead of silently producing an imbalanced prompt.

Example few-shot config data block:

```json
{
  "data": {
    "train_subjects": ["S2", "S3", "S4", "S5", "S6"],
    "test_subjects": ["S7", "S8"]
  },
  "lm_usage": {
    "type": "few_shot",
    "n_per_class": 2
  }
}
```

## Experiment Matrix

| ID | Input | LM usage | Output | Config |
| -- | -- | -- | -- | -- |
| E1 | `raw_data` | `direct` | `label_only` | `configs/E1_raw_direct_label_only.json` |
| E2 | `feature_description` | `direct` | `label_only` | `configs/E2_feature_direct_label_only.json` |
| E3 | `raw_data` | `few_shot` | `label_only` | `configs/E3_raw_fewshot_label_only.json` |
| E4 | `feature_description` | `few_shot` | `label_only` | `configs/E4_feature_fewshot_label_only.json` |

Run one experiment:

```powershell
python run_experiment.py --config configs/E1_raw_direct_label_only.json
```
