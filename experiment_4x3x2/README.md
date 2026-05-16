# Modular LLM Experiment Pipeline

This folder contains the current official modular experiment framework for stress/activity classification. The experiment is organized into five visible parts:

- `Dataset/`: dataset loaders and optional local dataset folders
- `Input/`: concrete input representations
- `LM/`: concrete LLM usage strategies
- `Output/`: concrete output parsers
- `Evaluation/`: metrics and result saving

`main.py` is intentionally thin. It only parses command-line arguments and asks `core/runner.py` to select and combine the requested modules. `core/runner.py` is the single experiment execution path used by both `main.py` and `run_experiment.py`.

The repository root is only an outer container. Run official experiments from inside `experiment_4x3x2/`.
The old `experiment_pipeline/` path is no longer part of the official framework. Reference-only scripts belong under `../legacy/`.

## Project Structure

```text
.
|-- Dataset/
|   |-- WESAD/        optional local data folder
|   |-- HHAR/         default HHAR local data folder
|   |-- DREAMT/       default DREAMT local data folder
|   |-- registry.py
|   |-- wesad_loader.py
|   |-- hhar_loader.py
|   |-- dreamt_loader.py
|
|-- Input/
|   |-- raw_data.py
|   |-- embedding_alignment.py
|   |-- extra_knowledge.py
|   |-- feature_description/
|       |-- __init__.py
|       |-- factory.py
|       |-- feature_functions.py
|       |-- basic_feature_description.py
|       |-- wesad_feature_description.py
|       |-- hhar_feature_description.py
|       |-- dreamt_feature_description.py
|
|-- LM/
|   |-- direct.py
|   |-- few_shot.py
|   |-- multi_agent.py
|
|-- Output/
|   |-- label_only.py
|   |-- label_explanation.py
|
|-- Evaluation/
|   |-- metrics.py
|
|-- core/
|   |-- cli.py
|   |-- runner.py
|   |-- signal_utils.py
|   |-- splits.py
|   |-- evaluation.py
|   |-- lm_client.py
|   |-- schema.py
|   |-- inputs.py      compatibility forwarding module
|   |-- lm_usage.py    compatibility forwarding module
|   |-- outputs.py     compatibility forwarding module
|
|-- Results/
|-- configs/
|-- main.py
|-- run_experiment.py
|-- requirements.txt
```

Current default dataset locations are defined in `Dataset/registry.py`. WESAD defaults to the outer repository's subject folders such as `../S2/`, `../S3/`, ... (`data_dir: ".."`). HHAR defaults to `Dataset/HHAR/`, and DREAMT defaults to `Dataset/DREAMT/`. You can override `dataset.data_dir` in a config file.

## Installation

Install the Python dependencies first:

```powershell
pip install -r requirements.txt
```

Core dependencies include `numpy`, `pandas`, `scikit-learn`, `requests`, `PyYAML`, `scipy`, and `neurokit2`.

## File Responsibilities

| File | Responsibility |
| --- | --- |
| `Dataset/wesad_loader.py` | WESAD reading, window slicing, and sampling-rate alignment |
| `Dataset/hhar_loader.py` | HHAR CSV reading, window slicing, and label mapping |
| `Dataset/dreamt_loader.py` | DREAMT CSV reading, window slicing, and label mapping |
| `Dataset/registry.py` | Dataset defaults such as data directory, default subjects, and loader kwargs |
| `core/signal_utils.py` | z-score, downsampling, slicing, packing, and feature-stat helpers |
| `core/schema.py` | `SensorSample`, `LLMSample`, and dataset-specific label names |
| `Input/raw_data.py` | Convert WESAD `SensorSample.signals` into raw sequence text; HHAR/DREAMT raw formatting is intentionally not enabled yet |
| `Input/embedding_alignment.py` | Convert `SensorSample.signals` into SensorLLM-inspired textual encoded time-series; this is not true embedding-level alignment |
| `Input/feature_description/factory.py` | Dataset-aware Feature Description selector |
| `Input/feature_description/feature_functions.py` | Shared feature extraction and formatting helpers |
| `Input/feature_description/basic_feature_description.py` | Basic Feature Description base class |
| `Input/feature_description/wesad_feature_description.py` | WESAD-specific paper-style Feature Description class |
| `Input/feature_description/hhar_feature_description.py` | HHAR-specific Feature Description class |
| `Input/feature_description/dreamt_feature_description.py` | DreaMT-specific Feature Description class |
| `LM/direct.py` | Build direct-classification prompts |
| `LM/few_shot.py` | Build Brown-style prompt-level few-shot / in-context classification prompts |
| `LM/multi_agent.py` | Run three-call agent-based reasoning: evidence extraction, candidate evaluation, final decision |
| `Output/label_only.py` | Label-only output instruction and parser |
| `Output/label_explanation.py` | Label + explanation output instruction and parser |
| `Evaluation/metrics.py` | Accuracy, Macro-F1, Weighted-F1, confusion matrices, usage summary, cost estimate, and result saving |
| `core/lm_client.py` | OpenAI-compatible LM Studio client with per-call usage/runtime logging |
| `core/runner.py` | Shared experiment executor that combines Dataset + Input + LM + Output + Evaluation |
| `core/inputs.py`, `core/lm_usage.py`, `core/outputs.py` | Compatibility forwarding modules; they keep no independent experiment logic |

## Dataset-Specific Labels

Label names are dataset-specific and are used in both prompts and evaluation reports.

| Dataset | Label 1 | Label 2 | Label 3 |
| --- | --- | --- | --- |
| `WESAD` | Baseline | Stress | Amusement |
| `HHAR` | Static activity | Dynamic activity | Stairs activity |
| `DREAMT` | Baseline/Neutral/Relax | Stress | Amusement/Happy |

The numeric label IDs remain the output target. The names are descriptive text for prompts, reports, and confusion-matrix readability.

## Supported Modules

Input:

- `raw_data`
- `feature_description`
- `embedding_alignment` / `encoded_time_series`
- `extra_knowledge`

LM usage:

- `direct`
- `few_shot`
- `multi_agent`

Output:

- `label_only`
- `label_explanation`

Use these exact output names. Do not use `label` as an alias.

## Run

Run with explicit dataset, input, LM usage, and output:

```powershell
python main.py -dataset WESAD -Input raw_data -LM direct -output label_only
```

By default, `main.py` evaluates all loaded samples for the selected subjects. Use `--balanced-per-label` only for debug subsets.
For WESAD, the configured default subjects are currently `S2` and `S3`. To evaluate more WESAD subjects, pass them explicitly with `--subjects`.

`raw_data` is currently WESAD-specific. Use `feature_description`, `embedding_alignment`, or `extra_knowledge` for HHAR/DREAMT until dataset-aware raw formatting is added.

Feature-description few-shot example:

```powershell
python main.py -dataset WESAD -Input feature_description -LM few_shot -output label_only --train-subjects S2 S3 S4 S5 S6 --test-subjects S7 S8 -llm qwen2.5-14b-instruct
```

Small debug run:

```powershell
python main.py -dataset WESAD -Input feature_description -LM direct -output label_only --subjects S2 --balanced-per-label 1 --log-every 1
```

Encoded time-series direct-prediction example:

```powershell
python main.py -dataset WESAD -Input encoded_time_series -LM direct -output label_only --subjects S2 --balanced-per-label 1 --log-every 1
```

Extra-knowledge input can use built-in dataset knowledge only, append external knowledge, or replace built-in knowledge with external knowledge:

```powershell
python main.py -dataset WESAD -Input extra_knowledge -LM direct -output label_only --subjects S2 --balanced-per-label 1 --knowledge-mode append --knowledge-file local_knowledge.txt
```

By default, `main.py` saves results under `Results/` using this naming style:

```text
WESAD_raw_data_direct_label_only_20260512213815.csv
```

Config files can override this with `output_dir`.

The metrics JSON includes:

- Valid-only Accuracy / Macro-F1 / Weighted-F1
- All-samples Accuracy / Macro-F1 / Weighted-F1 with invalid predictions counted as wrong
- Valid-only and all-samples confusion matrices
- Invalid prediction count and invalid rate
- `usage_summary` with LLM call count, character counts, token counts when available, and elapsed runtime
- `cost_estimate` using optional per-1M-token input/output prices from config
- `scaling_estimate` when `estimated_total_samples_for_full_experiment` is provided

The prediction CSV also includes per-sample usage columns:

- `llm_call_count`
- `prompt_chars`
- `completion_chars`
- `total_chars`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `elapsed_time_sec`
- `estimated_input_cost`
- `estimated_output_cost`
- `estimated_total_cost`

## Batch / Config Runner

`run_experiment.py` is a batch/config wrapper. It reads JSON or YAML configs, expands optional grids, and calls the same shared runner as `main.py`:

```powershell
python run_experiment.py --config configs/E1_raw_direct_label_only.json
```

Future grid configs can use this shape for direct and multi-agent runs:

```yaml
base:
  dataset: WESAD
  labels: [1, 2]
  data:
    subjects: [S2]
  evaluation:
    balanced_per_label: 1
    log_every: 1
grid:
  input: [raw_data, feature_description]
  lm_usage: [direct, multi_agent]
  output: [label_only, label_explanation]
```

Few-shot configs must use explicit non-overlapping train/test subjects:

```yaml
base:
  dataset: WESAD
  labels: [1, 2]
  data:
    train_subjects: [S2, S3, S4, S5, S6]
    test_subjects: [S7, S8]
  lm_usage:
    n_per_class: 2
    random_state: 42
  evaluation:
    balanced_per_label: 1
    log_every: 1
grid:
  input: [raw_data, feature_description]
  lm_usage: [few_shot]
  output: [label_only, label_explanation]
```

Extra-knowledge config options live under `input`:

```yaml
input:
  type: extra_knowledge
  knowledge_mode: append
  knowledge_file: local_knowledge.txt
  knowledge_text: ""
```

Encoded time-series can also be selected through config files with input type `embedding_alignment` or `encoded_time_series`.
This input is a prompt-compatible textual adaptation inspired by SensorLLM. It does not train projectors, modify LLM embeddings, or replace time-series token embeddings.
The default LM usage remains prompt-based direct/few-shot/multi-agent prediction.

The official 4 x 3 x 2 experiment path supports the registered `LM/` methods: `direct`, `few_shot`, and `multi_agent`.

All LM prompts use a prompt-scoped knowledge rule:

```text
Use only the information provided in this prompt.
Do not use knowledge outside the provided prompt.
```

This keeps `raw_data`, `feature_description`, `encoded_time_series`, and `extra_knowledge` compatible.

## Local Files Not Tracked

The following files are intentionally not uploaded to GitHub:

- WESAD subject folders such as `S2/`, `S3/`, ...
- optional local dataset folders such as `Dataset/WESAD/`, `Dataset/HHAR/`, and `Dataset/DREAMT/`
- feature CSV files such as `S2_features_paperstyle.csv`
- generated output folders and CSVs
- local base models such as Chronos/TinyLlama paths referenced in config files

## Development Notes

- Parser failures are saved as invalid predictions, not converted to a default label.
- Few-shot runs must explicitly separate train and test subjects.
- Few-shot runs require at least `n_per_class` training examples for every label in `labels`; otherwise the run stops with a clear error.
- `experiment_pipeline/` is a removed legacy path. The maintained entry is `main.py` / `run_experiment.py` -> `core/runner.py`.
- `main.py` should stay as a module-composition entry point. Put real method logic in `Input/`, `LM/`, `Output/`, or `core/`.
