# Modular LLM Experiment Pipeline

This repository contains a modular experiment framework for stress/activity classification. The experiment is organized into five visible parts:

- `Dataset/`: local packaged datasets, not uploaded to GitHub
- `Input/`: concrete input representations
- `LM/`: concrete LLM usage strategies
- `Output/`: concrete output parsers
- `Evaluation/`: metrics and result saving

`main.py` is intentionally thin. It only parses command-line arguments and asks `core/runner.py` to select and combine the requested modules.

## Project Structure

```text
.
|-- Dataset/
|   |-- WESAD/
|   |-- HHAR/
|   |-- DREAMT/
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
|
|-- Results/
|-- configs/
|-- main.py
|-- run_experiment.py
|-- requirements.txt
```

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
| `core/signal_utils.py` | z-score, downsampling, slicing, packing, and feature-stat helpers |
| `core/schema.py` | `SensorSample`, `LLMSample`, and dataset-specific label names |
| `Input/raw_data.py` | Convert `SensorSample.signals` into raw sequence text |
| `Input/embedding_alignment.py` | Convert `SensorSample.signals` into SensorLLM-inspired encoded time-series text |
| `Input/feature_description/factory.py` | Dataset-aware Feature Description selector |
| `Input/feature_description/feature_functions.py` | Shared feature extraction and formatting helpers |
| `Input/feature_description/basic_feature_description.py` | Basic Feature Description base class |
| `Input/feature_description/wesad_feature_description.py` | WESAD-specific paper-style Feature Description class, based on `preprocess_original_paper_4.22.py` |
| `Input/feature_description/hhar_feature_description.py` | HHAR-specific Feature Description class |
| `Input/feature_description/dreamt_feature_description.py` | DreaMT-specific Feature Description class |
| `LM/direct.py` | Build direct-classification prompts |
| `Output/label_only.py` | Label-only output instruction and parser |
| `Output/label_explanation.py` | Label + explanation output instruction and parser |
| `Evaluation/metrics.py` | Accuracy, Macro-F1, Weighted-F1, confusion matrix, and result saving |
| `core/runner.py` | Shared experiment executor that combines Dataset + Input + LM + Output + Evaluation |

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

Feature-description few-shot example:

```powershell
python main.py -dataset WESAD -Input feature_description -LM few_shot -output label_only -llm qwen2.5-14b-instruct
```

Small debug run:

```powershell
python main.py -dataset WESAD -Input feature_description -LM direct -output label_only --subjects S2 --balanced-per-label 1 --log-every 1
```

Encoded time-series direct-prediction example:

```powershell
python main.py -dataset WESAD -Input encoded_time_series -LM direct -output label_only --subjects S2 --balanced-per-label 1 --log-every 1
```

Results are saved under `Results/` using this naming style:

```text
WESAD_raw_data_direct_label_only_20260512213815.csv
```

The metrics JSON includes:

- Valid-only Accuracy / Macro-F1 / Weighted-F1
- All-samples Accuracy / Macro-F1 / Weighted-F1 with invalid predictions counted as wrong
- Valid-only and all-samples confusion matrices
- Invalid prediction count and invalid rate
- `usage_summary` with LLM call count, token counts, and elapsed runtime
- `cost_estimate` using optional per-1M-token input/output prices from config
- `scaling_estimate` when `estimated_total_samples_for_full_experiment` is provided

The prediction CSV also includes per-sample usage columns:

- `llm_call_count`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `elapsed_time_sec`

## Batch / Config Runner

`run_experiment.py` is a batch/config wrapper. It reads JSON or YAML configs, expands optional grids, and calls the same shared runner as `main.py`:

```powershell
python run_experiment.py --config configs/E1_raw_direct_label_only.json
```

Future grid configs can use this shape:

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
  lm_usage: [direct, few_shot, multi_agent]
  output: [label_only, label_explanation]
```

Encoded time-series can also be selected through config files with input type `embedding_alignment` or `encoded_time_series`.
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
- feature CSV files such as `S2_features_paperstyle.csv`
- generated output folders and CSVs
- SensorLLM fold files such as `sensorllm_wesad_binary_loso/fold_S2/eval_data.pkl`
- SensorLLM checkpoints such as `sensorllm_wesad_binary_output_formal/fold_S2/`
- local base models such as Chronos/TinyLlama paths referenced in config files

## Development Notes

- Parser failures are saved as invalid predictions, not converted to a default label.
- Few-shot runs must explicitly separate train and test subjects.
- Few-shot runs require at least `n_per_class` training examples for every label in `labels`; otherwise the run stops with a clear error.
- `main.py` should stay as a module-composition entry point. Put real method logic in `Input/`, `LM/`, `Output/`, or `core/`.
