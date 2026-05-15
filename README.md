# Modular WESAD LLM Experiment Pipeline

This repository contains a modular experiment framework for stress/activity classification. The experiment is organized into four visible parts:

- `Dataset/`: local packaged datasets, not uploaded to GitHub
- `Input/`: concrete input representations
- `LM/`: concrete LLM usage strategies
- `Output/`: concrete output parsers

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
```

## File Responsibilities

| File | Responsibility |
| --- | --- |
| `Dataset/wesad_loader.py` | WESAD reading, window slicing, and sampling-rate alignment |
| `Dataset/hhar_loader.py` | HHAR CSV reading, window slicing, and label mapping |
| `Dataset/dreamt_loader.py` | DREAMT CSV reading, window slicing, and label mapping |
| `core/signal_utils.py` | z-score, downsampling, slicing, packing, and feature-stat helpers |
| `core/schema.py` | `SensorSample` and `LLMSample` data structures |
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
| `core/runner.py` | Combine Dataset + Input + LM + Output |

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

## Config Runner

The older config-based runner is still available:

```powershell
python run_experiment.py --config configs/E1_raw_direct_label_only.json
```

Encoded time-series can also be selected through config files with input type `embedding_alignment` or `encoded_time_series`.
The default LM usage remains prompt-based direct/few-shot/multi-agent prediction.

For the legacy full SensorLLM checkpoint path, set `lm_usage.type` explicitly to `sensorllm_checkpoint`:

```powershell
$env:PYTHONUTF8='1'
python run_experiment.py --config configs/embedding_alignment_sensorllm_fold_s2_label_only.json
```

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
- `main.py` should stay as a module-composition entry point. Put real method logic in `Input/`, `LM/`, `Output/`, or `core/`.
