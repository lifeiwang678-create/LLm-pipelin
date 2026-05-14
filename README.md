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
|
|-- Input/
|   |-- raw_data.py
|   |-- feature_description.py
|   |-- embedding_alignment.py
|   |-- extra_knowledge.py
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

## Supported Modules

Input:

- `raw_data`
- `feature_description`
- `embedding_alignment`
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

Results are saved under `Results/` using this naming style:

```text
WESAD_raw_data_direct_label_only_20260512213815.csv
```

## Config Runner

The older config-based runner is still available:

```powershell
python run_experiment.py --config configs/E1_raw_direct_label_only.json
```

For SensorLLM embedding/alignment inference on Windows paths with non-ASCII characters:

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

