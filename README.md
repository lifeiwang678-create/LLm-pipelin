# Modular WESAD LLM Experiment Pipeline

This repository contains a modular experiment framework for WESAD stress classification with three replaceable layers:

- Input: `raw_data`, `feature_description`, `embedding_alignment`, `extra_knowledge`
- LM usage: `direct`, `few_shot`, `multi_agent`, plus SensorLLM checkpoint routing for `embedding_alignment`
- Output: `label_only`, `label_explanation`

The framework code is lightweight. WESAD data files, generated feature CSVs, SensorLLM fold data, and model checkpoints are intentionally not included.

## Run

Edit a config file under `configs/`, then run:

```powershell
python run_experiment.py --config configs/example_experiment.json
```

For SensorLLM embedding/alignment inference on Windows paths with non-ASCII characters:

```powershell
$env:PYTHONUTF8='1'
python run_experiment.py --config configs/embedding_alignment_sensorllm_fold_s2_label.json
```

## Local Files Not Tracked

Prepare these locally before running the corresponding experiments:

- WESAD subject folders such as `S2/`, `S3/`, ...
- feature CSV files such as `S2_features_paperstyle.csv`
- SensorLLM fold files such as `sensorllm_wesad_binary_loso/fold_S2/eval_data.pkl`
- SensorLLM checkpoints such as `sensorllm_wesad_binary_output_formal/fold_S2/`
- local base models such as Chronos/TinyLlama paths referenced in config files

