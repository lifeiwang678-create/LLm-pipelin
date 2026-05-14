# Modular WESAD LLM Experiment Pipeline

This repository contains a modular experiment framework for stress/activity classification with three replaceable layers:

- Input: `raw_data`, `feature_description`, `embedding_alignment`, `extra_knowledge`
- LM usage: `direct`, `few_shot`, `multi_agent`, plus SensorLLM checkpoint routing for `embedding_alignment`
- Output: `label_only`, `label_explanation`

The framework code is lightweight. Dataset packages, generated feature CSVs, SensorLLM fold data, and model checkpoints are intentionally not included.

## Structure

```text
.
├── Dataset/              # Local packaged datasets, not uploaded
│   ├── WESAD/
│   ├── HHAR/
│   └── DREAMT/
├── Input/
│   ├── raw_data.py
│   ├── feature_description.py
│   ├── embedding_alignment.py
│   └── extra_knowledge.py
├── LM/
│   ├── direct.py
│   ├── few_shot.py
│   └── multi_agent.py
├── Output/
│   ├── label.py
│   └── label_explanation.py
├── Results/              # Experiment CSV outputs, not uploaded
├── main.py               # CLI entry point
└── experiment_pipeline/  # Shared implementation backend
```

## Run

Run with explicit dataset, input, LM usage, and output:

```powershell
python main.py -dataset WESAD -Input raw_data -LM direct -output label
```

Choose the LM Studio model with `-llm`:

```powershell
python main.py -dataset WESAD -Input feature_description -LM few_shot -output label -llm qwen2.5-14b-instruct
```

Results are saved under `Results/` with this naming style:

```text
WESAD_raw_data_direct_label_20260512213815.csv
```

The older config-based entry point is still available:

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
