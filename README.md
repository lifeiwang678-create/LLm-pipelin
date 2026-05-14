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
│   ├── raw_data.py              # Raw WESAD pkl loading, windowing, normalization
│   ├── feature_description.py   # Feature CSV loading and text formatting
│   ├── embedding_alignment.py   # SensorLLM fold QA/data alignment
│   └── extra_knowledge.py       # Feature input with appended knowledge text
├── LM/
│   ├── direct.py                # Direct prompt construction
│   ├── few_shot.py              # Few-shot example sampling and prompt construction
│   └── multi_agent.py           # Multi-agent prompt construction
├── Output/
│   ├── label_only.py            # Strict label-only JSON parser
│   └── label_explanation.py     # Strict label + explanation JSON parser
├── Results/              # Experiment CSV outputs, not uploaded
├── main.py               # Thin module-composition entry point
└── core/                 # CLI, runner, shared schema, LM client, evaluation helpers
```

`main.py` only parses arguments and delegates module selection/execution to `core/runner.py`.

## Run

Run with explicit dataset, input, LM usage, and output:

```powershell
python main.py -dataset WESAD -Input raw_data -LM direct -output label_only
```

Choose the LM Studio model with `-llm`:

```powershell
python main.py -dataset WESAD -Input feature_description -LM few_shot -output label_only -llm qwen2.5-14b-instruct
```

Results are saved under `Results/` with this naming style:

```text
WESAD_raw_data_direct_label_only_20260512213815.csv
```

The older config-based entry point is still available:

```powershell
python run_experiment.py --config configs/example_experiment.json
```

For SensorLLM embedding/alignment inference on Windows paths with non-ASCII characters:

```powershell
$env:PYTHONUTF8='1'
python run_experiment.py --config configs/embedding_alignment_sensorllm_fold_s2_label_only.json
```

## Local Files Not Tracked

Prepare these locally before running the corresponding experiments:

- WESAD subject folders such as `S2/`, `S3/`, ...
- feature CSV files such as `S2_features_paperstyle.csv`
- SensorLLM fold files such as `sensorllm_wesad_binary_loso/fold_S2/eval_data.pkl`
- SensorLLM checkpoints such as `sensorllm_wesad_binary_output_formal/fold_S2/`
- local base models such as Chronos/TinyLlama paths referenced in config files
