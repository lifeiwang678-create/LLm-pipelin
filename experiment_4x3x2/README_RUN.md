# Running Guide for `experiment_4x3x2`

This document describes how to run the `LLm-pipelin/experiment_4x3x2` experimental framework. It covers environment setup, data locations, single-run experiments, preprocessing caches, fixed subset experiments, few-shot experiments, 72-combination Slurm jobs, result inspection, and common error handling.

The commands below assume that you are working inside the experiment directory:

```bash
cd ~/projects/LLm-pipelin/experiment_4x3x2
```

If the repository is cloned locally, use the corresponding local path, for example:

```bash
cd /path/to/LLm-pipelin/experiment_4x3x2
```

---

## 1. Experimental Design

The current framework uses the following design:

```text
3 datasets × 4 input types × 3 LM usages × 2 output formats = 72 experiment combinations
```

| Dimension | Available options |
| --- | --- |
| Dataset | `WESAD`, `HHAR`, `DREAMT` |
| Input type | `raw_data`, `feature_description`, `encoded_time_series`, `extra_knowledge` |
| LM usage | `direct`, `few_shot`, `multi_agent` |
| Output format | `label_only`, `label_explanation` |

Binary label definitions:

| Dataset | Label 0 | Label 1 |
| --- | --- | --- |
| `WESAD` | no stress | stress |
| `HHAR` | walking downstairs | walking upstairs |
| `DREAMT` | wake | sleep |

---

## 2. Installation and Environment Setup

### 2.1 Clone the repository

```bash
git clone https://github.com/lifeiwang678-create/LLm-pipelin.git
cd LLm-pipelin/experiment_4x3x2
```

If the repository has already been cloned, update it with:

```bash
git pull
```

### 2.2 Create the Python environment

The normal experiment entry uses the project-level `.venv` environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check whether the main CLI is available:

```bash
python main.py -h
```

If the help message of `main.py` is printed, the basic Python environment is working.

### 2.3 Environments used by Slurm jobs

The 72-combination Slurm debug script may use two environments:

```text
.venv_vllm_cu121  # used to start the vLLM service
.venv             # used to run main.py and the experiment code
```

The normal single-run commands use `.venv`.

---

## 3. Data Locations

Default dataset paths are defined in:

```text
Dataset/registry.py
```

| Dataset | Default location | Notes |
| --- | --- | --- |
| WESAD | `..` | Expected layout: `../S2/S2.pkl`, `../S3/S3.pkl`, etc. |
| HHAR | `Dataset/HHAR` | Can be overridden with `--data-dir`. |
| DREAMT | `Dataset/DREAMT` | Can be overridden with `--data-dir`. |

Raw datasets, processed caches, results, local model files, and historical outputs are usually not uploaded to GitHub.

Common local directories:

```text
Dataset/      # raw or local dataset files, depending on the dataset
Processed/    # preprocessed dataset caches and fixed subset caches
Results/      # prediction CSV files, metrics JSON files, and config JSON files
```

Check whether cache and result directories already exist:

```bash
ls Processed
ls Results
```

If these directories do not exist, they will be created by preprocessing or experiment scripts when needed.

---

## 4. Recommended Running Order

Do not submit the full 72-combination job before checking the basic pipeline, data, and caches.

### 4.1 Check the main entry

```bash
source .venv/bin/activate
python main.py -h
```

### 4.2 Run one minimal debug sample

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input raw_data \
  -LM direct \
  -output label_only \
  --subjects S2 \
  --balanced-per-label 1 \
  --log-every 1
```

This command is only used to verify that the pipeline can run. It should not be used as a formal result.

### 4.3 Build dataset caches

```bash
.venv/bin/python preprocess_datasets.py -dataset WESAD --subjects S2 --overwrite
.venv/bin/python preprocess_datasets.py -dataset HHAR --data-dir "<HHAR_DATA_DIR>" --max-rows 200000 --overwrite
.venv/bin/python preprocess_datasets.py -dataset DREAMT --data-dir "<DREAMT_DATA_DIR>" --subjects S099 --overwrite
```

The dataset cache stores windowed `SensorSample` objects.

### 4.4 Build input caches

```bash
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input all --subjects S2 --overwrite
```

The input cache stores prompt-ready `LLMSample` objects.

### 4.5 Build fixed subsets

```bash
.venv/bin/python prepare_data_subsets.py
.venv/bin/python prepare_subset_inputs.py
```

Fixed subsets are used for reproducible debug, pilot, and main experiments.

### 4.6 Run one fixed-subset experiment

```bash
.venv/bin/python main.py \
  -dataset HHAR \
  -Input raw_data \
  -LM direct \
  -output label_only \
  --use-input-cache \
  --subject-split all \
  --eval-input-cache-file "Processed/LLMSubsets/HHAR/debug/HHAR_raw_data_debug_samples.pkl" \
  --log-every 1
```

After this single fixed-subset run succeeds, submit the 72-combination Slurm job.

---

## 5. Running a Single Experiment

Minimal command format:

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM direct \
  -output label_only
```

Use an external OpenAI-compatible LLM service:

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM direct \
  -output label_only \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key lm-studio \
  -llm qwen2.5-7b-instruct
```

Default output files are saved under `Results/`:

```text
Results/<run_name>_<timestamp>.csv
Results/<run_name>_<timestamp>_metrics.json
Results/<run_name>_<timestamp>_config.json
```

File meanings:

| File | Content |
| --- | --- |
| `*.csv` | Per-sample prediction results. |
| `*_metrics.json` | Accuracy, Macro-F1, Weighted-F1, token usage, runtime, invalid rate, and related statistics. |
| `*_config.json` | Configuration used for the run. |

---

## 6. Preprocessing and Cache System

The framework supports two cache layers.

| Cache type | Content | Common option |
| --- | --- | --- |
| Dataset cache | Windowed `SensorSample` objects | `--use-processed` |
| Input cache | Prompt-ready `LLMSample` objects | `--use-input-cache` |

Run with an input cache:

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM direct \
  -output label_only \
  --use-input-cache
```

Fixed subset output paths:

```text
Processed/DataSubsets/<DATASET>/<debug|pilot|main>/<DATASET>_<subset>_windows.pkl
Processed/LLMSubsets/<DATASET>/<debug|pilot|main>/<DATASET>_<INPUT>_<subset>_samples.pkl
```

Check whether debug and pilot input caches exist:

```bash
for dataset in WESAD HHAR DREAMT; do
  for level in debug pilot; do
    for input in raw_data feature_description encoded_time_series extra_knowledge; do
      path="Processed/LLMSubsets/${dataset}/${level}/${dataset}_${input}_${level}_samples.pkl"
      [[ -s "$path" ]] || echo "MISSING $path"
    done
  done
done
```

If the command prints no `MISSING ...` lines, the required debug and pilot input caches exist.

---

## 7. Fixed Subset Rules

Current fixed subset rules:

| Dataset | Rule |
| --- | --- |
| WESAD | Class-balanced sampling, for example `160:160`. |
| HHAR | 9 users, 50 samples per class per user. |
| DREAMT | 100 subjects, 5 samples per class per subject. |

The random seed is fixed as:

```text
seed = 42
```

Repeated runs of the same sampling process should produce the same subsets and few-shot examples. If the outputs differ, check whether any random process in the code is not controlled by the fixed seed.

---

## 8. Few-shot Running Rules

The recommended few-shot selection strategy is:

```text
leave_one_subject_out
```

Meaning:

- `test_subjects` or the evaluation cache determines the evaluation samples.
- For each evaluation sample, few-shot examples must not include samples from the same subject.
- Few-shot examples are sampled from other subjects.
- The recommended rule is to randomly select non-test subjects and sample a fixed number of examples per label from each selected subject.
- `--few-shot-examples-per-subject-per-label` controls the number of examples per subject per label.
- `--few-shot-n-per-class` belongs to the older `class_balanced` strategy and should not be mixed with `leave_one_subject_out` using inconsistent values.

Do not use `--subjects` to express train/test splitting in few-shot experiments. Use:

```text
--train-subjects ...
--test-subjects ...
```

or specify fixed caches directly:

```text
--train-input-cache-file ...
--eval-input-cache-file ...
```

Few-shot run using fixed caches:

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input feature_description \
  -LM few_shot \
  -output label_only \
  --use-input-cache \
  --subject-split all \
  --train-input-cache-file "Processed/LLMSubsets/WESAD/pilot/WESAD_feature_description_pilot_samples.pkl" \
  --eval-input-cache-file "Processed/LLMSubsets/WESAD/debug/WESAD_feature_description_debug_samples.pkl" \
  --few-shot-example-selection leave_one_subject_out \
  --few-shot-example-subjects 3 \
  --few-shot-examples-per-subject-per-label 1 \
  --few-shot-example-max-chars 800 \
  --log-every 1
```

If few-shot examples need to be exported or inspected, check whether the result CSV contains columns such as:

```text
few_shot_example_subjects
few_shot_example_count
few_shot_example_ids
```

If these columns are empty or contain mixed subject information under concurrent execution, check whether the runner reads metadata from the return value of `build_prompt_with_metadata` instead of shared mutable state inside the few-shot instance.

---

## 9. Running the 72-combination Debug Job with Slurm

Before submitting the job, check that the fixed subset caches exist:

```bash
for dataset in WESAD HHAR DREAMT; do
  for level in debug pilot; do
    for input in raw_data feature_description encoded_time_series extra_knowledge; do
      path="Processed/LLMSubsets/${dataset}/${level}/${dataset}_${input}_${level}_samples.pkl"
      [[ -s "$path" ]] || echo "MISSING $path"
    done
  done
done
```

Submit the 72-combination debug job:

```bash
sbatch run_3datasets_72_debug_sbatch.sh
```

Default script settings:

| Variable | Default value |
| --- | --- |
| `SUBSET_LEVEL` | `debug` |
| `FEW_SHOT_TRAIN_SUBSET_LEVEL` | `pilot` |
| `CONCURRENCY` | `8` |
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` |
| `SERVED_MODEL_NAME` | `qwen2.5-7b-instruct` |
| `PORT` | `8000` |

Override defaults when submitting the job:

```bash
sbatch --export=ALL,SUBSET_LEVEL=pilot,CONCURRENCY=4 run_3datasets_72_debug_sbatch.sh
```

Check the queue:

```bash
squeue -u <USER_ID>
```

Check whether a specific job is still running:

```bash
squeue -j <JOBID>
```

Cancel a job:

```bash
scancel <JOBID>
```

---

## 10. Logs and Result Inspection

Main Slurm log paths usually follow this pattern:

```text
~/logs/llm_72_debug_<JOBID>.out
~/logs/llm_72_debug_<JOBID>.err
```

Each run also creates a detailed log directory:

```text
~/logs/llm_72_debug_<YYYYMMDDHHMMSS>/
```

Common files:

| File | Meaning |
| --- | --- |
| `status.csv` | Status summary for the 72 combinations. |
| `vllm.log` | vLLM service log. |
| `*.log` | Individual log file for each experiment combination. |
| `gpu_usage.csv` | GPU usage records. |

Inspect the last lines of the Slurm output:

```bash
tail -n 50 ~/logs/llm_72_debug_<JOBID>.out
```

Inspect the error log:

```bash
tail -n 80 ~/logs/llm_72_debug_<JOBID>.err
```

Count failed combinations:

```bash
awk -F, 'NR>1 {total++; if ($5 != 0) failed++} END {printf "total=%d failed=%d\n", total, failed+0}' \
  ~/logs/llm_72_debug_<YYYYMMDDHHMMSS>/status.csv
```

Print failed combinations:

```bash
awk -F, 'NR==1 || $5 != 0 {print}' \
  ~/logs/llm_72_debug_<YYYYMMDDHHMMSS>/status.csv
```

Inspect GPU usage:

```bash
head ~/logs/llm_72_debug_<YYYYMMDDHHMMSS>/gpu_usage.csv
tail ~/logs/llm_72_debug_<YYYYMMDDHHMMSS>/gpu_usage.csv
```

---

## 11. Cost and Runtime Summary

After a single experiment finishes, check the corresponding metrics file:

```bash
ls Results/*_metrics.json | tail
cat Results/<run_name>_<timestamp>_metrics.json
```

To summarize multiple metrics files:

```bash
.venv/bin/python summarize_cost_profile.py --results-dir Results
```

If GPU telemetry needs to be merged, confirm the path of `gpu_usage.csv` and run the corresponding summary command.

Fields to check:

```text
sample_count
llm_call_count
prompt_tokens
completion_tokens
total_tokens
wall_time_seconds
avg_latency_seconds
invalid_rate
accuracy
macro_f1
weighted_f1
```

---

## 12. Common Issues and Fixes

### 12.1 `cache metadata mismatch`

This usually happens when dataset loader parameters have changed, such as window length, stride, or HHAR `max_rows`. Old caches are rejected because they do not match the current configuration.

Delete and rebuild the related caches:

```bash
rm -f Processed/<DATASET>_*_samples.pkl
.venv/bin/python preprocess_inputs.py -dataset <DATASET> -Input all --overwrite
```

If fixed subsets are used, rebuild them as well:

```bash
.venv/bin/python prepare_data_subsets.py --overwrite
.venv/bin/python prepare_subset_inputs.py --overwrite
```

### 12.2 `balanced_per_label` errors

`--balanced-per-label N` requires exactly `N` samples for each label after filtering.

Possible fixes:

- If one label has too few samples, reduce `N` or include more subjects/cache files.
- If there are enough samples but the error remains, check additional limits, label filtering, or subject filtering logic.

### 12.3 Few-shot subject errors

Do not use `--subjects` to represent the train/test split in few-shot experiments.

Use:

```text
--train-subjects ...
--test-subjects ...
```

or fixed caches:

```text
--train-input-cache-file ...
--eval-input-cache-file ...
```

### 12.4 Few-shot metadata mixing under concurrency

When running concurrently, few-shot subject metadata may be overwritten if it is stored in shared mutable instance variables.

Fix direction:

```text
_run_one_sample should call build_prompt_with_metadata
Few-shot metadata should be read directly from the returned metadata object
Do not read last_example_subjects / last_example_count from a shared FewShotUsage instance
```

This should be checked especially when `--concurrency > 1` is used.

### 12.5 vLLM port already in use

The 72-combination script checks:

```text
http://127.0.0.1:8000/v1/models
```

If another service already uses the port, stop the old service or choose another port:

```bash
sbatch --export=ALL,PORT=8001 run_3datasets_72_debug_sbatch.sh
```

### 12.6 `pytest` is not available

Some cluster environments may not have `pytest` installed. Run a syntax check first:

```bash
.venv/bin/python -m py_compile core/runner.py LM/__init__.py LM/few_shot.py LM/multi_agent.py
```

To run tests, install the test dependency:

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest test_few_shot_sampling.py test_runner_regressions.py
```

### 12.7 LLM output is not valid JSON

If `invalid_rate` is high, check:

- Whether the prompt explicitly requires strict JSON.
- Whether `label_only` only allows `{"predicted_state": <label>}`.
- Whether the model outputs explanations, Markdown, or extra text.
- Whether `max_tokens` is too small.
- Whether temperature is too high.

### 12.8 Results are strongly biased toward one class

If the model predicts almost all samples as one label, check:

- Whether the ground-truth label distribution is balanced.
- Whether the prompt implies one class is more likely.
- Whether `raw_data` is too long and causes the model to focus on local abnormal patterns.
- Whether `feature_description` contains misleading descriptions.
- Whether few-shot examples are class-balanced.
- Whether train/evaluation subjects are mixed or leaked.

---

## 13. File Reference

### 13.1 Main entry files

| File | Role | Daily use |
| --- | --- | --- |
| `main.py` | Main CLI entry. Calls `core/runner.py`. | Common |
| `run_experiment.py` | Expands experiments from JSON/YAML config or grid, then calls the same runner. | Occasional |
| `requirements.txt` | Dependency list for the normal `.venv` environment. | Environment setup |

### 13.2 Data checking and cache construction

| File | Role | Daily use |
| --- | --- | --- |
| `count_dataset_samples.py` | Counts dataset/subject/label windows without calling the LLM. | Data checking |
| `preprocess_datasets.py` | Builds the first-level dataset cache, i.e., windowed `SensorSample` objects. | Cache building |
| `preprocess_inputs.py` | Builds the second-level input cache, i.e., prompt-ready `LLMSample` objects. | Cache building |
| `prepare_data_subsets.py` | Samples fixed `debug`, `pilot`, and `main` subsets from preprocessed windows. | Before reproducible 72-combination runs |
| `prepare_subset_inputs.py` | Builds LLM caches for four input types using fixed subsets. | Before reproducible 72-combination runs |

### 13.3 Slurm and batch scripts

| File | Role | Status |
| --- | --- | --- |
| `run_3datasets_72_debug_sbatch.sh` | Recommended Slurm entry for 3-dataset 72-combination debug/pilot/main runs. | Recommended |
| `run_wesad_24_full_sbatch.sh` | Runs 24 combinations for WESAD only, usually on a larger subset or full setting. | Special-purpose |
| `run_hhar_dreamt_preprocess_sbatch.sh` | Preprocesses HHAR and DREAMT caches on Slurm. | Auxiliary |
| `run_all_3datasets_4x3x2.ps1` | PowerShell version of the 72-combination batch run, mainly for local/Windows workflows. | Backup |
| `run_wesad_4x3x2_small.ps1` | PowerShell version of a small WESAD 24-combination run. | Backup |
| `run_wesad_remaining23.sh` | Earlier script for rerunning remaining WESAD combinations. | Historical |
| `run_wesad_remaining5_sbatch.sh` | Earlier script for rerunning five remaining WESAD combinations. | Historical |

### 13.4 Model service, benchmark, and smoke tests

| File | Role | Daily use |
| --- | --- | --- |
| `benchmark_vllm_batch.py` | Benchmarks OpenAI-compatible vLLM batching and concurrency. | Tuning |
| `run_vllm_batch_benchmark_sbatch.sh` | Slurm version of the vLLM batching benchmark. | Tuning |
| `setup_qwen3_vllm_env.sh` | Creates an independent vLLM environment for Qwen3. | When switching to Qwen3 |
| `serve_qwen3_transformers_openai.py` | Starts a lightweight OpenAI-compatible Qwen3 service using transformers. | Backup when vLLM is unsuitable |
| `smoke_qwen3_json.py` | Tests whether Qwen3 can produce stable parseable JSON. | When switching models |
| `run_qwen3_vllm_smoke_sbatch.sh` | Slurm Qwen3 vLLM smoke test. | When switching models |
| `run_qwen3_transformers_smoke_sbatch.sh` | Slurm Qwen3 transformers smoke test. | Backup when vLLM fails |
| `summarize_cost_profile.py` | Summarizes token usage, runtime, and cost estimates from metrics JSON files. | Result analysis |

### 13.5 Test files

| File | Role |
| --- | --- |
| `test_embedding_alignment_input.py` | Checks `encoded_time_series` and the older `embedding_alignment` input format. |
| `test_feature_description_factory.py` | Checks the feature description factory, dataset injection, and subject split logic. |
| `test_few_shot_sampling.py` | Checks few-shot leave-one-subject-out sampling. |
| `test_runner_regressions.py` | Checks recent runner fixes, including concurrency, cache metadata, balanced sampling, and few-shot subject semantics. |

---

## 14. Daily-use Files

For normal experiment runs, focus on:

```text
main.py
run_3datasets_72_debug_sbatch.sh
preprocess_datasets.py
preprocess_inputs.py
prepare_data_subsets.py
prepare_subset_inputs.py
```

Historical rerun scripts such as `run_wesad_remaining*` are kept for reproducing earlier runs and are not recommended as new experiment entry points.

Qwen3 and benchmark scripts are model-service debugging tools. They do not affect the current Qwen2.5 72-combination main workflow.

---

## 15. Minimal Command Checklist

Start from the experiment directory:

```bash
cd ~/projects/LLm-pipelin/experiment_4x3x2
source .venv/bin/activate
python main.py -h
```

Run one small sample:

```bash
.venv/bin/python main.py \
  -dataset WESAD \
  -Input raw_data \
  -LM direct \
  -output label_only \
  --subjects S2 \
  --balanced-per-label 1 \
  --log-every 1
```

Build fixed subsets:

```bash
.venv/bin/python prepare_data_subsets.py
.venv/bin/python prepare_subset_inputs.py
```

Check caches:

```bash
for dataset in WESAD HHAR DREAMT; do
  for level in debug pilot; do
    for input in raw_data feature_description encoded_time_series extra_knowledge; do
      path="Processed/LLMSubsets/${dataset}/${level}/${dataset}_${input}_${level}_samples.pkl"
      [[ -s "$path" ]] || echo "MISSING $path"
    done
  done
done
```

Submit the 72-combination debug job:

```bash
sbatch run_3datasets_72_debug_sbatch.sh
```

Check the queue:

```bash
squeue -u <USER_ID>
```

Check the 72-combination status file:

```bash
awk -F, 'NR>1 {total++; if ($5 != 0) failed++} END {printf "total=%d failed=%d\n", total, failed+0}' \
  ~/logs/llm_72_debug_<YYYYMMDDHHMMSS>/status.csv
```
