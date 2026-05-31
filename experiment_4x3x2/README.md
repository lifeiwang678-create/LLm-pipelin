# Modular 4 x 3 x 2 LLM Experiment Framework

This folder is the current official experiment framework. Run experiments from inside this directory:

```powershell
cd experiment_4x3x2
.venv/bin/python main.py -h
```

The framework keeps one shared execution path:

```text
main.py -> core/runner.py -> Dataset / Input / LM / Output / Evaluation
```

`main.py` only parses command-line arguments. `core/runner.py` selects modules, executes the experiment, calls the LLM, parses outputs, and saves results. `run_experiment.py` is only a config/batch wrapper around the same runner.

## 4 x 3 x 2 Design

Datasets:

- `WESAD`
- `HHAR`
- `DREAMT`

Input representations:

- `raw_data`
- `feature_description`
- `encoded_time_series`
- `extra_knowledge`

`embedding_alignment` is accepted as a backward-compatible alias for `encoded_time_series`.
The implementation is a SensorLLM-inspired textual representation, not true embedding-level alignment.

LM usage:

- `direct`
- `few_shot`
- `multi_agent`

Output formats:

- `label_only`
- `label_explanation`

`label` is accepted as a compatibility alias for `label_only`.

The full study has:

```text
3 datasets x 4 inputs x 3 LM usages x 2 outputs = 72 runs
```

## Binary Tasks

All three datasets are currently configured as binary classification tasks.

| Dataset | Label 0 | Label 1 |
| --- | --- | --- |
| `WESAD` | no stress | stress |
| `HHAR` | walking downstairs | walking upstairs |
| `DREAMT` | wake | sleep |

Dataset-specific label names are defined in `core/schema.py` and are used in prompts, reports, and confusion matrices.

## Subject-Independent Protocol

The default experiment policy is subject-independent.

- `train_subjects` and `test_subjects` must not overlap.
- `few_shot` with `class_balanced` uses `train_subjects` for in-context examples.
- `few_shot` with `leave_one_subject_out` evaluates `test_subjects` and samples examples from non-evaluation subjects; use `examples_per_subject_per_label`, not `n_per_class`, to control per-subject examples.
- Do not use `subjects` as a shorthand for few-shot runs; use `train_subjects` and `test_subjects` so example-source and evaluation subjects are explicit.
- `direct` and `multi_agent` do not use training examples, but they still evaluate only on held-out `test_subjects` by default so all LM usages are compared on the same subjects.
- Dataset defaults live in `Dataset/registry.py`. Override them with `--train-subjects` and `--test-subjects` when needed.
- Use `--subject-split all` only for debugging or legacy runs where evaluating every available subject is intentional.

### WESAD Mapping

WESAD original labels are mapped as:

| Original state | Binary label |
| --- | --- |
| Baseline | `0 = no stress` |
| Amusement | `0 = no stress` |
| Meditation | `0 = no stress` |
| Recovery-like local labels | `0 = no stress` |
| Stress | `1 = stress` |

The loader ignores undefined/transient WESAD label `0`.

### HHAR Mapping

HHAR is a fine-grained stair activity task:

| Original activity | Binary label |
| --- | --- |
| downstairs / stairs_down / down / walking_downstairs | `0 = walking downstairs` |
| upstairs / stairs_up / up / walking_upstairs | `1 = walking upstairs` |

Other HHAR activities, such as sitting, standing, walking, biking, and null labels, are filtered out.

### DREAMT Mapping

DREAMT sleep-stage labels are mapped as:

| Original state | Binary label |
| --- | --- |
| wake / awake / w | `0 = wake` |
| sleep / REM / NREM / N1 / N2 / N3 | `1 = sleep` |

## Project Structure

```text
.
|-- Dataset/
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
|   |-- lm_client.py
|   |-- schema.py
|   |-- signal_utils.py
|   |-- splits.py
|
|-- Results/
|-- Processed/
|-- configs/
|-- main.py
|-- run_experiment.py
|-- preprocess_datasets.py
|-- preprocess_inputs.py
|-- prepare_data_subsets.py
|-- prepare_subset_inputs.py
|-- count_dataset_samples.py
|-- requirements.txt
```

## Data Locations

Default data locations are in `Dataset/registry.py`.

- WESAD defaults to `..`, so it expects subject folders such as `../S2/S2.pkl`.
- HHAR defaults to `Dataset/HHAR`, but local data can be supplied with `--data-dir`.
- DREAMT defaults to `Dataset/DREAMT`, but local data can be supplied with `--data-dir`.

Examples for local data paths used in development:

```powershell
--data-dir "<HHAR_DATA_DIR>"
```

for HHAR, where the folder contains:

```text
Phones_accelerometer.csv
Phones_gyroscope.csv
```

```powershell
--data-dir "<DREAMT_DATA_DIR>"
```

for DREAMT, where the folder contains files such as:

```text
data_64Hz/S099_whole_df.csv
```

Large local datasets are intentionally not tracked in Git.

## vLLM Shard Settings

On the current `a100` shard environment, use the OpenAI-compatible vLLM server with the maintained model name:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b-instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 8192 \
  --max-num-seqs 96 \
  --max-num-batched-tokens 24576 \
  --disable-log-requests
```

The runner supports concurrent client requests with `--concurrency`. The shard benchmark was stable up to server concurrency 96; for full experiments, `--concurrency 64` is the default in the Slurm WESAD scripts to leave some headroom.

## Qwen3 Smoke Test

Qwen3 can generate thinking blocks by default. For this framework, disable
thinking before formal experiments because the output parser expects strict
JSON only.

Create a separate Qwen3 vLLM environment:

```bash
./setup_qwen3_vllm_env.sh
```

Run the Qwen3 JSON smoke test:

```bash
sbatch run_qwen3_vllm_smoke_sbatch.sh
```

For vLLM builds that expose server-level template defaults, the smoke job
starts Qwen3 with:

```bash
--default-chat-template-kwargs '{"enable_thinking": false}'
```

On the current A100 shard nodes, NVIDIA driver 535/CUDA 12.2 cannot run the
latest PyPI vLLM stacks that require CUDA 12.4+ or CUDA 13. If vLLM fails for
Qwen3, use the Transformers fallback smoke first:

```bash
sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3-0.6B,SERVED_MODEL_NAME=qwen3-0.6b run_qwen3_transformers_smoke_sbatch.sh
```

Both smoke paths check direct JSON responses and a small real-pipeline subset
across WESAD, HHAR, and DREAMT. Only run formal Qwen3 experiments after
`qwen3_smoke_report.json` shows `"ok": true` and `invalid_count` is 0.

## Dataset Loaders

`Dataset/wesad_loader.py`

- Loads WESAD `.pkl` files.
- Uses LLM-friendly WESAD windows:
  - physiology window: 60 seconds
  - ACC window: 5 seconds
  - stride: 60 seconds
  - dense 0.25-second stride should only be used for traditional ML reproduction, not routine LLM API runs
- Maps original states to binary no-stress/stress labels.

`Dataset/hhar_loader.py`

- Loads `Phones_accelerometer.csv` and `Phones_gyroscope.csv` when available.
- Filters to upstairs/downstairs stair activities only.
- Downsamples phone IMU streams to 10 Hz tokens for LLM prompts.
- Uses 2-second windows with 1-second stride by default:
  - `window_size = 20`
  - `stride_size = 10`
  - `sampling_rate = 10`
  - `include_gyroscope = true`
- Assigns each window label by majority activity label.
- Use explicit user-level `--train-subjects` and `--test-subjects` for few-shot runs to avoid user leakage.
- Supports `--max-rows` for debug reads of large CSV files.

`Dataset/dreamt_loader.py`

- Loads raw DREAMT files such as `data_64Hz/S099_whole_df.csv`; DREAMT defaults to 64 Hz for the wearable-only experiment setting.
- Uses 30-second epochs by default.
- Maps sleep stages to binary wake/sleep labels.

## Input Modules

`raw_data`

- Converts sensor channels into compact raw sequence text.
- Uses WESAD-specific formatting for WESAD.
- Uses generic channel packing for HHAR and DREAMT.

`feature_description`

- Uses dataset-aware feature description classes:
  - WESAD paper-style physiological features
  - HHAR motion features
  - DREAMT sleep/wearable features

`encoded_time_series`

- SensorLLM-inspired textual encoded time-series input.
- Channel-aware and segment-level temporal descriptions.
- Prompt-compatible only.
- It does not train projectors, modify LLM embeddings, run Chronos, or replace time-series token embeddings.
- `embedding_alignment` is still accepted as a backward-compatible alias.

`extra_knowledge`

- Builds on feature description.
- Adds dataset context, channel knowledge, decision guidance, and optional external knowledge.
- Does not define labels itself.

## LM Usage Modules

`direct`

- One prompt, one LLM call.

`few_shot`

- Prompt-level in-context learning.
- Uses explicit non-overlapping train/test subjects when provided, otherwise the dataset's subject-independent defaults.
- The caller provides training examples; the module does not access test samples.

`multi_agent`

- Three LLM calls per sample:
  - evidence extraction
  - candidate evaluation
  - final decision
- Only the final decision is parsed by the Output module.
- This is much slower than `direct` and `few_shot`.

## Output Modules

`label_only`

Expected JSON:

```json
{
  "predicted_state": 0
}
```

`label_explanation`

Expected JSON:

```json
{
  "predicted_state": 0,
  "explanation": "one short sentence"
}
```

Parser failures are saved as invalid predictions. They are not converted to a fallback label.

## Running Experiments

Small WESAD debug run:

```powershell
.venv/bin/python main.py -dataset WESAD -Input raw_data -LM direct -output label_only --subjects S2 --balanced-per-label 1 --log-every 1
```

Small HHAR debug run:

```powershell
.venv/bin/python main.py -dataset HHAR -Input raw_data -LM direct -output label_only --data-dir "<HHAR_DATA_DIR>" --max-rows 200000 --balanced-per-label 1 --log-every 1
```

Small DREAMT debug run:

```powershell
.venv/bin/python main.py -dataset DREAMT -Input feature_description -LM direct -output label_only --data-dir "<DREAMT_DATA_DIR>" --subjects S099 --balanced-per-label 1 --log-every 1
```

Use `--balanced-per-label` only for debug subsets. Formal full-data runs should omit it.

Use `--max-rows` only for debugging large CSV datasets such as HHAR. Formal HHAR runs should omit it.

## Processed Dataset Cache

Use `preprocess_datasets.py` to cut windows once and save reusable binary
`SensorSample` caches. This avoids re-reading raw files and re-cutting windows
for every Input x LM x Output combination.

Keep full processed dataset caches for traditional ML baselines and dataset
statistics. Full LLM inference can be expensive, so the recommended LLM
protocol uses subject-independent / unseen-user subset caches generated from
the full input caches.

Processed files are stored in `Processed/`:

```text
Processed/WESAD_binary_windows.pkl
Processed/HHAR_binary_windows.pkl
Processed/DREAMT_binary_windows_manifest.json
Processed/DREAMT_binary_windows_S002.pkl
Processed/DREAMT_binary_windows_S003.pkl
...
```

Single-file `.pkl` caches contain one dataset-level list of window samples.
Subject-shard manifests point to one `.pkl` file per subject. Each sample keeps
its own `label`, `signals`, and metadata. Labels are for evaluation only; Input
modules must still avoid writing true labels into prompts.

For WESAD LLM experiments, avoid saving full raw-window dataset caches. Even
subject shards can become very large because each `SensorSample` contains full
signal arrays. Prefer the input-cache workflow below, especially
`preprocess_inputs.py --from-raw`.

For full DREAMT, use subject shards. A single dataset-level DREAMT raw-window
pickle can exceed memory because each `SensorSample` keeps full 30-second
64 Hz signal arrays.

Subject-shard examples:

```text
Processed/WESAD_binary_windows_manifest.json
Processed/WESAD_binary_windows_S2.pkl
Processed/WESAD_binary_windows_S3.pkl
Processed/DREAMT_binary_windows_manifest.json
Processed/DREAMT_binary_windows_S002.pkl
Processed/DREAMT_binary_windows_S003.pkl
...
```

Examples:

```powershell
.venv/bin/python preprocess_datasets.py -dataset WESAD --subjects S2 --overwrite
.venv/bin/python preprocess_datasets.py -dataset HHAR --data-dir "<HHAR_DATA_DIR>" --max-rows 200000 --overwrite
.venv/bin/python preprocess_datasets.py -dataset DREAMT --data-dir "<DREAMT_DATA_DIR>" --subjects S099 --overwrite
.venv/bin/python preprocess_datasets.py -dataset DREAMT --shard-by-subject --overwrite
```

Then run experiments from the cache:

```powershell
.venv/bin/python main.py -dataset WESAD -Input feature_description -LM direct -output label_only --use-processed --subjects S2 --balanced-per-label 1 --log-every 1
```

Use `--processed-dir` or `--processed-file` if the cache is stored somewhere else.

## Input Cache

Use `preprocess_inputs.py` to build the second cache layer. It reads
`Processed/<DATASET>_binary_windows.pkl` or a subject-shard manifest,
applies one Input module, and saves ready-to-use `LLMSample` objects with
`input_text`.

For full WESAD, the recommended path is to build input caches directly from raw
subject files. This reads one subject at a time, converts windows immediately to
text, and avoids saving very large raw-window pickle files.

After changing WESAD window parameters such as `stride_sec`, delete and rebuild
the WESAD input caches. Stale cache metadata should not show `stride_sec: 0.25`.

```powershell
Remove-Item .\Processed\WESAD_*_samples.pkl -Force
Remove-Item .\Processed\WESAD_*_samples.json -Force
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input feature_description --from-raw --overwrite
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input extra_knowledge --from-raw --overwrite
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input encoded_time_series --from-raw --overwrite
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input raw_data --from-raw --overwrite
```

Examples:

```powershell
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input feature_description --subjects S2 --overwrite
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input all --subjects S2 --overwrite
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input feature_description --overwrite
```

This creates files such as:

```text
Processed/WESAD_raw_data_samples.pkl
Processed/WESAD_feature_description_samples.pkl
Processed/WESAD_encoded_time_series_samples.pkl
Processed/WESAD_extra_knowledge_samples.pkl
```

Then run experiments directly from the input cache:

```powershell
.venv/bin/python main.py -dataset WESAD -Input feature_description -LM few_shot -output label_only --use-input-cache --train-subjects S2 --test-subjects S3 --balanced-per-label 1 --log-every 1
```

For `extra_knowledge`, the input cache depends on `knowledge_text`,
`knowledge_file`, and `knowledge_mode`. Rebuild the cache if those values change.

## LLM Evaluation Subsets

The current subset protocol uses fixed, reproducible evaluation subsets and
leave-one-subject-out few-shot examples:

- Evaluation subset seed: `42`.
- `debug`: 3 samples per label for smoke/debug runs.
- `pilot`: 50 samples per label for rough method checks.
- `main`: official subset size below.

| Dataset | `main` subset |
| --- | --- |
| WESAD | `160:160` label-balanced samples from available WESAD subjects |
| HHAR | 9 users, each user `50:50` |
| DREAMT | 100 subjects, each subject `5:5` |

DREAMT must be precomputed from the full `data_64Hz` subject set before this
strict subset step; a cache containing only smoke-test subjects will fail.

Generate fixed data subsets from preprocessed `SensorSample` windows first:

```powershell
python prepare_data_subsets.py -dataset all --overwrite
```

This writes dataset-level subsets under:

```text
Processed/DataSubsets/<DATASET>/<debug|pilot|main>/<DATASET>_<subset>_windows.pkl
```

Then build the four LLM input representations from exactly those fixed subsets:

```powershell
python prepare_subset_inputs.py -dataset all -Input all --overwrite
```

This writes `LLMSample` caches under:

```text
Processed/LLMSubsets/<DATASET>/<debug|pilot|main>/<DATASET>_<INPUT>_<subset>_samples.pkl
```

Each data subset JSON sidecar records `subset_spec` and a two-pass
`reproducibility_check`. The LLM input subset sidecar records
`source_data_subset_metadata`, so every input representation can be traced back
to the same sampled preprocessed windows. By default the script fails if the
official target size is unavailable; use `--allow-shortage` only for diagnostics.

Direct evaluation on an HHAR debug subset:

```powershell
python main.py -dataset HHAR -Input raw_data -LM direct -output label_only `
  --use-input-cache `
  --subject-split all `
  --eval-input-cache-file "Processed/LLMSubsets/HHAR/debug/HHAR_raw_data_debug_samples.pkl" `
  --log-every 1
```

Few-shot evaluation uses the same fixed subset input cache as the example
source and as evaluation data. For each evaluation subject, that same subject is
excluded from examples; then 5 other subjects are sampled, with 1 example per
subject per label, using seed `42`.

```powershell
python main.py -dataset HHAR -Input raw_data -LM few_shot -output label_only `
  --use-input-cache `
  --subject-split all `
  --train-input-cache-file "Processed/LLMSubsets/HHAR/debug/HHAR_raw_data_debug_samples.pkl" `
  --eval-input-cache-file "Processed/LLMSubsets/HHAR/debug/HHAR_raw_data_debug_samples.pkl" `
  --few-shot-example-selection leave_one_subject_out `
  --few-shot-example-subjects 3 `
  --few-shot-examples-per-subject-per-label 1 `
  --few-shot-example-max-chars 800 `
  --log-every 1
```

Prediction CSVs include `few_shot_example_subjects` and
`few_shot_example_count` for auditing subject leakage.

For dataset-size checks without saving samples or calling the LLM:

```powershell
.venv/bin/python count_dataset_samples.py -dataset WESAD --subjects S2
```

Run commands from the `experiment_4x3x2/` directory so relative dataset paths
such as `data_dir: ".."` resolve correctly.

## 72-Combination Script

Run the Slurm 72-combination pass from fixed subset caches. By default it uses
`SUBSET_LEVEL=debug` (3 samples per label):

```bash
sbatch run_3datasets_72_debug_sbatch.sh
```

Use the fixed main subsets instead:

```bash
sbatch --export=ALL,SUBSET_LEVEL=main run_3datasets_72_debug_sbatch.sh
```

The script expects fixed subset input caches
(`Processed/LLMSubsets/<DATASET>/<subset>/...`). Few-shot uses that same subset
cache as its example source and excludes the current evaluation subject at
prompt-build time.

## Few-Shot Runs

Few-shot runs default to leave-one-subject-out examples when launched through
the CLI. Use fixed subset caches for evaluation and full input caches for the
example source.

WESAD example:

```powershell
.venv/bin/python main.py -dataset WESAD -Input feature_description -LM few_shot -output label_only --use-input-cache --subject-split all --input-cache-dir Processed --eval-input-cache-file "Processed/LLMSubsets/WESAD/debug/WESAD_feature_description_debug_samples.pkl" --few-shot-example-selection leave_one_subject_out --few-shot-example-subjects 3 --few-shot-examples-per-subject-per-label 1 --few-shot-example-max-chars 800 --log-every 1
```

DREAMT example:

```powershell
.venv/bin/python main.py -dataset DREAMT -Input feature_description -LM few_shot -output label_only --data-dir "<DREAMT_DATA_DIR>" --train-subjects S002 --test-subjects S003 --few-shot-n-per-class 1 --few-shot-example-max-chars 800 --balanced-per-label 1 --log-every 1
```

HHAR subjects are HHAR user IDs from the CSV, such as `a`, `b`, etc. Use the IDs present in your local file.

## Results and Metrics

Results are saved under `Results/`.

Prediction CSV files include:

- `sample_id`
- `subject`
- `true_label`
- `predicted_label`
- `y_true`
- `y_pred`
- `valid`
- `parse_error`
- `raw_response`
- `input_type`
- `lm_type`
- `output_type`
- `few_shot_example_subjects`
- `few_shot_example_count`
- per-sample LLM usage and cost columns

Metrics JSON files include:

- Accuracy
- Macro-F1
- Weighted-F1
- confusion matrix
- true label distribution
- predicted label distribution
- invalid prediction count and rate
- LLM usage summary
- cost estimate
- optional scaling estimate

Invalid predictions are also counted in all-sample invalid-as-wrong metrics.

For small-sample cost profiling, the Slurm scripts write `gpu_usage.csv` with
continuous `nvidia-smi` telemetry. Combine a metrics JSON file and that GPU log:

```powershell
python summarize_cost_profile.py `
  --metrics-json "Results/<RUN>_metrics.json" `
  --gpu-csv "<LOGROOT>/gpu_usage.csv" `
  --output "<LOGROOT>/cost_profile.json"
```

## Full 72-Run Study

The complete binary study is:

```text
3 datasets x 4 inputs x 3 LM usages x 2 outputs = 72 runs
```

Recommended order:

1. Build fixed data subsets with `prepare_data_subsets.py`.
2. Build subset input caches with `prepare_subset_inputs.py`.
3. Run all 72 combinations with `SUBSET_LEVEL=debug`.
4. Confirm every combination saves valid output files.
5. Run larger subsets with `SUBSET_LEVEL=pilot` or `SUBSET_LEVEL=main`.

Run `multi_agent` last. It is the slowest path because it makes three LLM calls per sample.

## Installation

```powershell
.venv/bin/pip install -r requirements.txt
```

Core dependencies include:

- `numpy`
- `pandas`
- `scikit-learn`
- `requests`
- `PyYAML`
- `scipy`
- `neurokit2`
- `google-genai`

## Gemini Provider

The runner can call Gemini through the official `google-genai` SDK. Set the
API key as an environment variable instead of putting it in code or JSON:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
python main.py -dataset WESAD -Input feature_description -LM direct -output label_only --lm-provider gemini
```

For JSON/YAML configs, set:

```json
"lm_client": {
  "provider": "gemini",
  "model": "gemini-3.5-flash"
}
```

Gemini runs use larger default output limits than the local OpenAI-compatible
default to reduce truncated JSON: 384 tokens for `label_only` and 768 tokens
for `label_explanation`.

## Development Notes

- Do not use `legacy/` as the active experiment path.
- Do not add generated results or local raw data to Git.
- Keep `main.py` thin.
- Put method logic in `Dataset/`, `Input/`, `LM/`, `Output/`, `Evaluation/`, or `core/`.
- The maintained execution path is `main.py` / `run_experiment.py` -> `core/runner.py`.
