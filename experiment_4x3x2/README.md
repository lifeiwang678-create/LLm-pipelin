# Modular 4 x 3 x 2 LLM Experiment Framework

This folder is the current official experiment framework. Run experiments from inside this directory:

```powershell
cd experiment_4x3x2
python main.py -h
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
|-- prepare_llm_subsets.py
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
S099_whole_df.csv
```

Large local datasets are intentionally not tracked in Git.

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

- Loads raw 64 Hz DREAMT files such as `S099_whole_df.csv`.
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
- Requires explicit non-overlapping train/test subjects.
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
python main.py -dataset WESAD -Input raw_data -LM direct -output label_only --subjects S2 --balanced-per-label 1 --log-every 1
```

Small HHAR debug run:

```powershell
python main.py -dataset HHAR -Input raw_data -LM direct -output label_only --data-dir "<HHAR_DATA_DIR>" --max-rows 200000 --balanced-per-label 1 --log-every 1
```

Small DREAMT debug run:

```powershell
python main.py -dataset DREAMT -Input feature_description -LM direct -output label_only --data-dir "<DREAMT_DATA_DIR>" --subjects S099 --balanced-per-label 1 --log-every 1
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
Processed/DREAMT_binary_windows.pkl
```

Each `.pkl` contains one dataset-level list of window samples. Each sample keeps
its own `label`, `signals`, and metadata. Labels are for evaluation only; Input
modules must still avoid writing true labels into prompts.

For WESAD LLM experiments, avoid saving full raw-window dataset caches. Even
subject shards can become very large because each `SensorSample` contains full
signal arrays. Prefer the input-cache workflow below, especially
`preprocess_inputs.py --from-raw`.

Subject shards are kept only as an emergency/compatibility option:

```text
Processed/WESAD_binary_windows_manifest.json
Processed/WESAD_binary_windows_S2.pkl
Processed/WESAD_binary_windows_S3.pkl
...
```

Examples:

```powershell
python preprocess_datasets.py -dataset WESAD --subjects S2 --overwrite
python preprocess_datasets.py -dataset HHAR --data-dir "<HHAR_DATA_DIR>" --max-rows 200000 --overwrite
python preprocess_datasets.py -dataset DREAMT --data-dir "<DREAMT_DATA_DIR>" --subjects S099 --overwrite
```

Then run experiments from the cache:

```powershell
python main.py -dataset WESAD -Input feature_description -LM direct -output label_only --use-processed --subjects S2 --balanced-per-label 1 --log-every 1
```

Use `--processed-dir` or `--processed-file` if the cache is stored somewhere else.

## Input Cache

Use `preprocess_inputs.py` to build the second cache layer. It reads
`Processed/<DATASET>_binary_windows.pkl` or the WESAD subject-shard manifest,
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
python preprocess_inputs.py -dataset WESAD -Input feature_description --from-raw --overwrite
python preprocess_inputs.py -dataset WESAD -Input extra_knowledge --from-raw --overwrite
python preprocess_inputs.py -dataset WESAD -Input encoded_time_series --from-raw --overwrite
python preprocess_inputs.py -dataset WESAD -Input raw_data --from-raw --overwrite
```

Examples:

```powershell
python preprocess_inputs.py -dataset WESAD -Input feature_description --subjects S2 --overwrite
python preprocess_inputs.py -dataset WESAD -Input all --subjects S2 --overwrite
python preprocess_inputs.py -dataset WESAD -Input feature_description --overwrite
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
python main.py -dataset WESAD -Input feature_description -LM few_shot -output label_only --use-input-cache --train-subjects S2 --test-subjects S3 --balanced-per-label 1 --log-every 1
```

For `extra_knowledge`, the input cache depends on `knowledge_text`,
`knowledge_file`, and `knowledge_mode`. Rebuild the cache if those values change.

## LLM Evaluation Subsets

The recommended LLM protocol fixes train/test subjects before sampling:

- Few-shot examples are selected only from `train_subjects`.
- LLM evaluation samples are selected only from unseen `test_subjects`.
- The same subject must not appear in both few-shot examples and evaluation.

Default subject-independent splits are defined in `Dataset/registry.py`:

| Dataset | Train subjects | Unseen evaluation subjects |
| --- | --- | --- |
| WESAD | `S2 S3 S4 S5 S6` | `S7 S8` |
| HHAR | `a b c d` | `g h i` |
| DREAMT | `S099` | `S100` |

Generate the three LLM subset levels after input caches exist:

```powershell
python prepare_llm_subsets.py -dataset all -Input all --overwrite
```

This writes evaluation-only `LLMSample` caches under:

```text
Processed/LLMSubsets/<DATASET>/<debug|pilot|main>/
```

Subset levels:

- `debug`: 3 samples per label, used to check the 24-combination flow.
- `pilot`: up to 50 samples per label, used to compare rough method trends.
- `main`: subject-and-label balanced unseen-subject subset. By default this
  selects up to 100 samples per subject per label, clipped by available data.

Direct evaluation on an unseen HHAR debug subset:

```powershell
python main.py -dataset HHAR -Input raw_data -LM direct -output label_only `
  --use-input-cache `
  --eval-input-cache-file "Processed/LLMSubsets/HHAR/debug/HHAR_raw_data_debug_samples.pkl" `
  --log-every 1
```

Few-shot evaluation with examples from train subjects and evaluation from an
unseen subset:

```powershell
python main.py -dataset HHAR -Input raw_data -LM few_shot -output label_only `
  --use-input-cache `
  --train-subjects a b c d `
  --test-subjects g h i `
  --train-input-cache-file "Processed/HHAR_raw_data_samples.pkl" `
  --eval-input-cache-file "Processed/LLMSubsets/HHAR/debug/HHAR_raw_data_debug_samples.pkl" `
  --few-shot-n-per-class 1 `
  --few-shot-example-max-chars 800 `
  --log-every 1
```

For dataset-size checks without saving samples or calling the LLM:

```powershell
python count_dataset_samples.py -dataset WESAD --subjects S2
```

Run commands from the `experiment_4x3x2/` directory so relative dataset paths
such as `data_dir: ".."` resolve correctly.

## 72-Combination Script

Run a 72-combination debug pass from precomputed input caches:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_3datasets_4x3x2.ps1 -UseInputCache -BalancedPerLabel 1 -LogEvery 1
```

Run without balanced debug sampling:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all_3datasets_4x3x2.ps1 -UseInputCache -FullData -LogEvery 10
```

With `-FullData`, direct and multi-agent WESAD runs do not pass a subject filter.
Few-shot still requires explicit train/test subjects; override them with
`-WesadFewShotTrainSubjects` and `-WesadFewShotTestSubjects` if needed.

## Few-Shot Runs

Few-shot runs require explicit train/test subjects.

WESAD example:

```powershell
python main.py -dataset WESAD -Input feature_description -LM few_shot -output label_only --train-subjects S2 --test-subjects S3 --few-shot-n-per-class 1 --few-shot-example-max-chars 800 --balanced-per-label 1 --log-every 1
```

DREAMT example:

```powershell
python main.py -dataset DREAMT -Input feature_description -LM few_shot -output label_only --data-dir "<DREAMT_DATA_DIR>" --train-subjects S002 --test-subjects S003 --few-shot-n-per-class 1 --few-shot-example-max-chars 800 --balanced-per-label 1 --log-every 1
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

## Full 72-Run Study

The complete binary study is:

```text
3 datasets x 4 inputs x 3 LM usages x 2 outputs = 72 runs
```

Recommended order:

1. Run all 72 combinations with `--balanced-per-label 1`.
2. Confirm every combination saves valid output files.
3. Increase `--balanced-per-label` for a larger subset.
4. Run full data without `--balanced-per-label`.

Run `multi_agent` last. It is the slowest path because it makes three LLM calls per sample.

## Installation

```powershell
pip install -r requirements.txt
```

Core dependencies include:

- `numpy`
- `pandas`
- `scikit-learn`
- `requests`
- `PyYAML`
- `scipy`
- `neurokit2`

## Development Notes

- Do not use `legacy/` as the active experiment path.
- Do not add generated results or local raw data to Git.
- Keep `main.py` thin.
- Put method logic in `Dataset/`, `Input/`, `LM/`, `Output/`, `Evaluation/`, or `core/`.
- The maintained execution path is `main.py` / `run_experiment.py` -> `core/runner.py`.
