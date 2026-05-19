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
- `embedding_alignment`
- `extra_knowledge`

`encoded_time_series` is accepted as an alias for `embedding_alignment`.

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
|-- configs/
|-- main.py
|-- run_experiment.py
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
- Uses paper-style windows:
  - physiology window: 60 seconds
  - ACC window: 5 seconds
  - stride: 0.25 seconds
- Maps original states to binary no-stress/stress labels.

`Dataset/hhar_loader.py`

- Loads `Phones_accelerometer.csv`.
- Filters to upstairs/downstairs stair activities only.
- Uses 2-second windows with 50% overlap by default:
  - `window_size = 128`
  - `stride_size = 64`
  - `sampling_rate = 64`
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

`embedding_alignment` / `encoded_time_series`

- SensorLLM-inspired textual encoded time-series input.
- Channel-aware and segment-level temporal descriptions.
- Prompt-compatible only.
- It does not train projectors, modify LLM embeddings, run Chronos, or replace time-series token embeddings.

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
python main.py -dataset WESAD -Input raw_data -LM direct -output label --subjects S2 --balanced-per-label 1 --log-every 1
```

Small HHAR debug run:

```powershell
python main.py -dataset HHAR -Input raw_data -LM direct -output label --data-dir "<HHAR_DATA_DIR>" --max-rows 200000 --balanced-per-label 1 --log-every 1
```

Small DREAMT debug run:

```powershell
python main.py -dataset DREAMT -Input feature_description -LM direct -output label --data-dir "<DREAMT_DATA_DIR>" --subjects S099 --balanced-per-label 1 --log-every 1
```

Use `--balanced-per-label` only for debug subsets. Formal full-data runs should omit it.

Use `--max-rows` only for debugging large CSV datasets such as HHAR. Formal HHAR runs should omit it.

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
