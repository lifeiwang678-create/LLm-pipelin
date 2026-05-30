# LLm-pipelin

The current official experiment framework lives in:

```text
experiment_4x3x2/
```

Use it from inside that folder:

```powershell
cd experiment_4x3x2
.venv/bin/python main.py -h
```

For a step-by-step Chinese runbook, see `experiment_4x3x2/RUNNING.md`.

`experiment_4x3x2/` contains the maintained binary-classification 4 x 3 x 2 modular pipeline:

- Datasets: `WESAD`, `HHAR`, `DREAMT`
- Inputs: `raw_data`, `feature_description`, `encoded_time_series`, `extra_knowledge`
- LM usage: `direct`, `few_shot`, `multi_agent`
- Outputs: `label_only`, `label_explanation`

Current binary task definitions:

| Dataset | Label 0 | Label 1 |
| --- | --- | --- |
| `WESAD` | no stress | stress |
| `HHAR` | walking downstairs | walking upstairs |
| `DREAMT` | wake | sleep |

The complete study has 72 combinations:

```text
3 datasets x 4 input types x 3 LM usage types x 2 output formats
```

For faster repeated experiments, the official framework supports two cache layers:

- Dataset cache: `Processed/<DATASET>_binary_windows.pkl`
- Input cache: `Processed/<DATASET>_<INPUT>_samples.pkl`

Build them from inside `experiment_4x3x2/`:

```powershell
.venv/bin/python preprocess_datasets.py -dataset WESAD --subjects S2 S3 --overwrite
.venv/bin/python preprocess_inputs.py -dataset WESAD -Input all --subjects S2 S3 --overwrite
```

Then run from cache with `--use-processed` or `--use-input-cache`.

`legacy/` is for reference-only scripts from older experiment paths. Do not use it as the current experiment entry.

`embedding_alignment` remains accepted inside the code only as a backward-compatible alias for `encoded_time_series`.

Local raw data, generated results, model checkpoints, and historical output folders are intentionally not uploaded to GitHub.
