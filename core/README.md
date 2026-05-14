# Modular WESAD LLM Experiments

This folder separates the experiment into three replaceable layers:

- The top-level `Input/`, `LM/`, and `Output/` folders contain the concrete method logic used by `main.py`.
- This `core/` folder keeps CLI parsing, module composition, shared schema, LM client, evaluation helpers, and legacy config runner support.
- `inputs.py`: builds the input representation.
  Interfaces: `raw_data`, `feature_description`, `embedding_alignment`, `extra_knowledge`.
  Implemented now: `raw_data`, `feature_description`.
- `lm_usage.py`: builds the LLM prompt strategy.
  Interfaces: `direct`, `few_shot`, `multi_agent`.
- `outputs.py`: parses the model output.
  Interfaces: `label_only`, `label_explanation`.

Run an experiment with:

```powershell
python main.py -dataset WESAD -Input raw_data -LM direct -output label_only
```

`main.py` saves compact CSV files to `Results/`, for example:

```text
WESAD_raw_data_direct_label_only_20260512213815.csv
```

The config-based runner is still available:

```powershell
python run_experiment.py --config configs/example_experiment.json
```

Change combinations by editing the config:

```json
{
  "input": {"type": "raw_data"},
  "lm_usage": {"type": "multi_agent"},
  "output": {"type": "label_explanation"}
}
```

For 3-class experiments, change `labels` to `[1, 2, 3]`.

`embedding_alignment` and `extra_knowledge` are intentionally registered as interfaces first.
Selecting either one will raise `NotImplementedError` until its `load()` method is filled in.

Parsing failures are not converted to a default label. They are saved as invalid predictions and excluded from accuracy.

For debug runs, use balanced sampling:

```json
{
  "evaluation": {
    "balanced_per_label": 10,
    "log_every": 10
  }
}
```

Few-shot runs must explicitly set both `data.train_subjects` and `data.test_subjects`; overlapping subjects are rejected.

## Experiment Matrix

| ID | Input | LM usage | Output | Config |
| -- | -- | -- | -- | -- |
| E1 | `raw_data` | `direct` | `label_only` | `configs/E1_raw_direct_label_only.json` |
| E2 | `feature_description` | `direct` | `label_only` | `configs/E2_feature_direct_label_only.json` |
| E3 | `raw_data` | `few_shot` | `label_only` | `configs/E3_raw_fewshot_label_only.json` |
| E4 | `feature_description` | `few_shot` | `label_only` | `configs/E4_feature_fewshot_label_only.json` |

Run one experiment:

```powershell
python run_experiment.py --config configs/E1_raw_direct_label_only.json
```
