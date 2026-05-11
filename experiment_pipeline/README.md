# Modular WESAD LLM Experiments

This folder separates the experiment into three replaceable layers:

- `inputs.py`: builds the input representation.
  Interfaces: `raw_data`, `feature_description`, `embedding_alignment`, `extra_knowledge`.
  Implemented now: `raw_data`, `feature_description`.
- `lm_usage.py`: builds the LLM prompt strategy.
  Interfaces: `direct`, `few_shot`, `multi_agent`.
- `outputs.py`: parses the model output.
  Interfaces: `label_only`, `label_explanation`.

Run an experiment with:

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
