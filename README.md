# LLm-pipelin

The current official experiment framework lives in:

```text
experiment_4x3x2/
```

Use it from inside that folder:

```powershell
cd experiment_4x3x2
python main.py -h
```

`experiment_4x3x2/` contains the maintained 4 x 3 x 2 modular pipeline:

- Datasets: `WESAD`, `HHAR`, `DREAMT`
- Inputs: `raw_data`, `feature_description`, `embedding_alignment`, `extra_knowledge`
- LM usage: `direct`, `few_shot`, `multi_agent`
- Outputs: `label_only`, `label_explanation`

`legacy/` is for reference-only scripts from older experiment paths. Do not use it as the current experiment entry.

Local raw data, generated results, model checkpoints, and historical output folders are intentionally not uploaded to GitHub.
