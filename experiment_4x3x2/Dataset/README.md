# Dataset

Dataset loaders live in this folder. Raw and processed data files are intentionally ignored by Git.

Current default data locations are defined in `Dataset/registry.py`:

- `WESAD`: outer repository subject folders such as `../S2/`, `../S3/`, ... (`data_dir: ".."`)
- `HHAR`: `Dataset/HHAR/`
- `DREAMT`: `Dataset/DREAMT/`

WESAD uses paper-style windowing defaults:

```json
{
  "physiology_window_sec": 60.0,
  "acc_window_sec": 5.0,
  "stride_sec": 0.25
}
```

Use these keys in new configs. The older `window_sec` key is retained only for compatibility with older experiments.

You can override the location in a config file with:

```json
{
  "dataset": {
    "name": "WESAD",
    "data_dir": "../Dataset/WESAD"
  }
}
```
