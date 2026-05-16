# Dataset

Dataset loaders live in this folder. Raw and processed data files are intentionally ignored by Git.

Current default data locations are defined in `Dataset/registry.py`:

- `WESAD`: repository-root subject folders such as `S2/`, `S3/`, ... (`data_dir: "."`)
- `HHAR`: `Dataset/HHAR/`
- `DREAMT`: `Dataset/DREAMT/`

You can override the location in a config file with:

```json
{
  "dataset": {
    "name": "WESAD",
    "data_dir": "Dataset/WESAD"
  }
}
```
