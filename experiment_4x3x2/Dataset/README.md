# Dataset

Dataset loaders live in this folder. Raw and processed data files are intentionally ignored by Git.

Current default data locations are defined in `Dataset/registry.py`:

- `WESAD`: outer repository subject folders such as `../S2/`, `../S3/`, ... (`data_dir: ".."`)
- `HHAR`: `Dataset/HHAR/`
- `DREAMT`: `Dataset/DREAMT/`

WESAD uses LLM-friendly windowing defaults:

```json
{
  "physiology_window_sec": 60.0,
  "acc_window_sec": 5.0,
  "stride_sec": 60.0
}
```

This keeps the 60-second physiology and 5-second ACC windows, but avoids dense
0.25-second overlap because each window becomes an LLM prompt. Use an explicit
config override only when reproducing traditional sliding-window ML settings.

Use these keys in new configs. The older `window_sec` key is retained only for compatibility with older experiments.

HHAR uses a HARGPT-style binary stair task:

```text
0 = walking downstairs
1 = walking upstairs
```

The loader keeps only downstairs/upstairs phone IMU samples, removes invalid or
null labels, uses phone accelerometer plus gyroscope when available, downsamples
the IMU stream to 10 Hz tokens, and then creates 2-second windows with 1-second
stride:

```json
{
  "window_size": 20,
  "stride_size": 10,
  "sampling_rate": 10.0,
  "include_gyroscope": true
}
```

For few-shot HHAR experiments, use explicit user-level `--train-subjects` and
`--test-subjects` so examples and evaluation samples do not share users.

The framework now supports reusable processed dataset caches. Run from the
`experiment_4x3x2/` folder:

```powershell
python preprocess_datasets.py -dataset WESAD --subjects S2 S3 --overwrite
python preprocess_datasets.py -dataset HHAR --data-dir "<HHAR_DATA_DIR>" --overwrite
python preprocess_datasets.py -dataset DREAMT --data-dir "<DREAMT_DATA_DIR>" --subjects S099 S100 --overwrite
```

For full WESAD LLM experiments, prefer `preprocess_inputs.py --from-raw`
instead of saving raw-window dataset caches. WESAD raw-window caches can become
very large because they keep full sensor arrays for every window.

This saves dataset-level binary window files:

```text
Processed/WESAD_binary_windows.pkl
Processed/HHAR_binary_windows.pkl
Processed/DREAMT_binary_windows.pkl
```

These cache files contain `SensorSample` objects with `signals`, `label`, and
metadata. They are ignored by Git.

Input-level caches are generated separately with `preprocess_inputs.py`, for
example:

```powershell
python preprocess_inputs.py -dataset WESAD -Input all --subjects S2 S3 --overwrite
```

You can override the location in a config file with:

```json
{
  "dataset": {
    "name": "WESAD",
    "data_dir": "../Dataset/WESAD"
  }
}
```
