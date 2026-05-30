#!/usr/bin/env python
"""Aggregate per-(model, combination) metrics into a single Results/statistics.csv.

One row per source x (dataset, input, lm, output). Sources are each model folder
under Results/ plus the majority-vote folder (multi-agents). Metrics: accuracy
(valid-only), accuracy_all (invalid counted wrong), macro/weighted F1, counts.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

RESULTS = Path("Results")
VOTE_DIR = "multi-agents"
KEY = ["dataset", "input_type", "lm_type", "output_type"]


def metrics_for(df: pd.DataFrame) -> dict:
    n = len(df)
    valid = df[df["valid"].astype(bool)].copy()
    n_valid = len(valid)
    labels = sorted(int(x) for x in pd.unique(df["y_true"].dropna()))
    out = {"n_samples": n, "n_valid": n_valid, "invalid": n - n_valid}
    if n_valid:
        yt = valid["y_true"].astype(int)
        yp = valid["y_pred"].astype(int)
        out["accuracy"] = round(accuracy_score(yt, yp), 4)
        out["macro_f1"] = round(f1_score(yt, yp, labels=labels, average="macro", zero_division=0), 4)
        out["weighted_f1"] = round(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0), 4)
        # accuracy counting invalid as wrong (over all samples)
        out["accuracy_all"] = round(int((yt.values == yp.values).sum()) / n, 4) if n else None
    else:
        out.update(accuracy=None, macro_f1=None, weighted_f1=None, accuracy_all=None)
    return out


def source_dirs():
    dirs = [d for d in RESULTS.iterdir() if d.is_dir() and not d.name.startswith(".")]
    # models first, vote last
    models = sorted(d for d in dirs if d.name != VOTE_DIR)
    vote = [d for d in dirs if d.name == VOTE_DIR]
    return models + vote


def main():
    rows = []
    for d in source_dirs():
        source = "majority_vote" if d.name == VOTE_DIR else d.name
        for f in sorted(d.glob("*.csv")):
            df = pd.read_csv(f)
            if not {"y_true", "y_pred", "valid", *KEY} <= set(df.columns):
                continue
            r = df.iloc[0]
            rec = {
                "source": source,
                "dataset": r["dataset"],
                "input": r["input_type"],
                "lm": r["lm_type"],
                "output": r["output_type"],
            }
            rec.update(metrics_for(df))
            rows.append(rec)

    stats = pd.DataFrame(rows).sort_values(
        ["dataset", "input", "lm", "output", "source"]
    ).reset_index(drop=True)
    out_path = RESULTS / "statistics.csv"
    stats.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(stats)} rows -> {out_path}")
    return stats


if __name__ == "__main__":
    main()
