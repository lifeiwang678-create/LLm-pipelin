#!/usr/bin/env python
"""Re-parse saved raw_response in result CSVs with the current Output parser.

Use after improving the parser (e.g. truncation recovery): recovers predictions
for rows that were marked invalid but whose raw_response still contains a usable
predicted_state. Only touches currently-invalid rows; valid rows are unchanged.
Updates y_pred / predicted_label / valid / parse_error / explanation in place.

    python reparse_results.py                 # all model folders under Results/
    python reparse_results.py Results/Qwen3.6-35B-A3B
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

from Output.label_only import LabelOnlyOutput
from Output.label_explanation import LabelExplanationOutput

EXCLUDE_DIRS = {"multi-agents"}


def handler_for(output_type: str, labels):
    if output_type == "label_explanation":
        return LabelExplanationOutput(labels)
    return LabelOnlyOutput(labels)


def reparse_csv(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path)
    if "raw_response" not in df.columns or "valid" not in df.columns:
        return 0, 0
    labels = sorted(int(x) for x in pd.unique(df["y_true"].dropna()))
    if not labels:
        labels = [0, 1]
    recovered = 0
    invalid_before = int((~df["valid"].astype(bool)).sum())
    for idx, row in df.iterrows():
        if bool(row["valid"]):
            continue
        raw = row.get("raw_response")
        if not isinstance(raw, str) or not raw.strip():
            continue
        out_type = str(row.get("output_type", "label_only"))
        parsed = handler_for(out_type, labels).parse(raw)
        if parsed.get("valid") and parsed.get("label") is not None:
            lab = int(parsed["label"])
            df.at[idx, "y_pred"] = lab
            df.at[idx, "predicted_label"] = lab
            df.at[idx, "valid"] = True
            df.at[idx, "parse_error"] = ""
            if parsed.get("explanation"):
                df.at[idx, "explanation"] = parsed["explanation"]
            recovered += 1
    if recovered:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return recovered, invalid_before


def main():
    roots = sys.argv[1:] or [
        d for d in glob.glob("Results/*") if Path(d).is_dir() and Path(d).name not in EXCLUDE_DIRS
    ]
    total_recovered = total_invalid = 0
    for root in roots:
        files = sorted(glob.glob(str(Path(root) / "*.csv")))
        r_root = inv_root = 0
        for f in files:
            rec, inv = reparse_csv(Path(f))
            r_root += rec
            inv_root += inv
            if rec:
                print(f"  {Path(f).name}: recovered {rec}")
        print(f"{root}: recovered {r_root} / {inv_root} invalid rows across {len(files)} CSVs")
        total_recovered += r_root
        total_invalid += inv_root
    print(f"\nTOTAL recovered: {total_recovered} / {total_invalid} invalid")


if __name__ == "__main__":
    main()
