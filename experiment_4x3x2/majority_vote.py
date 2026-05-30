#!/usr/bin/env python
"""Majority-vote ensemble over per-model results in Results/.

Scans every model sub-folder under Results/ (each model writes its own
Results/<MODEL_TAG>/ via --output-dir), aligns predictions by sample_id within
each (dataset, input, lm, output) combination, and writes the per-sample
majority vote to Results/multi-agents/.

The output folder (default "multi-agents") is itself skipped when scanning, so
re-running never votes on its own output. New model folders are picked up
automatically.

Usage:
    python majority_vote.py
    python majority_vote.py --results-dir Results --out multi-agents
    python majority_vote.py --tie-break min            # smallest label on ties
    python majority_vote.py --tie-break Qwen3.6-35B-A3B # this model breaks ties
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

KEY_COLS = ["dataset", "input_type", "lm_type", "output_type"]
NEEDED = {"sample_id", "y_true", "y_pred", "valid", *KEY_COLS}


def parse_args():
    p = argparse.ArgumentParser(description="Majority-vote ensemble over Results/<model>/ folders.")
    p.add_argument("--results-dir", default="Results", help="Root results directory.")
    p.add_argument("--out", default="multi-agents", help="Output sub-folder name under results-dir (also excluded from scanning).")
    p.add_argument("--tie-break", default="min",
                   help='Tie handling: "min" (smallest label, default), "max", or a model folder name whose vote wins ties.')
    p.add_argument("--min-models", type=int, default=2,
                   help="Skip combinations covered by fewer than this many models (default 2).")
    return p.parse_args()


def model_dirs(results: Path, out_name: str):
    return sorted(d for d in results.iterdir()
                  if d.is_dir() and d.name != out_name and not d.name.startswith("."))


def combo_key(df: pd.DataFrame):
    r = df.iloc[0]
    return tuple(str(r[c]) for c in KEY_COLS)


def latest_per_combo(model_dir: Path):
    """combo_key -> DataFrame (latest CSV by mtime for that combo)."""
    best: dict[tuple, tuple[float, Path, pd.DataFrame]] = {}
    for f in model_dir.glob("*.csv"):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) == 0 or not NEEDED <= set(df.columns):
            continue
        key = combo_key(df)
        mt = f.stat().st_mtime
        if key not in best or mt > best[key][0]:
            best[key] = (mt, f, df)
    return {k: v[2] for k, v in best.items()}


def valid_pred(row):
    """Return int label if this row is a valid prediction, else None."""
    v = row.get("valid")
    if v is False or str(v).lower() in ("false", "0", "nan", ""):
        return None
    yp = row.get("y_pred")
    if pd.isna(yp) or str(yp) == "":
        return None
    try:
        return int(float(yp))
    except (TypeError, ValueError):
        return None


def vote_one(per_model_pred: dict[str, int | None], tie_break: str):
    """Given {model: label|None}, return (voted_label|None, n_valid, votes_for_pred, tie)."""
    votes = {m: lab for m, lab in per_model_pred.items() if lab is not None}
    if not votes:
        return None, 0, 0, False
    counts = Counter(votes.values())
    top = counts.most_common()
    best_n = top[0][1]
    leaders = sorted(lab for lab, n in top if n == best_n)
    tie = len(leaders) > 1
    if not tie:
        pred = leaders[0]
    elif tie_break == "min":
        pred = leaders[0]
    elif tie_break == "max":
        pred = leaders[-1]
    else:  # a model name breaks the tie
        pref = votes.get(tie_break)
        pred = pref if pref in leaders else leaders[0]
    return pred, len(votes), counts[pred], tie


def main():
    args = parse_args()
    results = Path(args.results_dir)
    out_dir = results / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    models = model_dirs(results, args.out)
    if not models:
        print(f"No model folders found under {results}/ (excluding '{args.out}').")
        return
    print(f"Models found ({len(models)}): {[m.name for m in models]}")

    # combo_key -> {model_name: df}
    combos: dict[tuple, dict[str, pd.DataFrame]] = {}
    for md in models:
        for key, df in latest_per_combo(md).items():
            combos.setdefault(key, {})[md.name] = df

    summary = []
    written = 0
    for key in sorted(combos):
        dataset, inp, lm, out = key
        model_to_df = combos[key]
        if len(model_to_df) < args.min_models:
            print(f"  skip {key}: only {len(model_to_df)} model(s) "
                  f"({list(model_to_df)}) < min-models={args.min_models}")
            continue

        # index each model's rows by sample_id (keep first if dup)
        indexed = {m: df.drop_duplicates("sample_id").set_index("sample_id")
                   for m, df in model_to_df.items()}
        # row order: from the model with the most samples
        anchor = max(model_to_df.values(), key=len)
        ordered_ids = list(dict.fromkeys(anchor["sample_id"].tolist()))
        extra = sorted(set().union(*[set(ix.index) for ix in indexed.values()]) - set(ordered_ids),
                       key=str)
        ordered_ids += extra

        rows = []
        per_model_acc = {m: [0, 0] for m in indexed}   # [correct, total_valid]
        vote_correct = vote_total = 0
        for sid in ordered_ids:
            y_true = None
            per_model_pred = {}
            subject = ""
            for m, ix in indexed.items():
                if sid in ix.index:
                    r = ix.loc[sid]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    if y_true is None and not pd.isna(r.get("y_true")):
                        y_true = int(r["y_true"]); subject = r.get("subject", "")
                    lab = valid_pred(r)
                    per_model_pred[m] = lab
                    if lab is not None and y_true is not None:
                        per_model_acc[m][1] += 1
                        per_model_acc[m][0] += int(lab == y_true)
                else:
                    per_model_pred[m] = None

            pred, n_valid, votes_for, tie = vote_one(per_model_pred, args.tie_break)
            valid = pred is not None
            if valid and y_true is not None:
                vote_total += 1
                vote_correct += int(pred == y_true)

            row = {
                "sample_id": sid, "dataset": dataset, "subject": subject,
                "input_type": inp, "lm_type": lm, "output_type": out,
                "y_true": y_true,
                "y_pred": pred if valid else "",
                "valid": valid,
                "n_models": len(indexed), "n_valid": n_valid,
                "votes_for_pred": votes_for, "tie": tie,
            }
            for m in indexed:
                row[f"vote__{m}"] = "" if per_model_pred[m] is None else per_model_pred[m]
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_path = out_dir / f"{dataset}_{inp}_{lm}_{out}.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        written += 1

        vote_acc = vote_correct / vote_total if vote_total else None
        rec = {
            "dataset": dataset, "input": inp, "lm": lm, "output": out,
            "n_models": len(indexed), "models": list(indexed),
            "n_samples": len(out_df),
            "vote_accuracy": round(vote_acc, 4) if vote_acc is not None else None,
            "model_accuracy": {m: (round(c / t, 4) if t else None) for m, (c, t) in per_model_acc.items()},
            "file": str(out_path),
        }
        summary.append(rec)
        accs = " ".join(f"{m.split('-')[0][:8]}={rec['model_accuracy'][m]}" for m in indexed)
        print(f"  {dataset}/{inp}/{lm}/{out}: {len(indexed)} models, {len(out_df)} samples | "
              f"VOTE={rec['vote_accuracy']} | {accs}")

    (out_dir / "_vote_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # overall mean accuracy comparison
    if summary:
        import statistics
        va = [s["vote_accuracy"] for s in summary if s["vote_accuracy"] is not None]
        print(f"\nWrote {written} voted combination CSVs to {out_dir}/")
        print(f"Summary: {out_dir}/_vote_summary.json")
        if va:
            print(f"Mean vote accuracy over {len(va)} combos: {statistics.mean(va):.4f}")


if __name__ == "__main__":
    main()
