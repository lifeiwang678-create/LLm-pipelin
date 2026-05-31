import json
import sys
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])

def find_numbers(obj, key):
    vals = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (int, float)):
                vals.append(int(v))
            else:
                vals.extend(find_numbers(v, key))
    elif isinstance(obj, list):
        for x in obj:
            vals.extend(find_numbers(x, key))
    return vals

def parse_name(name):
    name = name.replace("_metrics.json", "")
    parts = name.split("_")
    dataset = parts[0]

    if "_label_explanation_" in name:
        output = "label_explanation"
        base = name.split("_label_explanation_")[0]
    elif "_label_only_" in name:
        output = "label_only"
        base = name.split("_label_only_")[0]
    else:
        output = "unknown"
        base = name

    base_parts = base.split("_")
    dataset = base_parts[0]

    if base.endswith("_few_shot"):
        lm = "few_shot"
        input_type = "_".join(base_parts[1:-2])
    elif base.endswith("_direct"):
        lm = "direct"
        input_type = "_".join(base_parts[1:-1])
    elif base.endswith("_multi_agent"):
        lm = "multi_agent"
        input_type = "_".join(base_parts[1:-2])
    else:
        lm = "unknown"
        input_type = "_".join(base_parts[1:])

    return dataset, input_type, lm, output

rows = []
by_dataset = defaultdict(lambda: defaultdict(int))
by_combo = defaultdict(lambda: defaultdict(int))

for f in sorted(root.rglob("*_metrics.json")):
    data = json.loads(f.read_text(encoding="utf-8"))

    dataset, input_type, lm, output = parse_name(f.name)

    total_tokens_list = find_numbers(data, "total_tokens")
    prompt_tokens_list = find_numbers(data, "prompt_tokens")
    completion_tokens_list = find_numbers(data, "completion_tokens")
    llm_calls_list = find_numbers(data, "llm_calls")

    total_tokens = sum(total_tokens_list)
    prompt_tokens = sum(prompt_tokens_list)
    completion_tokens = sum(completion_tokens_list)
    llm_calls = sum(llm_calls_list)

    rows.append({
        "file": f.name,
        "dataset": dataset,
        "input": input_type,
        "lm": lm,
        "output": output,
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    })

    for group in (by_dataset[dataset], by_combo[(dataset, input_type, lm, output)]):
        group["files"] += 1
        group["llm_calls"] += llm_calls
        group["prompt_tokens"] += prompt_tokens
        group["completion_tokens"] += completion_tokens
        group["total_tokens"] += total_tokens

total = defaultdict(int)
for r in rows:
    total["files"] += 1
    total["llm_calls"] += r["llm_calls"]
    total["prompt_tokens"] += r["prompt_tokens"]
    total["completion_tokens"] += r["completion_tokens"]
    total["total_tokens"] += r["total_tokens"]

print("=== TOTAL SUCCESSFUL RESULTS ===")
for k, v in total.items():
    print(f"{k}: {v:,}")

print("\n=== BY DATASET ===")
for dataset, s in sorted(by_dataset.items()):
    print(
        f"{dataset}: files={s['files']}, "
        f"llm_calls={s['llm_calls']:,}, "
        f"prompt_tokens={s['prompt_tokens']:,}, "
        f"completion_tokens={s['completion_tokens']:,}, "
        f"total_tokens={s['total_tokens']:,}"
    )

out = root / "token_summary_success_by_combo.csv"
with out.open("w", encoding="utf-8", newline="") as w:
    w.write("dataset,input,lm,output,files,llm_calls,prompt_tokens,completion_tokens,total_tokens\n")
    for (dataset, input_type, lm, output), s in sorted(by_combo.items()):
        w.write(
            f"{dataset},{input_type},{lm},{output},"
            f"{s['files']},{s['llm_calls']},{s['prompt_tokens']},"
            f"{s['completion_tokens']},{s['total_tokens']}\n"
        )

print(f"\nsaved: {out}")
