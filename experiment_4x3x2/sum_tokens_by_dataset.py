import json
import sys
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])

def find_number(obj, keys):
    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        for v in obj.values():
            r = find_number(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_number(v, keys)
            if r is not None:
                return r
    return None

summary = defaultdict(lambda: defaultdict(int))
total = defaultdict(int)

files = sorted(root.rglob("*_metrics.json"))

for f in files:
    data = json.loads(f.read_text(encoding="utf-8"))

    name = f.name.replace("_metrics.json", "")
    parts = name.split("_")
    dataset = parts[0]

    llm_calls = find_number(data, ["llm_calls"]) or 0
    prompt_tokens = find_number(data, ["prompt_tokens"]) or 0
    completion_tokens = find_number(data, ["completion_tokens"]) or 0
    total_tokens = find_number(data, ["total_tokens"]) or 0

    summary[dataset]["files"] += 1
    summary[dataset]["llm_calls"] += llm_calls
    summary[dataset]["prompt_tokens"] += prompt_tokens
    summary[dataset]["completion_tokens"] += completion_tokens
    summary[dataset]["total_tokens"] += total_tokens

    total["files"] += 1
    total["llm_calls"] += llm_calls
    total["prompt_tokens"] += prompt_tokens
    total["completion_tokens"] += completion_tokens
    total["total_tokens"] += total_tokens

print("=== Total ===")
print(dict(total))

print("\n=== By dataset ===")
for dataset, s in sorted(summary.items()):
    print(dataset, dict(s))

out = root / "token_summary_by_dataset.json"
out.write_text(json.dumps({"total": dict(total), "by_dataset": {k: dict(v) for k, v in summary.items()}}, indent=2), encoding="utf-8")
print("\nsaved:", out)
