import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Read SensorLLM eval metrics from trainer_state.json.")
    parser.add_argument(
        "path",
        nargs="?",
        default=r".\sensorllm_wesad_binary_output_formal\fold_S2",
        help="Fold output directory or trainer_state.json path.",
    )
    return parser.parse_args()


args = parse_args()
input_path = Path(args.path)
trainer_state_path = input_path if input_path.name == "trainer_state.json" else input_path / "trainer_state.json"

if not trainer_state_path.exists():
    raise FileNotFoundError(f"Cannot find: {trainer_state_path}")

with trainer_state_path.open("r", encoding="utf-8") as f:
    state = json.load(f)

eval_logs = [
    log for log in state.get("log_history", [])
    if any(k.startswith("eval_") for k in log.keys())
]

print("\n================ All Eval Logs ================\n")

for i, log in enumerate(eval_logs, 1):
    print(f"[Eval {i}]")
    for k, v in log.items():
        if k.startswith("eval_") or k in ["epoch", "step"]:
            print(f"{k}: {v}")
    print()

print("\n================ Best Metric ================\n")
print("best_metric:", state.get("best_metric"))
print("best_model_checkpoint:", state.get("best_model_checkpoint"))

if eval_logs:
    best_log = max(
        eval_logs,
        key=lambda x: x.get("eval_f1_macro", x.get("eval_f1", -1)),
    )

    print("\n================ Best Eval Metrics ================\n")
    for k in [
        "eval_accuracy",
        "eval_f1_macro",
        "eval_f1_weighted",
        "eval_precision_macro",
        "eval_recall_macro",
        "eval_loss",
        "epoch",
        "step",
    ]:
        if k in best_log:
            print(f"{k}: {best_log[k]}")
