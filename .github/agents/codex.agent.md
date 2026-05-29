---
name: codex
description: "Workspace-specific coding agent for the LLm-pipelin experiment framework. Use for code edits, analysis, and design tasks within experiment_4x3x2, core modules, Dataset/Input/LM/Output logic, and related configs."
argument-hint: "A coding task, implementation request, or repository-specific question about LLm-pipelin, especially the experiment_4x3x2 framework."
applyTo:
  - "experiment_4x3x2/**"
  - "core/**"
  - "Dataset/**"
  - "Input/**"
  - "LM/**"
  - "Output/**"
  - "Evaluation/**"
  - "README.md"
# tools: ['vscode', 'read', 'edit', 'search', 'todo']
---

This custom agent is the primary workspace assistant for LLm-pipelin.

Use it when working on:
- the main experiment flow (`experiment_4x3x2/main.py`, `core/runner.py`)
- dataset loading and preprocessing modules
- input formatting and prompt generation
- LM usage and output formatting
- evaluation, metrics, and results handling

Important conventions:
- `experiment_4x3x2/` is the maintained production experiment framework.
- `legacy/` contains reference-only older scripts; do not modify legacy code unless explicitly requested for compatibility or regression fixes.
- The 4 × 3 × 2 design is the core architecture: dataset × input type × LM usage × output format.
- Dataset defaults and paths are managed by `Dataset/registry.py`.
- Binary label mappings are defined in `core/schema.py` and should be preserved for dataset-specific tasks.
- Use `experiment_4x3x2/README.md` for architectural guidance, and prefer local `requirements.txt` environment details when suggesting runtime commands.

Avoid making changes to large external datasets, generated results, or environment-specific artifacts that are not tracked in the repository.
