# Custom Agents

This repository defines workspace-specific agent support for AI coding assistance.

## Available Agents

- `codex`
  - File: `.github/agents/codex.agent.md`
  - Purpose: Workspace-focused coding assistant for the LLm-pipelin experiment framework.
  - Use when working on repository code, especially `experiment_4x3x2`, `core`, dataset loaders, input formatting, LM orchestration, output handling, and evaluation logic.

## Notes

- `experiment_4x3x2/` is the maintained production experiment framework.
- `legacy/` contains older reference scripts and should not be updated unless a regression or compatibility request explicitly requires it.
- The repository uses a modular design around dataset × input type × LM usage × output format.
