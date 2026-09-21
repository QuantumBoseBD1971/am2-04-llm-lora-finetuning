# AM2 Evidence Notes

This repository demonstrates four progressively more specialised LLM adaptation strategies.

## Prompt engineering

Phase 1 establishes zero-shot and fixed few-shot prompting.

## Retrieval-assisted prompting

Phase 2 dynamically retrieves semantically related demonstrations without modifying model parameters.

## LoRA / PEFT

Phase 3 adds parameter-efficient adaptation and records:

- LoRA rank / alpha / dropout
- target modules
- total parameters
- trainable parameters
- trainable percentage
- base-vs-adapter evaluation

## Robustness and error analysis

Phase 4 adds deterministic error flags and per-strategy summaries, including:

- exact-match errors
- partial token matches
- empty outputs

## Experiment tracking

A lightweight immutable JSONL registry records:

- run id
- timestamp
- strategy
- parameters
- metrics
- artefact references
- notes

## Data governance

The project documents training-data provenance, licensing, train/evaluation separation, leakage control, sensitive-data review and retention considerations.

## MLOps

The deployment design treats the base model and adapter as separately versioned artefacts.

It covers:

- immutable adapters
- promotion aliases
- monitoring
- regression checks
- rollback
- upstream base-model revision tracking

## Responsible AI

The local fixture is intentionally too small for claims about general model quality.

A real fine-tuning project requires a licensed and representative training set, a held-out evaluation set, broader regression testing and appropriate human oversight.

## Reflection

The project demonstrates that prompting, retrieval and LoRA are different adaptation mechanisms with different operational costs. Model quality must therefore be considered alongside parameter efficiency, governance, reproducibility and lifecycle complexity.
