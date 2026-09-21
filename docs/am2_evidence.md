# AM2 Evidence Notes

This document evolves with the LLM adaptation project.

## Problem framing

The project compares increasingly specialised adaptation strategies rather than assuming model fine-tuning is always required.

## Prompt engineering

Phase 1 establishes zero-shot and fixed few-shot prompting.

## Retrieval-assisted prompting

Phase 2 dynamically selects semantically related demonstrations without changing model parameters.

## LoRA / PEFT

Phase 3 introduces parameter-efficient fine-tuning.

Evidence includes:

- explicit LoRA hyperparameter configuration
- configuration validation
- optional PEFT integration
- supervised instruction/response record formatting
- adapter training pipeline
- base-vs-adapter evaluation
- trainable-parameter analysis

## Parameter efficiency

The project records:

- total parameters
- trainable parameters
- trainable percentage

This demonstrates why LoRA is different from full fine-tuning.

## Experimental control

The base and adapted model are evaluated using the same target examples and prompt format so adapter impact can be isolated.

## Reproducibility

Heavy training dependencies are optional and excluded from normal CI.

CI instead verifies the deterministic components required for a correct fine-tuning pipeline.

## Responsible design

The local fixture dataset demonstrates the engineering pipeline but is explicitly documented as too small for claims about general LLM performance.

## Evidence still to add

- robustness/error analysis
- experiment tracking
- model card
- training-data governance
- adapter/model versioning
- monitoring and rollback
- deployment/MLOps design
- final reflection
