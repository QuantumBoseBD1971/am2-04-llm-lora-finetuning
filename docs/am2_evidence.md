# AM2 Evidence Notes

This document evolves with the LLM adaptation project.

## Problem framing

The project asks when prompting is sufficient and when model adaptation is justified.

## Prompt engineering

Phase 1 implements:

- zero-shot prompting
- fixed few-shot prompting
- reusable prompt templates

## Evaluation

Responses are evaluated using:

- exact match
- token-level F1
- per-example outputs

## Software engineering

The repository uses:

- packaged Python modules
- typed benchmark structures
- generator interfaces
- unit tests
- GitHub Actions CI

## Reproducibility

The core benchmark is deterministic and CI-safe.

A Hugging Face generator is available behind an optional dependency group so model downloads remain separate from the core test path.

## Evidence still to add

- semantic retrieval of demonstrations
- retrieval-assisted prompting
- public instruction dataset
- LoRA adapter configuration
- parameter-efficient fine-tuning
- base-vs-adapter evaluation
- trainable-parameter analysis
- robustness/error analysis
- experiment tracking
- model card
- deployment/MLOps design
- final reflection
