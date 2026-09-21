# AM2 Evidence Notes

This document evolves with the LLM adaptation project.

## Problem framing

The project asks when prompting is sufficient and when retrieval or parameter-efficient adaptation is justified.

## Prompt engineering

Phase 1 implements zero-shot and fixed few-shot prompting.

## Retrieval-assisted prompting

Phase 2 introduces:

- pluggable embedding encoder
- semantic demonstration retrieval
- cosine-similarity ranking
- dynamic few-shot prompt construction
- explicit logging of retrieved example ids

The production sentence-transformer dependency is optional; deterministic fake embeddings support CI.

## Comparative evaluation

The same evaluation harness compares:

- zero-shot
- fixed few-shot
- retrieved few-shot

Metrics remain:

- exact match
- token-level F1
- per-example outputs

This isolates the effect of context-selection strategy before any model weights are changed.

## Software engineering

The retrieval component depends on a small encoder protocol, keeping it replaceable and testable.

## Reproducibility

Heavy model dependencies remain outside the normal CI path, while deterministic tests cover retrieval logic and prompt assembly.

## Evidence still to add

- LoRA adapter configuration
- parameter-efficient fine-tuning
- base-vs-adapter evaluation
- trainable-parameter analysis
- robustness/error analysis
- experiment tracking
- model card
- data-governance considerations
- deployment/MLOps design
- final reflection
