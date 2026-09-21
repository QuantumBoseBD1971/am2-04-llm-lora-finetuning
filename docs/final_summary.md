# Final Project Summary

## Research question

When is prompting sufficient, when does retrieval improve adaptation, and when does LoRA fine-tuning justify the additional training and lifecycle cost?

## Technical progression

```text
zero-shot
   ↓
fixed few-shot
   ↓
retrieval-assisted few-shot
   ↓
LoRA / PEFT adapter
```

## Evidence chain

The repository covers:

- prompt construction
- semantic example retrieval
- dynamic in-context learning
- LoRA configuration
- PEFT training pipeline
- base-vs-adapter evaluation
- trainable-parameter analysis
- robustness/error analysis
- experiment tracking
- data governance
- model card
- adapter versioning and rollback

## Interpretation principle

The most specialised approach is not automatically the preferred one.

A deployment choice should consider:

- task quality
- training cost
- inference cost
- maintenance burden
- data governance
- reproducibility
- safety
- rollback complexity

## Reproducibility

Core tests remain CI-safe and do not download model weights.

Heavy LLM, embedding and PEFT dependencies are optional and can be installed only for the relevant experiment.
