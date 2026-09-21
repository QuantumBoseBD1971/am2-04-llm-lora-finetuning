# Phase 3 — LoRA / Parameter-Efficient Fine-Tuning

## Objective

Phase 3 introduces parameter-efficient fine-tuning and asks whether adapting a small subset of model parameters adds measurable value beyond:

- zero-shot prompting
- fixed few-shot prompting
- retrieval-assisted few-shot prompting

## What LoRA changes

LoRA (Low-Rank Adaptation) keeps the pretrained model weights frozen and injects trainable low-rank matrices into selected linear projections.

A conventional weight update can be represented as:

[
W' = W + \Delta W
]

LoRA constrains the update:

[
\Delta W = BA
]

where the rank of (A) and (B) is much smaller than the full weight matrix.

This substantially reduces the number of trainable parameters.

## Configuration

The default educational configuration is:

- rank (r = 8)
- alpha = 16
- dropout = 0.05
- target modules = `q`, `v`
- sequence-to-sequence language-model task

The configuration is explicit and validated before a PEFT config is constructed.

## Parameter efficiency

The project reports:

- total model parameters
- trainable model parameters
- percentage of parameters being trained

This is important because the point of LoRA is not merely model quality; it is **adaptation efficiency**.

## Training data

The local demonstration and benchmark examples can be converted to supervised source/target records.

This small fixture is for pipeline demonstration only. It is not sufficient for a meaningful production fine-tune.

A substantive experiment should use a larger licensed instruction dataset with a strict train/evaluation split.

## Optional heavy dependencies

Fine-tuning requires:

- Transformers
- PyTorch
- PEFT
- Datasets
- Accelerate

These are isolated behind:

```bash
pip install -e ".[finetune]"
```

Normal CI validates configuration, parameter counting and data formatting without downloading a model.

## Base vs adapter evaluation

The evaluation script loads:

1. the untouched base model
2. the same base model with the saved LoRA adapter

Both are evaluated with the same zero-shot prompt and the same benchmark examples.

This isolates the effect of the adapter itself.

## Artefact strategy

Adapters are written to `artifacts/` and are intentionally not committed.

A production registry should retain:

- base model id/revision
- adapter version
- LoRA configuration
- training-data version
- metrics
- dependency versions
- code commit

## Next phase

Phase 4 will add robustness/error analysis, experiment tracking, model card, governance and deployment/MLOps evidence.
