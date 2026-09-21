# Final Reflection

## What the project demonstrates

The project deliberately starts with the cheapest adaptation mechanism and only adds complexity when justified:

1. zero-shot prompting
2. fixed few-shot prompting
3. retrieval-assisted few-shot prompting
4. LoRA fine-tuning

This progression makes it possible to discuss adaptation as an engineering trade-off rather than treating fine-tuning as the default.

## Key technical lesson

Prompting changes the input context.

Retrieval changes which examples are included in that context.

LoRA changes a small trainable subset of model behaviour while retaining the frozen base model.

These are fundamentally different mechanisms and should be benchmarked separately.

## Parameter efficiency

LoRA's value is not only predictive quality. Its purpose is to reduce training cost and artefact size relative to full fine-tuning.

Therefore trainable-parameter percentage is a first-class result.

## Production lesson

A tuned adapter creates lifecycle obligations that prompting alone does not:

- training-data governance
- adapter versioning
- reproducibility
- monitoring
- regression testing
- rollback

## What I would improve with more time

I would add:

- a larger licensed public instruction dataset
- strict train/validation/test split
- multiple open-source base models
- QLoRA
- held-out control tasks for catastrophic-regression checks
- latency and GPU-memory benchmarking
- automated hyperparameter search
- human preference evaluation
- safety regression suite

## AM2 relevance

The repository demonstrates prompt engineering, semantic retrieval, parameter-efficient fine-tuning, evaluation, reproducibility, governance and MLOps within one coherent comparison.
