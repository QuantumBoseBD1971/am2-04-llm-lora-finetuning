# Phase 2 — Retrieval-Assisted Prompting

## Objective

Phase 2 asks whether **dynamic demonstration selection** improves in-context learning compared with a fixed few-shot prompt.

The comparison becomes:

```text
zero-shot
   vs
fixed few-shot
   vs
retrieved few-shot
```

## Why retrieve demonstrations

A fixed few-shot prompt gives every target instruction the same examples.

Retrieval-assisted prompting instead selects examples whose instructions are semantically similar to the current task.

For example, a target instruction about lowercasing text should ideally retrieve another text-transformation example rather than an arithmetic demonstration.

## Semantic retrieval

The optional production backend uses:

`sentence-transformers/all-MiniLM-L6-v2`

The encoder maps each demonstration instruction to a vector.

Cosine similarity is used to rank candidate demonstrations.

The implementation normalises vectors first, making cosine similarity equivalent to a dot product.

## CI-safe design

Normal CI does not download embedding models.

Tests inject a deterministic fake encoder and validate:

- vector normalisation
- retrieval ordering
- prompt construction
- selected demonstration ids

This preserves testability without hiding the real production path.

## Evaluation

The same target examples and generation metrics are used across:

- zero-shot
- fixed few-shot
- retrieved few-shot

The result table also records which example ids were retrieved for each target.

This provides evidence for explaining *why* a retrieval-assisted prompt changed model behaviour.

## Interpretation

Retrieval-assisted prompting changes the **context** but not the underlying model parameters.

That distinction matters when comparing it with LoRA in Phase 3:

- retrieval adapts the prompt per request
- LoRA adapts model behaviour through trainable parameters

## Next phase

Phase 3 introduces parameter-efficient fine-tuning and compares the base model with a LoRA-adapted version.
