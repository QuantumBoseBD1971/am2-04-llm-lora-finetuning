# Phase 1 — Prompting Baselines

## Objective

Establish a common evaluation harness before introducing retrieval or fine-tuning.

The first comparison is:

```text
zero-shot
   vs
few-shot
```

## Why start here

Fine-tuning should not be the default response to every language-model task.

Prompting is cheaper, faster to iterate and does not create a new model artefact.

A valid LoRA experiment therefore needs to show whether adaptation provides value beyond strong prompting baselines.

## Benchmark fixture

CI uses a tiny deterministic instruction/response dataset.

The purpose is not to claim meaningful LLM quality from four examples. The fixture exists to test:

- prompt formatting
- evaluation plumbing
- result persistence
- CI behaviour

Larger public datasets are introduced in later phases.

## Zero-shot

The model receives only:

- a system instruction
- the target instruction

## Few-shot

The model additionally receives demonstration instruction/answer pairs.

This tests in-context learning without changing model parameters.

## Metrics

### Exact match
Useful for tasks where the expected output is short and constrained.

### Token F1
Provides partial credit where wording differs slightly.

## Dependency isolation

The optional Hugging Face generator is installed through:

```bash
pip install -e ".[llm]"
```

Normal CI uses a deterministic generator so model weights are not downloaded for every pull request.

## Next phase

Phase 2 will replace fixed demonstrations with retrieved examples and compare:

- zero-shot
- fixed few-shot
- retrieval-assisted few-shot
