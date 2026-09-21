# AM2-04 — LLM Adaptation and LoRA Fine-Tuning Benchmark

A comparative open-source language-model adaptation project.

The project follows:

**zero-shot → few-shot → retrieval-assisted prompting → LoRA fine-tuning**

The aim is to compare increasingly specialised adaptation strategies under one evaluation framework rather than assuming fine-tuning is automatically the best option.

## Research question

> When is prompt engineering sufficient, when does retrieval help, and when does parameter-efficient fine-tuning provide measurable additional value?

## Phase 1 — prompting baseline

The first phase establishes:

- a deterministic instruction/response benchmark fixture
- prompt formatting
- zero-shot prompting
- few-shot prompting
- exact-match and token-F1 evaluation
- CI-safe mock generator
- optional Hugging Face generator
- tests and GitHub Actions CI

The benchmark is intentionally small in CI. Later phases can use a larger public instruction dataset.

## Later phases

### Phase 2 — retrieval-assisted prompting
- example retrieval
- semantic similarity
- dynamic in-context examples
- zero-shot vs fixed few-shot vs retrieved few-shot

### Phase 3 — LoRA / PEFT
- LoRA adapter configuration
- supervised fine-tuning
- trainable-parameter comparison
- base vs adapted model evaluation

### Phase 4 — productionisation
- robustness/error analysis
- experiment tracking
- model card
- safety and data-governance considerations
- deployment/MLOps design
- final AM2 evidence synthesis

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts/run_prompting_baseline.py
pytest
```

Optional local open-source model:

```bash
pip install -e ".[llm]"
```

## Evaluation

Initial metrics:

- exact match
- token-level F1
- per-example error table

Later phases add adaptation cost, trainable-parameter count, runtime and robustness.

## Responsible use

This repository is educational. Fine-tuning can reproduce biases or errors present in training data, and benchmark performance does not establish suitability for high-stakes deployment.

## Licence

Code: MIT. External models/datasets retain their original licences.
