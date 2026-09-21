# AM2-04 — LLM Adaptation and LoRA Fine-Tuning Benchmark

A comparative open-source language-model adaptation project.

The project follows:

**zero-shot → fixed few-shot → retrieval-assisted few-shot → LoRA fine-tuning**

## Research question

> When is prompt engineering sufficient, when does retrieval help, and when does parameter-efficient fine-tuning provide measurable additional value?

## Implemented strategies

### Zero-shot
Only the target instruction is supplied.

### Fixed few-shot
The same demonstration examples are supplied to every target.

### Retrieval-assisted few-shot
Semantically similar demonstrations are selected dynamically using vector similarity.

The optional production encoder uses Sentence Transformers; CI uses deterministic injected embeddings.

## Quick start

```bash
pip install -e ".[dev]"
python scripts/run_prompting_baseline.py
python scripts/run_retrieval_prompting.py
pytest
```

Optional semantic embedding backend:

```bash
pip install -e ".[retrieval]"
```

Optional local Hugging Face generator:

```bash
pip install -e ".[llm]"
```

## Evaluation

- exact match
- token F1
- retrieved example ids
- per-example outputs

## Development status

- **Phase 1 — complete:** zero-shot and fixed few-shot prompting.
- **Phase 2 — in progress:** semantic demonstration retrieval.
- **Phase 3 — planned:** LoRA / PEFT fine-tuning and base-vs-adapter evaluation.
- **Phase 4 — planned:** robustness, tracking, model card, governance and MLOps.

## Responsible use

Fine-tuning and retrieved demonstrations can reproduce errors or biases present in source data. Benchmark improvement does not establish high-stakes suitability.

## Licence

Code: MIT. External models/datasets retain their original licences.
