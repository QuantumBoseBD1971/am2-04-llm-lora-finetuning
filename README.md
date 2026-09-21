# AM2-04 — LLM Adaptation and LoRA Fine-Tuning Benchmark

A comparative open-source language-model adaptation project.

The project follows:

**zero-shot → fixed few-shot → retrieval-assisted few-shot → LoRA fine-tuning**

## Research question

> When is prompt engineering sufficient, when does retrieval help, and when does parameter-efficient fine-tuning provide measurable additional value?

## Implemented strategies

- zero-shot prompting
- fixed few-shot prompting
- semantic retrieval of demonstrations
- retrieval-assisted few-shot prompting
- LoRA / PEFT adapter configuration
- optional supervised adapter training
- base-vs-adapter evaluation
- trainable-parameter analysis

## LoRA design

Default educational configuration:

- rank: 8
- alpha: 16
- dropout: 0.05
- target modules: `q`, `v`

The project reports both absolute trainable parameter count and trainable percentage.

## Quick start

CI-safe components:

```bash
pip install -e ".[dev]"
python scripts/run_prompting_baseline.py
python scripts/run_retrieval_prompting.py
pytest
```

Optional LoRA experiment:

```bash
pip install -e ".[finetune]"
python scripts/train_lora.py
python scripts/evaluate_lora.py
```

## Development status

- **Phase 1 — complete:** zero-shot and fixed few-shot prompting.
- **Phase 2 — complete:** semantic retrieval-assisted prompting.
- **Phase 3 — in progress:** LoRA / PEFT adaptation and evaluation.
- **Phase 4 — planned:** robustness, tracking, governance, model card and MLOps.

## Responsible use

The bundled fixture exists to demonstrate the adaptation pipeline and is not large enough to support claims about general model improvement. Fine-tuning data must be licensed, reviewed and versioned for real use.

## Licence

Code: MIT. External models/datasets retain their original licences.
