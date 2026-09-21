# AM2-04 — LLM Adaptation and LoRA Fine-Tuning Benchmark

A comparative open-source language-model adaptation project:

**zero-shot → fixed few-shot → retrieval-assisted few-shot → LoRA fine-tuning**

## Research question

> When is prompt engineering sufficient, when does retrieval help, and when does parameter-efficient fine-tuning provide measurable additional value?

## Implemented evidence

- zero-shot prompting
- fixed few-shot prompting
- semantic demonstration retrieval
- retrieval-assisted prompting
- LoRA / PEFT configuration
- optional adapter training
- base-vs-adapter evaluation
- trainable-parameter analysis
- robustness/error analysis
- experiment tracking
- training-data governance
- model card
- deployment/MLOps design

## Quick start

CI-safe workflow:

```bash
pip install -e ".[dev]"
python scripts/run_prompting_baseline.py
python scripts/run_retrieval_prompting.py
python scripts/finalise_project.py
pytest
```

Optional real-model experiments:

```bash
pip install -e ".[retrieval]"
pip install -e ".[finetune]"
python scripts/train_lora.py
python scripts/evaluate_lora.py
python scripts/finalise_project.py
```

## Evaluation

- exact match
- token F1
- retrieved example ids
- trainable parameter count
- trainable percentage
- per-example error analysis

## Development status

- **Phase 1 — complete:** prompting baselines
- **Phase 2 — complete:** retrieval-assisted prompting
- **Phase 3 — complete:** LoRA / PEFT adaptation
- **Phase 4 — complete:** robustness, tracking, governance, model card and MLOps

## Documentation

See `docs/` for methodology, governance, model card, deployment/MLOps, final summary, reflection and AM2 evidence.

## Responsible use

The bundled fixture demonstrates the engineering workflow but is not sufficient to establish general model improvement or production safety.

## Licence

Code: MIT. External models/datasets retain their original licences.
