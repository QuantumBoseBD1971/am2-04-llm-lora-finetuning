# Model Card

## Scope

This repository compares four adaptation strategies:

- zero-shot prompting
- fixed few-shot prompting
- retrieval-assisted few-shot prompting
- LoRA / PEFT fine-tuning

## Intended use

Educational benchmarking and AM2 portfolio evidence for open-source LLM adaptation techniques.

## Out-of-scope use

The bundled benchmark and tiny training fixture are not sufficient for production or high-stakes decision making.

## Base model

The optional implementation defaults to `google/flan-t5-small`.

Any real experiment should record the exact upstream model revision, tokenizer revision and licence.

## Adapter

The default LoRA configuration is:

- rank: 8
- alpha: 16
- dropout: 0.05
- target modules: `q`, `v`

## Evaluation

The repository uses:

- exact match
- token F1
- per-example outputs
- retrieval-selected example ids
- trainable-parameter count
- trainable-parameter percentage
- error taxonomy

## Risks and limitations

- training examples may encode bias or incorrect behaviour
- tiny evaluation sets can produce unstable conclusions
- data leakage can exaggerate fine-tuning gains
- adapter performance may regress on unrelated tasks
- upstream model changes can invalidate previous comparisons
- LoRA reduces trainable parameters but does not remove inference cost of the base model

## Human oversight

A real deployment should retain human review, data-governance approval, model/version traceability and rollback capability.
