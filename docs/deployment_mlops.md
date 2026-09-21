# Deployment and MLOps Design

## Adaptation lifecycle

A production system should treat base model and adapter as separate versioned artefacts.

Record:

- base model id and revision
- tokenizer id and revision
- adapter version
- LoRA hyperparameters
- training-data version
- code commit
- dependency versions
- evaluation metrics
- hardware/runtime environment

## Immutable artefacts

Never overwrite an existing adapter.

Each training run should write a uniquely versioned adapter directory and immutable metadata.

A separate alias such as `production` or `champion` can point to the active adapter.

## Promotion criteria

Promotion should consider:

- target-task improvement
- regression on held-out control tasks
- trainable-parameter efficiency
- inference latency
- memory use
- safety / policy checks
- data-governance approval

## Monitoring

Monitor:

- input distribution changes
- output-length shifts
- exact-match / task metrics when labels arrive
- abstention / invalid-output rates
- latency
- memory
- adapter load failures
- base-model revision drift

## Rollback

Rollback should repoint the production alias to a previously validated adapter/base-model pair.

Historical artefacts and training metadata should remain available for audit.

## Prompting vs retrieval vs LoRA

These strategies have different operational costs:

- prompting changes no model artefacts
- retrieval adds an embedding/index dependency
- LoRA adds a trained adapter lifecycle

The deployment decision should therefore consider maintainability as well as benchmark accuracy.
