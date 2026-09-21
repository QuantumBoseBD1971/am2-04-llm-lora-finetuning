# Training Data Governance

## Why governance matters

LoRA fine-tuning changes model behaviour using project-specific examples. That makes the provenance and permissions of the training data part of the model lifecycle.

## Minimum governance record

For every fine-tuning dataset, record:

- dataset name and version
- source URL or internal system
- licence / usage rights
- collection date
- intended purpose
- train / validation / test split definition
- personally identifiable or sensitive fields
- data-cleaning transformations
- deduplication rules
- known biases or coverage gaps
- approver / owner

## Data leakage

Evaluation examples must not be duplicated into the training set.

Near-duplicate instructions should also be checked where possible because instruction paraphrases can create hidden contamination.

## Retention

Training records and adapters should follow the retention policy of the underlying data source.

If a record must be removed for legal or governance reasons, the impact on any adapter trained from that record should be assessed.

## This repository

The bundled fixture is synthetic and intentionally tiny. It exists to demonstrate the engineering workflow and must not be presented as evidence of general model quality.
