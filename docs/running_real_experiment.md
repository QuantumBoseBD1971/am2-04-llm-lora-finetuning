# Running the real LLM adaptation experiment

The manual **Run real LLM adaptation experiment** workflow executes Project 04 and stores compact evidence.

## Standard run

With **Train and evaluate the heavier LoRA adapter** disabled, the workflow runs:

1. deterministic prompt benchmark
2. deterministic retrieval-assisted benchmark
3. real Hugging Face generation with google/flan-t5-small
4. real SentenceTransformer demonstration retrieval
5. evidence synthesis

This is the recommended first run.

## LoRA run

Enable the LoRA option to additionally:

1. install PEFT / Datasets / Accelerate
2. fine-tune a LoRA adapter on the local instructional fixture
3. save training metrics
4. compare the base model against the adapted model
5. retain only compact metrics and outputs in Git

The adapter weights remain excluded from version control.

## Evidence

The full result set is uploaded as an Actions artifact for 90 days. A compact cumulative evidence directory is committed when the evidence option is enabled.
