"""Optional PEFT/LoRA fine-tuning utilities."""

from __future__ import annotations

from pathlib import Path

from modern_llm_benchmark.lora import LoraSettings, build_peft_config, parameter_stats


def train_lora_adapter(
    training_records: list[dict[str, str]],
    output_dir: Path,
    base_model_name: str = "google/flan-t5-small",
    settings: LoraSettings | None = None,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
) -> dict[str, float | int | str]:
    """Fine-tune a sequence-to-sequence model with LoRA.

    Heavy dependencies are imported lazily so CI remains lightweight.
    """
    try:
        from datasets import Dataset
        from peft import get_peft_model
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            'Fine-tuning support is optional. Install with: pip install -e ".[finetune]"'
        ) from exc

    if not training_records:
        raise ValueError("At least one training record is required.")

    settings = settings or LoraSettings()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = get_peft_model(base_model, build_peft_config(settings))
    stats = parameter_stats(model)

    dataset = Dataset.from_list(training_records)

    def tokenise(batch):
        model_inputs = tokenizer(
            batch["source"],
            truncation=True,
            max_length=256,
        )
        labels = tokenizer(
            text_target=batch["target"],
            truncation=True,
            max_length=64,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenised = dataset.map(
        tokenise,
        batched=True,
        remove_columns=dataset.column_names,
    )

    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=tokenised,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    train_result = trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    return {
        "base_model": base_model_name,
        "trainable_parameters": stats.trainable_parameters,
        "total_parameters": stats.total_parameters,
        "trainable_percentage": stats.trainable_percentage,
        "train_loss": float(train_result.training_loss),
    }
