"""Generation using a saved LoRA adapter."""

from __future__ import annotations


class PeftGenerator:
    """Load a base seq2seq model plus a PEFT adapter for inference."""

    def __init__(
        self,
        adapter_path: str,
        base_model_name: str = "google/flan-t5-small",
        max_new_tokens: int = 32,
    ) -> None:
        try:
            from peft import PeftModel
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                'Adapter inference is optional. Install with: pip install -e ".[finetune]"'
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, adapter_path)

        self.pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        output = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )[0]["generated_text"]
        return str(output).strip()
