"""Generation interfaces and optional Hugging Face backend."""

from __future__ import annotations

from typing import Protocol


class Generator(Protocol):
    """Minimal text-generation contract."""

    def generate(self, prompt: str) -> str:
        """Generate one response."""


class TransformersGenerator:
    """Optional Hugging Face text2text generator."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        max_new_tokens: int = 32,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                'LLM support is optional. Install with: pip install -e ".[llm]"'
            ) from exc

        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        output = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )[0]["generated_text"]
        return str(output).strip()
