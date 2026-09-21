"""Retrieval-assisted prompt construction."""

from __future__ import annotations

from modern_llm_benchmark.prompts import few_shot_prompt
from modern_llm_benchmark.retrieval import ExampleRetriever
from modern_llm_benchmark.types import Example


def retrieval_assisted_prompt(
    example: Example,
    retriever: ExampleRetriever,
    top_k: int = 2,
) -> tuple[str, list[Example]]:
    """Build a few-shot prompt from dynamically retrieved demonstrations."""
    demonstrations = retriever.retrieve(example.instruction, top_k=top_k)
    return few_shot_prompt(example, demonstrations), demonstrations
