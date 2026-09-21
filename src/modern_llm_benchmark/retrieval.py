"""Semantic retrieval of in-context demonstration examples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from modern_llm_benchmark.types import Example


class Encoder(Protocol):
    """Minimal embedding interface."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into a two-dimensional array."""


class SentenceTransformerEncoder:
    """Optional sentence-transformer embedding backend."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                'Retrieval support is optional. Install with: pip install -e ".[retrieval]"'
            ) from exc

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=float)


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise embedding rows."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional.")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return values / norms


class ExampleRetriever:
    """Retrieve semantically similar demonstration examples."""

    def __init__(
        self,
        demonstrations: list[Example],
        encoder: Encoder,
    ) -> None:
        if not demonstrations:
            raise ValueError("At least one demonstration example is required.")

        self.demonstrations = demonstrations
        self.encoder = encoder
        self.embeddings = l2_normalise(
            encoder.encode([example.instruction for example in demonstrations])
        )

    def retrieve(self, instruction: str, top_k: int = 2) -> list[Example]:
        """Return the top-k most similar examples."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query = l2_normalise(self.encoder.encode([instruction]))[0]
        scores = self.embeddings @ query
        indices = np.argsort(-scores)[:top_k]

        return [self.demonstrations[int(index)] for index in indices]
