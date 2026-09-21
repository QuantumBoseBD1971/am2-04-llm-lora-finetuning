"""LoRA configuration and parameter-efficiency utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q", "v")
    bias: str = "none"


@dataclass(frozen=True)
class ParameterStats:
    total_parameters: int
    trainable_parameters: int

    @property
    def trainable_percentage(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return 100.0 * self.trainable_parameters / self.total_parameters


def validate_lora_settings(settings: LoraSettings) -> None:
    """Validate core LoRA hyperparameters."""
    if settings.rank <= 0:
        raise ValueError("LoRA rank must be positive.")
    if settings.alpha <= 0:
        raise ValueError("LoRA alpha must be positive.")
    if not 0 <= settings.dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1).")
    if not settings.target_modules:
        raise ValueError("At least one target module is required.")


def parameter_stats(model: Any) -> ParameterStats:
    """Count total and trainable parameters without importing PyTorch."""
    total = 0
    trainable = 0

    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if bool(parameter.requires_grad):
            trainable += count

    return ParameterStats(
        total_parameters=total,
        trainable_parameters=trainable,
    )


def build_peft_config(settings: LoraSettings):
    """Build a PEFT LoRA configuration lazily."""
    validate_lora_settings(settings)

    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError(
            'LoRA support is optional. Install with: pip install -e ".[finetune]"'
        ) from exc

    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias=settings.bias,
    )
