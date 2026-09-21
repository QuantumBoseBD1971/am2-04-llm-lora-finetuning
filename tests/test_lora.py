from modern_llm_benchmark.fixtures import benchmark_examples
from modern_llm_benchmark.lora import (
    LoraSettings,
    parameter_stats,
    validate_lora_settings,
)
from modern_llm_benchmark.training_data import build_training_records


class FakeParameter:
    def __init__(self, count: int, requires_grad: bool):
        self.count = count
        self.requires_grad = requires_grad

    def numel(self):
        return self.count


class FakeModel:
    def parameters(self):
        return [
            FakeParameter(900, False),
            FakeParameter(100, True),
        ]


def test_default_lora_settings_are_valid() -> None:
    validate_lora_settings(LoraSettings())


def test_invalid_lora_rank_raises() -> None:
    try:
        validate_lora_settings(LoraSettings(rank=0))
    except ValueError as exc:
        assert "rank" in str(exc).lower()
    else:
        raise AssertionError("Expected invalid rank to raise ValueError.")


def test_parameter_stats_reports_trainable_fraction() -> None:
    stats = parameter_stats(FakeModel())
    assert stats.total_parameters == 1000
    assert stats.trainable_parameters == 100
    assert stats.trainable_percentage == 10.0


def test_training_records_preserve_instruction_and_answer() -> None:
    example = benchmark_examples()[0]
    record = build_training_records([example])[0]

    assert example.instruction in record["source"]
    assert record["target"] == example.expected
