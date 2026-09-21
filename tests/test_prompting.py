from modern_llm_benchmark.fixtures import (
    benchmark_examples,
    demonstration_examples,
)
from modern_llm_benchmark.metrics import exact_match, token_f1
from modern_llm_benchmark.prompts import few_shot_prompt, zero_shot_prompt


def test_zero_shot_prompt_contains_instruction() -> None:
    example = benchmark_examples()[0]
    prompt = zero_shot_prompt(example)
    assert example.instruction in prompt
    assert "Answer:" in prompt


def test_few_shot_prompt_contains_demonstrations() -> None:
    example = benchmark_examples()[0]
    demonstrations = demonstration_examples()
    prompt = few_shot_prompt(example, demonstrations)

    assert demonstrations[0].instruction in prompt
    assert demonstrations[0].expected in prompt
    assert example.instruction in prompt


def test_exact_match_normalises_case_and_space() -> None:
    assert exact_match("Hello", " hello ") == 1.0


def test_token_f1_perfect_match() -> None:
    assert token_f1("battery storage", "battery storage") == 1.0
