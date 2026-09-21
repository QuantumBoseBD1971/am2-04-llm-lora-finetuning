import numpy as np

from modern_llm_benchmark.fixtures import benchmark_examples, demonstration_examples
from modern_llm_benchmark.retrieval import ExampleRetriever, l2_normalise
from modern_llm_benchmark.retrieved_prompting import retrieval_assisted_prompt


class FakeEncoder:
    def encode(self, texts):
        mapping = {
            "Return the opposite of 'hot'.": [1.0, 0.0],
            "Convert 'WORLD' to lowercase.": [0.0, 1.0],
            "Return only the number: one plus one.": [0.5, 0.5],
            "Return the opposite of 'increase'.": [1.0, 0.1],
        }
        return np.asarray([mapping[text] for text in texts], dtype=float)


def test_l2_normalise() -> None:
    values = l2_normalise(np.array([[3.0, 4.0]]))
    assert np.allclose(values, np.array([[0.6, 0.8]]))


def test_retriever_selects_semantically_similar_demo() -> None:
    retriever = ExampleRetriever(demonstration_examples(), FakeEncoder())
    example = benchmark_examples()[0]

    retrieved = retriever.retrieve(example.instruction, top_k=1)

    assert retrieved[0].example_id == "d1"


def test_retrieval_assisted_prompt_contains_selected_demo() -> None:
    retriever = ExampleRetriever(demonstration_examples(), FakeEncoder())
    example = benchmark_examples()[0]

    prompt, demos = retrieval_assisted_prompt(example, retriever, top_k=1)

    assert demos[0].instruction in prompt
    assert example.instruction in prompt
