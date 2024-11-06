from functools import partial

import pytest

import llm_mcq_bias as lmb

models = [
    "llama3.2:3b",
    "gemma2:9b",
]


@pytest.mark.parametrize("model", models)
def test_text_generation(model: str):
    #
    # Givens
    #

    # Massachusetts prompt
    prompt = "What is the capital of Massachusetts? Answer in one word."

    # I limited output to single most likely token
    options = {
        "top_k": 1,
        "num_predict": 1,
    }

    # I packaged model and options into ollama generator
    generator = partial(lmb.models.ollama, model=model, options=options)

    #
    # Whens
    #

    # I generate answer
    answer = generator(prompt=prompt)

    #
    # Thens
    #

    # The answer should be "Boston"
    assert answer == "Boston"
