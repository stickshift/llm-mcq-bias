from functools import partial

import pytest

import llm_mcq_bias as lmb

models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-9b-it",
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
        "do_sample": False,
        "max_new_tokens": 1,
    }

    # I packaged model and options into ollama generator
    generator = partial(lmb.models.hf, model=model, options=options)

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
