from functools import partial

import pytest

import llm_mcq_bias as lmb

models = [
    "gpt-4o-mini",
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
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
    }

    # I packaged model and options into openai generator
    generator = partial(lmb.models.openai, model=model, options=options)

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
