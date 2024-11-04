import llm_mcq_bias as lmb


def test_text_generation_llama32():
    #
    # Givens
    #

    # I limited predicted tokens to get consistent results
    options = {
        "top_k": 1,
        "num_predict": 1,
    }

    # Prompt
    prompt = "What is the capital of Massachusetts? Answer in one word."

    #
    # Whens
    #

    # I generate 1 token answer
    answer = lmb.models.llama_323b(prompt=prompt, options=options)

    #
    # Thens
    #

    # The answer should be "Boston"
    assert answer == "Boston"
