import llm_mcq_bias as lmb


def test_text_generation_gpt_4o_mini():
    #
    # Givens
    #

    # Prompt
    prompt = "What is the capital of Massachusetts? Answer in one word."

    # I selected gpt 4o mini model
    generator = lmb.models.gpt_4o_mini

    # I limited output to single most likely token
    options = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
    }

    #
    # Whens
    #

    # I generate answer
    answer = generator(prompt=prompt, options=options)

    #
    # Thens
    #

    # The answer should be "Boston"
    assert answer == "Boston"
