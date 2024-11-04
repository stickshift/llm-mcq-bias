import json
import logging
from pathlib import Path
from time import perf_counter_ns as timer

import llm_mcq_bias as lmb


logger = logging.getLogger(__name__)


def test_ollama_323b(datasets_path: Path):

    #
    # Givens
    #

    # We limit generated tokens to mitigate extra time consumed by invalid responses
    options = {
        "num_predict": 10,
    }

    # I loaded example questions
    example_questions = lmb.datasets.mmlu.load_dataset(datasets_path, segment="dev")

    # I loaded test questions
    test_questions = lmb.datasets.mmlu.load_dataset(datasets_path, segment="test")

    # I selected a subset of test questions from all categories
    n = 50
    test_questions = test_questions.sample(n=n)

    # I warmed up model
    lmb.models.llama_323b(prompt="What is the capital of Massachusetts? Answer in one word.", options=options)

    #
    # Whens
    #

    # I start timer
    start_time = timer()

    # I generate answers for each question
    correct, errors = 0, 0
    for index in test_questions.index:
        mcq = test_questions.loc[index]
        prompt = lmb.datasets.mmlu.generate_prompt(example_questions, test_questions, index)

        # Generate answer
        response = lmb.models.llama_323b(prompt=prompt, options=options)

        try:
            # Parse answer
            answer = json.loads(response)["answer"]
            if answer not in {"A", "B", "C", "D"}:
                raise ValueError(f"Invalid answer: {answer}")

            # Check answer
            if answer == mcq.answer:
                correct += 1

        except Exception as e:
            logger.error(f"Error: {e}")
            errors += 1

    # I stop timer
    duration = timer() - start_time

    # I calculate metrics
    accuracy = correct / (n - errors)
    error_rate = errors / n
    request_rate = 1000000000 * n / duration

    logger.info(f"Metrics: total {n}, correct {correct}, errors {errors}, duration {duration}, accuracy {accuracy}, error_rate {error_rate}, request_rate {request_rate}")

    #
    # Thens
    #

    # accuracy should be reasonable: > than 50%
    assert accuracy > 0.4

    # error_rate should be reasonable: less than 10%
    assert error_rate < 0.10

    # request_rate should be reasonable: more than 1 request per second
    assert request_rate > 1.0
