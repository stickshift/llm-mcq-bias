from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from time import perf_counter_ns as timer

import llm_mcq_bias as lmb
from llm_mcq_bias.datasets.mmlu import Evaluation


logger = logging.getLogger(__name__)


def test_gpt_4o_mini(datasets_path: Path):
    #
    # Givens
    #

    # I loaded example questions
    examples = lmb.datasets.mmlu.load_dataset(datasets_path, segment="dev")

    # I loaded test questions
    questions = lmb.datasets.mmlu.load_dataset(datasets_path, segment="test")

    # I selected gpt 4o mini model
    generator = lmb.models.gpt_4o_mini

    # We limit generated tokens to mitigate extra time consumed by invalid responses
    options = {
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 1.0,
    }

    # n_questions = 20
    n_questions = 20

    #
    # Whens
    #

    # I initialize thread pool
    executor = ThreadPoolExecutor()

    # I select a subset of questions
    questions = questions.sample(n=n_questions)

    # I start timer
    start_time = timer()

    # I generate answers for each question in parallel
    def process_mcq(mcq):
        # Generate prompt
        prompt = lmb.datasets.mmlu.generate_prompt(examples, mcq)

        # Generate answer
        response = generator(prompt=prompt, options=options)

        # Evaluate response
        evaluation = lmb.datasets.mmlu.evaluate_response(mcq, response)

        return evaluation

    futures = [executor.submit(process_mcq, mcq) for _, mcq in questions.iterrows()]

    # I collect results
    correct, errors = 0, 0
    for future in as_completed(futures):
        evaluation = future.result()
        if evaluation is Evaluation.CORRECT:
            correct += 1
        elif evaluation is Evaluation.ERROR:
            errors += 1

    # I stop timer
    duration = timer() - start_time

    # I calculate metrics
    accuracy = correct / (n_questions - errors)
    error_rate = errors / n_questions
    rps = 1000000000 * n_questions / duration

    logger.info(
        f"Metrics: total {n_questions}, correct {correct}, errors {errors}, duration {duration}, accuracy {accuracy}, error_rate {error_rate}, rps {rps}"
    )

    #
    # Thens
    #

    # accuracy should be reasonable: > than 50%
    assert accuracy > 0.4

    # error_rate should be reasonable: less than 10%
    assert error_rate < 0.10

    # rps should be reasonable: more than 1 request per second
    assert rps > 1.0