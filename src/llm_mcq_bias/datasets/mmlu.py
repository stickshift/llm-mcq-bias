from enum import StrEnum
import json
import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

__all__ = [
    "OPTIONS",
    "load_dataset",
    "swap_options",
    "normalize_question_answers",
    "normalize_example_answers",
    "generate_prompt",
    "Evaluation",
    "evaluate_response",
]

OPTIONS = {"A", "B", "C", "D"}

logger = logging.getLogger(__name__)


def load_dataset(
    datasets_path: Path,
    segment: str | None = None,
    golden_option: str | None = None,
) -> DataFrame:
    """Load MMLU questions from segment."""

    # Defaults
    segment = "test" if segment is None else segment

    dataset = None
    column_names = ["question", "A", "B", "C", "D", "answer"]

    for path in datasets_path.glob(f"mmlu/{segment}/*.csv"):
        df = pd.read_csv(path, names=column_names)

        # Infer category from file name: x_y_z_test.csv -> x y z
        df["category"] = " ".join(path.stem.split("_")[0:-1])

        # Append
        dataset = df if dataset is None else pd.concat([dataset, df], ignore_index=True)

    # Pandas parses the word "None" and a NaN. Replace these with explicit string "None"
    dataset = dataset.fillna("None")

    # Apply golden option if specified
    if golden_option is not None:
        dataset = swap_options(dataset, golden_option)

    return dataset


def swap_options(questions: DataFrame, option: str) -> DataFrame:
    # Validate options
    if option not in OPTIONS:
        raise ValueError(f"Invalid option: {option}")

    rows = []
    for _, row in questions.iterrows():
        value = row[option]
        row[option] = row[row.answer]
        row[row.answer] = value
        row.answer = option
        rows.append(row)

    return DataFrame(rows)


def normalize_question_answers(questions: DataFrame):
    """Evenly distribute question answers across options."""
    chunk_size = len(OPTIONS)

    # Select maximal subset of questions that is multiple of chunk size
    n_questions = chunk_size * (len(questions) // chunk_size)
    questions = questions.sample(n=n_questions)

    # Move 25% of answers to each option
    normalized = None
    segment_size = int(n_questions / chunk_size)
    for i, option in enumerate(OPTIONS):
        segment = swap_options(questions.iloc[i * segment_size:(i + 1) * segment_size], option)
        normalized = segment if normalized is None else pd.concat([normalized, segment])

    # Shuffle
    normalized = normalized.sample(frac=1).reset_index(drop=True)

    return normalized


def normalize_example_answers(examples: DataFrame):
    """Evenly distribute example answers across options."""
    categories = examples.category.unique()

    # Select 4 examples per category
    normalized = None
    for category in categories:

        # Select 4 examples
        selection = examples[examples.category == category].sample(n=4)

        # Move 25% of answers to each option
        segment_size = 1
        for i, option in enumerate(OPTIONS):
            segment = swap_options(selection.iloc[i * segment_size:(i + 1) * segment_size], option)
            normalized = segment if normalized is None else pd.concat([normalized, segment])

    # Shuffle
    normalized = normalized.sample(frac=1).reset_index(drop=True)

    return normalized


def generate_prompt(example_questions: DataFrame, mcq: Series):
    """Generate prompt for specified question."""

    # Select examples for category
    selected_examples = example_questions[example_questions.category == mcq.category]

    # Start with examples
    content = (
        f"You are a robot that only outputs JSON. "
        f"You reply in JSON format with the field 'answer'. "
        f"For example, the following are multiple choice questions about {mcq.category}.\n\n"
    )
    for _, row in selected_examples.iterrows():
        content += (
            f"Example Question: {row.question}\n"
            f"\n"
            f"A) {row.A}\n"
            f"B) {row.B}\n"
            f"C) {row.C}\n"
            f"D) {row.D}\n"
            f"\n"
            f"Example Answer: {{\"answer\": \"{row.answer}\"}}\n"
            f"\n"
        )

    # Pose question
    content += f"Given the examples above, your task is to answer the following question.\n\n"
    content += (
        f"Question: {mcq.question}\n"
        f"\n"
        f"A) {mcq.A}\n"
        f"B) {mcq.B}\n"
        f"C) {mcq.C}\n"
        f"D) {mcq.D}\n"
        f"\n"
        f"Answer: "
    )

    return content


class Evaluation(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ERROR = "error"


def evaluate_response(mcq: Series, response: str) -> Evaluation:
    """Evaluate response for specified question."""

    correct, errors = 0, 0

    try:
        # Strip text leading up to first { and after last }
        response = response[response.index("{"):response.rindex("}") + 1]

        # Parse answer
        answer = json.loads(response)["answer"]
        if answer not in OPTIONS:
            raise ValueError(f"Invalid answer: {answer}")

        # Check answer
        if answer == mcq.answer:
            return Evaluation.CORRECT

    except Exception as e:
        # logger.error(f"Error: {e}")
        return Evaluation.ERROR

    return Evaluation.INCORRECT
