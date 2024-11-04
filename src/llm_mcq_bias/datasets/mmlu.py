from pathlib import Path

import pandas as pd
from pandas import DataFrame

__all__ = [
    "load_dataset",
    "swap_options",
    "generate_prompt",
]

_options = {"A", "B", "C", "D"}


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
    if option not in _options:
        raise ValueError(f"Invalid option: {option}")

    rows = []
    for _, row in questions.iterrows():
        value = row[option]
        row[option] = row[row.answer]
        row[row.answer] = value
        row.answer = option
        rows.append(row)

    return DataFrame(rows)


def generate_prompt(example_questions, test_questions, index):
    """Generate prompt for specified question."""
    mcq = test_questions.loc[index]

    # Select examples for category
    selected_examples = example_questions[example_questions.category == mcq.category]

    # Sanity check
    assert len(selected_examples) == 5

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