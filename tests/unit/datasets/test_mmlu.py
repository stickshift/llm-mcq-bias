from pathlib import Path
from textwrap import dedent

import numpy as np
from pandas.testing import assert_frame_equal

import llm_mcq_bias as lmb


def test_load_example_questions(datasets_path: Path):
    #
    # Whens
    #

    # I load example questions
    questions = lmb.datasets.mmlu.load_dataset(datasets_path, segment="dev")

    #
    # Thens
    #

    # There should be 57 categories
    assert len(questions.category.unique()) == 57

    # Each category should have 5 questions
    assert np.all(questions.category.value_counts() == 5)


def test_load_test_questions(datasets_path: Path):
    #
    # Whens
    #

    # I load test questions
    questions = lmb.datasets.mmlu.load_dataset(datasets_path, segment="test")

    #
    # Thens
    #

    # indices should be unique
    assert questions.index.is_unique

    # questions columns should be "category", "question", "A", "B", "C", "D", "answer"
    assert set(questions.columns) == {
        "category",
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
    }

    # questions should include questions from "elementary_mathematics" category
    assert len(questions.query("category == 'elementary mathematics'")) > 0

    # questions should have mapped the word None to a string (instead of a NaN)
    assert questions.loc[1973]["D"] == "None"


def test_generate_prompt(datasets_path: Path):
    #
    # Givens
    #

    # I loaded example questions
    example_questions = lmb.datasets.mmlu.load_dataset(datasets_path, segment="dev")

    # I loaded test questions
    test_questions = lmb.datasets.mmlu.load_dataset(datasets_path, segment="test")

    # I selected question 11776
    mcq = test_questions.loc[11776]

    #
    # Whens
    #

    # I generate prompt for question 11776
    prompt = lmb.datasets.mmlu.generate_prompt(example_questions, mcq)

    #
    # Thens
    #

    # Prompt should be populated
    expected = dedent(
        """
        You are a robot that only outputs JSON. You reply in JSON format with the field 'answer'. For example, the following are multiple choice questions about elementary mathematics.
        
        Example Question: The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?
        
        A) 5 thousands
        B) 5 hundreds
        C) 5 tens
        D) 5 ones
        
        Example Answer: {"answer": "A"}
        
        Example Question: Olivia used the rule "Add 11" to create the number pattern shown below. 10, 21, 32, 43, 54 Which statement about the number pattern is true?
        
        A) The 10th number in the pattern will be an even number.
        B) The number pattern will never have two even numbers next to each other.
        C) The next two numbers in the pattern will be an even number then an odd number.
        D) If the number pattern started with an odd number then the pattern would have only odd numbers in it.
        
        Example Answer: {"answer": "B"}
        
        Example Question: A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?
        
        A) Add 5 to 30 to find 35 teams.
        B) Divide 30 by 5 to find 6 teams.
        C) Multiply 30 and 5 to find 150 teams.
        D) Subtract 5 from 30 to find 25 teams.
        
        Example Answer: {"answer": "B"}
        
        Example Question: A store sells 107 different colors of paint. They have 25 cans of each color in storage. The number of cans of paint the store has in storage can be found using the expression below. 107 × 25. How many cans of paint does the store have in storage?
        
        A) 749
        B) 2,675
        C) 2,945
        D) 4,250
        
        Example Answer: {"answer": "B"}
        
        Example Question: Which expression is equivalent to 5 x 9?
        
        A) (5 x 4) x (6 x 5)
        B) (5 x 5) + (5 x 4)
        C) (5 x 5) + (5 x 9)
        D) (5 x 9) x (6 x 9)
        
        Example Answer: {"answer": "B"}
        
        Given the examples above, your task is to answer the following question.
        
        Question: What is the value of p in 24 = 2p?
        
        A) p = 4
        B) p = 8
        C) p = 12
        D) p = 24
        
        Answer: """
    ).lstrip()
    assert prompt == expected


def test_golden_option(datasets_path: Path):
    #
    # Givens
    #

    # Option A
    golden_option = "A"

    # I loaded mmlu original dataset
    dataset1 = lmb.datasets.mmlu.load_dataset(datasets_path)

    #
    # Whens
    #

    # I load mmlu dataset with golden option
    dataset2 = lmb.datasets.mmlu.load_dataset(
        datasets_path, golden_option=golden_option
    )

    #
    # Thens
    #

    # Answers should be in target option
    for index, row1 in dataset1.iterrows():
        row2 = dataset2.loc[index]
        assert row2[golden_option] == row1[row1.answer], f"{index}, {row1.question}"
        assert row2[row1.answer] == row1[golden_option]
        assert row2.answer == "A"


def test_idempotent_datasets(datasets_path: Path):
    #
    # Whens
    #

    # I load mmlu dataset twice
    dataset1 = lmb.datasets.mmlu.load_dataset(datasets_path)
    dataset2 = lmb.datasets.mmlu.load_dataset(datasets_path)

    #
    # Thens
    #

    # datasets should be equal
    assert_frame_equal(dataset1, dataset2)
