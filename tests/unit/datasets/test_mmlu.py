from pathlib import Path
from textwrap import dedent

import numpy as np
from pandas.testing import assert_frame_equal

import llm_mcq_bias as lmb


def test_load_examples(datasets_path: Path):
    #
    # Givens
    #

    # Path to mmlu dataset
    dataset_path = datasets_path / "mmlu"

    #
    # Whens
    #

    # I load example questions
    questions = lmb.datasets.mmlu.load_dataset(dataset_path, segment="dev")

    #
    # Thens
    #

    # There should be 57 categories
    assert len(questions.category.unique()) == 57

    # Each category should have 5 questions
    assert np.all(questions.category.value_counts() == 5)


def test_load_questions(datasets_path: Path):
    #
    # Givens
    #

    # Path to mmlu dataset
    dataset_path = datasets_path / "mmlu"

    #
    # Whens
    #

    # I load test questions
    questions = lmb.datasets.mmlu.load_dataset(dataset_path, segment="test")

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

    # questions should not have any NaNs
    for column in questions.columns:
        assert questions[column].isna().sum() == 0


def test_debias_examples(datasets_path: Path):
    #
    # Givens
    #

    # Path to mmlu dataset
    dataset_path = datasets_path / "mmlu"

    #
    # Whens
    #

    # I load examples
    examples = lmb.datasets.mmlu.load_dataset(dataset_path, segment="dev")

    #
    # Thens
    #

    # Answers should NOT be evenly distributed
    for category in examples.category.unique():
        distribution = examples[examples.category == category].answer.value_counts()
        assert not (distribution == distribution.iloc[0]).all()

    #
    # Whens
    #

    # I debias examples
    examples = lmb.datasets.mmlu.debias_example_answers(examples)

    #
    # Thens
    #

    # Answers should be evenly distributed
    for category in examples.category.unique():
        distribution = examples[examples.category == category].answer.value_counts()
        assert (distribution == distribution.iloc[0]).all()


def test_debias_questions(datasets_path: Path):
    #
    # Givens
    #

    # Path to mmlu dataset
    dataset_path = datasets_path / "mmlu"

    #
    # Whens
    #

    # I load questions
    questions = lmb.datasets.mmlu.load_dataset(dataset_path, segment="test")

    #
    # Thens
    #

    # Answers should NOT be evenly distributed
    distribution = questions.answer.value_counts()
    assert not (distribution == distribution.iloc[0]).all()

    #
    # Whens
    #

    # I debias questions
    questions = lmb.datasets.mmlu.debias_question_answers(questions)

    #
    # Thens
    #

    # Answers should be evenly distributed
    distribution = questions.answer.value_counts()
    assert (distribution == distribution.iloc[0]).all()


def test_generate_prompt(datasets_path: Path):
    #
    # Givens
    #

    # Path to mmlu dataset
    dataset_path = datasets_path / "mmlu"

    # I loaded example questions
    example_questions = lmb.datasets.mmlu.load_dataset(dataset_path, segment="dev")

    # I loaded test questions
    test_questions = lmb.datasets.mmlu.load_dataset(dataset_path, segment="test")

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
        You are a robot that only outputs JSON. You reply in JSON format with the field 'answer'. For example, the following are multiple choice questions about professional law.
        
        Example Question: A state legislature has recently enacted a statute making it a misdemeanor to curse or revile or use obscene or opprobrious language toward or in reference to a police officer perfonning his duties. A student at a state university organized a demonstration on campus to protest the war. The rally was attended by a group of 50 students who shouted anti-war messages at cars passing by. To show his contempt for the United States, the student sewed the American flag to the rear of his jeans. When a police officer saw the flag sown on the student's jeans, he approached and told him to remove the flag or he would be placed under arrest. The student became angered and shouted at the police officer, "Listen, you bastard, I'll wear this rag anywhere I please. " The student was subsequently placed under arrest and charged with violating the state statute. The student subsequently brings suit in state court challenging the constitutionality of the statute. The strongest constitutional argument for the student is that
        
        A) the statute is void for vagueness under the Fourteenth Amendment's due process clause.
        B) the statute is invalid because it violates the petitioner's freedom of speech under the First Amendment.
        C) the statute is an abridgment of freedom of speech under the First Amendment because less restrictive means are available for achieving the same purpose.
        D) the statute is overbroad and consequently invalid under the First and FourteenthAmendments.
        
        Example Answer: {"answer": "D"}
        
        Example Question: A state has recently enacted a statute prohibiting the disposal of any nuclear wastes within the state. This law does not contravene or conflict with any federal statutes. A man operates a company in the state that is engaged in the disposal of nuclear wastes. Subsequent to the passage of the state statute, the man, not yet aware of the new law, entered into contracts with many out-of-state firms to dispose of their nuclear wastes in the state. On account of this new law, however, the man will be unable to perform these contracts. Assume that the man has standing to challenge this state law. Which of the following presents his strongest constitutional grounds to challenge the state law prohibiting the disposal of nuclear wastes within the state?
        
        A) The commerce clause.
        B) The equal protection clause of the Fourteenth Amendment.
        C) The privileges and immunities clause of Article IV, Section 2. 
        D) The contract clause.
        
        Example Answer: {"answer": "A"}
        
        Example Question: Judge took judicial notice of some facts at the beginning of the trial. Which of the following is not an appropriate kind of fact for judicial notice?
        
        A) Indisputable facts.
        B) Facts that have been asserted by individual political organizations.
        C) Facts recognized to be true by common knowledge.
        D) Facts capable of scientific verification.
        
        Example Answer: {"answer": "B"}
        
        Example Question: On October 1, 1980, a developer, owner of several hundred acres in a rural county, drafted a general development plan for the area. The duly recorded plan imposed elaborate limitations and restrictions upon the land in the plan, which was to be developed as a residential district. The restrictions were to extend to all persons acquiring any of the lots and to their heirs, assigns, and lessees. It was further provided that all subsequent owners would be charged with due notice of the restrictions. Among those restrictions in the general plan were the following:(22) A franchise right is created in a strip of land 10 feet in width along the rear of each lot for the use of public utility companies with right of ingress and egress. (23) No house or structure of any kind shall be built on the aforementioned strip of land running through the said blocks. In 2000, a retiree purchased one of the lots, built a house, and erected a fence in the rear of his property within the restricted area. In 2004, a teacher purchased a lot adjacent to the retiree's property and built a new house. Two years later, a librarian purchased the lot that adjoined the teacher's property. The three deeds to those properties each contained references to the deed book where the general plan was recorded. In 2008, the librarian began the construction of a seven-foot post-and-rail fence along the line dividing his lot with the teacher's, and along the center of the area subject to the franchise right. Although the teacher objected to its construction, the fence was completed. If the teacher seeks a mandatory injunction to compel removal of the librarian's fence, the court will most likely
        
        A) grant relief, because the fence was in violation of the easement restriction. 
        B) grant relief, because the encroachment of the fence violated the restriction in the original plan. 
        C) deny relief, because the teacher failed to enforce the restriction against the retiree. 
        D) deny relief, because the fence would not be construed as "a structure" within the terms of the restriction. 
        
        Example Answer: {"answer": "B"}
        
        Example Question: A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?
        
        A) The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. 
        B) Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. 
        C) The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. 
        D) By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. 
        
        Example Answer: {"answer": "A"}
        
        Given the examples above, your task is to answer the following question.
        
        Question: The defendant was caught in a thunderstorm while walking down the street. As the defendant was about to open an umbrella that she was carrying, a stranger to the defendant came up to her, snatched the umbrella out of the defendant's hand, and said, "You thief! That's my umbrella. " Enraged by being accosted in such a manner, the defendant grabbed for the umbrella with one hand and pushed the stranger with the other. The stranger hung on to the umbrella but fell over backward onto the sidewalk, which was wet and slippery. When the defendant saw the stranger hit the ground, she calmed down, decided the umbrella was not worth all the commotion, and walked off. As it turned out, the umbrella did in fact belong to the stranger, and the defendant had picked it up by mistake when she was left a restaurant earlier that day. A few moments later, the stranger got up in a daze and stepped into the gutter, where he was struck by a car that was passing another car on the right, in violation of a state law. The stranger died in the hospital two hours later. Which of the following is the most serious crime for which the defendant could be found guilty?
        
        A) Battery.
        B) Larceny.
        C) Involuntary manslaughter.
        D) No crime.
        
        Answer: """
    ).lstrip()
    assert prompt == expected


def test_golden_option(datasets_path: Path):
    #
    # Givens
    #

    # Path to mmlu dataset
    dataset_path = datasets_path / "mmlu"

    # Option A
    golden_option = "A"

    # I loaded mmlu original dataset
    dataset1 = lmb.datasets.mmlu.load_dataset(dataset_path)

    #
    # Whens
    #

    # I load mmlu dataset with golden option
    dataset2 = lmb.datasets.mmlu.load_dataset(dataset_path, golden_option=golden_option)

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
    # Givens
    #

    # Path to mmlu dataset
    dataset_path = datasets_path / "mmlu"

    #
    # Whens
    #

    # I load mmlu dataset twice
    dataset1 = lmb.datasets.mmlu.load_dataset(dataset_path)
    dataset2 = lmb.datasets.mmlu.load_dataset(dataset_path)

    #
    # Thens
    #

    # datasets should be equal
    assert_frame_equal(dataset1, dataset2)
