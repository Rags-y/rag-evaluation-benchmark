from pathlib import Path

from rag_eval.evaluation_dataset import EvaluationDataset


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation"
    / "questions.json"
)


def test_evaluation_dataset_loads():
    dataset = EvaluationDataset(DATASET_PATH)

    questions = dataset.get_questions()

    assert len(questions) == 40


def test_evaluation_dataset_validates():
    dataset = EvaluationDataset(DATASET_PATH)

    dataset.validate()


def test_get_by_type():
    dataset = EvaluationDataset(DATASET_PATH)

    questions = dataset.get_by_type(
        "direct_factual"
    )

    assert len(questions) == 15


def test_question_ids_are_unique():
    dataset = EvaluationDataset(DATASET_PATH)

    questions = dataset.get_questions()

    question_ids = [
        question["question_id"]
        for question in questions
    ]

    assert len(question_ids) == len(
        set(question_ids)
    )