import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "evaluation" / "questions.json"


def load_questions():
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def test_dataset_exists():
    assert DATASET_PATH.exists()


def test_dataset_has_40_questions():
    data = load_questions()

    assert len(data["questions"]) == 40


def test_question_ids_are_unique():
    data = load_questions()

    question_ids = [
        question["question_id"]
        for question in data["questions"]
    ]

    assert len(question_ids) == len(set(question_ids))


def test_required_question_types_are_present():
    data = load_questions()

    question_types = {
        question["question_type"]
        for question in data["questions"]
    }

    expected_types = {
        "direct_factual",
        "multi_hop",
        "unanswerable",
        "misleading",
    }

    assert expected_types.issubset(question_types)


def test_question_type_distribution():
    data = load_questions()

    counts = {}

    for question in data["questions"]:
        question_type = question["question_type"]
        counts[question_type] = counts.get(
            question_type,
            0,
        ) + 1

    assert counts["direct_factual"] == 15
    assert counts["multi_hop"] == 10
    assert counts["unanswerable"] == 8
    assert counts["misleading"] == 7


def test_answerability_matches_expected_answer():
    data = load_questions()

    for question in data["questions"]:
        if question["answerable"]:
            assert question["expected_answer"] is not None
        else:
            assert question["expected_answer"] is None


def test_all_questions_have_required_fields():
    data = load_questions()

    required_fields = {
        "question_id",
        "question",
        "question_type",
        "answerable",
        "expected_answer",
        "source_document_ids",
    }

    for question in data["questions"]:
        assert required_fields.issubset(
            question.keys()
        )