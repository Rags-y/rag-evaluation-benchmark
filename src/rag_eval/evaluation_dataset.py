import json
from pathlib import Path
from typing import Any


class EvaluationDataset:
    """Load and validate the RAG evaluation question dataset."""

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)

    def load(self) -> list[dict[str, Any]]:
        """Load evaluation questions from JSON."""

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {self.dataset_path}"
            )

        with self.dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if "questions" not in data:
            raise ValueError(
                "Dataset must contain a 'questions' field"
            )

        questions = data["questions"]

        if not isinstance(questions, list):
            raise ValueError(
                "'questions' must be a list"
            )

        return questions

    def validate(self) -> None:
        """Validate the structure of every evaluation question."""

        questions = self.load()

        required_fields = {
            "question_id",
            "question",
            "question_type",
            "answerable",
            "expected_answer",
            "source_document_ids",
        }

        question_ids = set()

        for question in questions:
            missing_fields = required_fields - question.keys()

            if missing_fields:
                raise ValueError(
                    f"Question is missing fields: {missing_fields}"
                )

            question_id = question["question_id"]

            if question_id in question_ids:
                raise ValueError(
                    f"Duplicate question_id: {question_id}"
                )

            question_ids.add(question_id)

            if question["answerable"]:
                if question["expected_answer"] is None:
                    raise ValueError(
                        f"Answerable question has no expected answer: "
                        f"{question_id}"
                    )
            else:
                if question["expected_answer"] is not None:
                    raise ValueError(
                        f"Unanswerable question has an expected answer: "
                        f"{question_id}"
                    )

    def get_questions(self) -> list[dict[str, Any]]:
        """Load and validate the dataset."""

        self.validate()
        return self.load()

    def get_by_type(
        self,
        question_type: str,
    ) -> list[dict[str, Any]]:
        """Return questions of a specific type."""

        questions = self.get_questions()

        return [
            question
            for question in questions
            if question["question_type"] == question_type
        ]