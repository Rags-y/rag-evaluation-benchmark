import json
from pathlib import Path
from typing import Any

from .evaluation_dataset import EvaluationDataset
from .evaluation_metrics import (
    answer_exact_match,
    is_abstention,
    retrieval_document_recall,
    retrieval_precision,
)
from .rag_pipeline import RAGPipeline


class EvaluationRunner:
    """Run the RAG system against an evaluation dataset."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        dataset: EvaluationDataset,
    ):
        self.pipeline = pipeline
        self.dataset = dataset

    def run(self) -> list[dict[str, Any]]:
        """Evaluate every question in the dataset."""

        questions = self.dataset.get_questions()

        results = []

        for item in questions:
            question = item["question"]

            print(
                f"Evaluating {item['question_id']}: "
                f"{question}"
            )

            rag_result = self.pipeline.answer(
                question
            )

            contexts = rag_result["contexts"]
            answer = rag_result["answer"]

            retrieval_recall = retrieval_document_recall(
                contexts,
                item["source_document_ids"],
            )

            retrieval_precision_score = retrieval_precision(
                contexts,
                item["source_document_ids"],
            )

            abstained = is_abstention(answer)

            exact_match = answer_exact_match(
                answer,
                item["expected_answer"],
            )

            results.append(
                {
                    "question_id": item["question_id"],
                    "question": question,
                    "question_type": item["question_type"],
                    "answerable": item["answerable"],
                    "expected_answer": item["expected_answer"],
                    "answer": answer,
                    "retrieval_precision": retrieval_precision_score,
                    "retrieval_document_recall": retrieval_recall,
                    "abstained": abstained,
                    "exact_match": exact_match,
                    "contexts": contexts,
                }
            )

        return results

    @staticmethod
    def save_results(
        results: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        """Save evaluation results as JSON."""

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )