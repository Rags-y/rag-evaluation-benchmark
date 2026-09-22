from rag_eval.evaluation_runner import EvaluationRunner


class FakePipeline:
    def answer(self, question):
        return {
            "question": question,
            "answer": "Python lists are mutable.",
            "contexts": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "python-data-structures",
                    "text": "Python lists are mutable sequences.",
                    "score": 0.9,
                    "metadata": {
                        "source_url": "https://example.com"
                    },
                }
            ],
        }


class FakeDataset:
    def get_questions(self):
        return [
            {
                "question_id": "q001",
                "question": "What are Python lists?",
                "question_type": "direct_factual",
                "answerable": True,
                "expected_answer": "Python lists are mutable.",
                "source_document_ids": [
                    "python-data-structures"
                ],
            }
        ]


def test_evaluation_runner():
    runner = EvaluationRunner(
        pipeline=FakePipeline(),
        dataset=FakeDataset(),
    )

    results = runner.run()

    assert len(results) == 1

    result = results[0]

    assert result["question_id"] == "q001"
    assert result["retrieval_precision"] == 1.0
    assert result["retrieval_document_recall"] == 1.0
    assert result["exact_match"] == 1.0
    assert result["abstained"] is False