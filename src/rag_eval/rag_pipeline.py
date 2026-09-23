from pathlib import Path

from .vector_store import VectorStore


class MockGenerator:
    """Deterministic generator used for local testing."""

    def generate(
        self,
        question: str,
        contexts: list[dict],
    ) -> str:
        if not contexts:
            return "I don't have enough information to answer this question."

        return contexts[0]["text"]


class RAGPipeline:
    """Baseline retrieval-augmented generation pipeline."""

    def __init__(
        self,
        index_dir: str | Path,
        top_k: int = 5,
        embedding_model: str = "all-MiniLM-L6-v2",
        generator=None,
        abstention_threshold: float | None = None,
    ):
        self.top_k = top_k
        self.abstention_threshold = abstention_threshold

        self.vector_store = VectorStore(
            model_name=embedding_model,
        )

        self.vector_store.load(index_dir)

        self.generator = generator or MockGenerator()

    def retrieve(
        self,
        question: str,
    ) -> list[dict]:
        """Retrieve relevant chunks for a question."""

        return self.vector_store.search(
            question,
            top_k=self.top_k,
        )

    def generate(
        self,
        question: str,
        contexts: list[dict],
    ) -> str:
        """Generate an answer from retrieved contexts."""

        return self.generator.generate(
            question=question,
            contexts=contexts,
        )

    def answer(
        self,
        question: str,
    ) -> dict:
        """Run retrieval and generation."""

        contexts = self.retrieve(question)

        if (
            self.abstention_threshold is not None
            and (
                not contexts
                or contexts[0]["score"]
                < self.abstention_threshold
            )
        ):
            answer = (
                "I don't have enough information "
                "to answer this question."
            )

            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }

        answer = self.generate(
            question=question,
            contexts=contexts,
        )

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }