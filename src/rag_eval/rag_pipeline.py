from pathlib import Path
import os
import requests

from .vector_store import VectorStore


class MockGenerator:
    """Deterministic generator used for local testing."""

    def generate(self, question: str, contexts: list[dict]) -> str:
        if not contexts:
            return "I don't have enough information to answer this question."
        return contexts[0]["text"]


class GeminiGenerator:
    """Optional Gemini generator using the Gemini REST API."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when using the Gemini generator."
            )

    def generate(self, question: str, contexts: list[dict]) -> str:
        if not contexts:
            return "I don't have enough information to answer this question."

        context_text = "\n\n".join(
            f"[Chunk {context['chunk_id']}]\n{context['text']}"
            for context in contexts
        )

        prompt = f"""Answer the question using only the provided context.

If the context does not contain enough information to answer the question,
say: "I don't have enough information to answer this question."

Question:
{question}

Context:
{context_text}

Answer:"""

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )

        response = requests.post(
            url,
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ]
                    }
                ]
            },
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Gemini returned an unexpected response format."
            ) from exc


def create_generator(config: dict):
    """Create the configured answer generator."""
    generation_config = config.get("generation", {})
    provider = generation_config.get("provider", "mock").lower()

    if provider == "mock":
        return MockGenerator()

    if provider == "gemini":
        return GeminiGenerator(
            model_name=generation_config.get(
                "model",
                "gemini-2.0-flash",
            ),
        )

    raise ValueError(
        f"Unsupported generation provider: {provider}. "
        "Supported providers: mock, gemini."
    )


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
        self.vector_store = VectorStore(model_name=embedding_model)
        self.vector_store.load(index_dir)
        self.generator = generator or MockGenerator()

    def retrieve(self, question: str) -> list[dict]:
        return self.vector_store.search(question, top_k=self.top_k)

    def generate(self, question: str, contexts: list[dict]) -> str:
        return self.generator.generate(
            question=question,
            contexts=contexts,
        )

    def answer(self, question: str) -> dict:
        contexts = self.retrieve(question)

        if (
            self.abstention_threshold is not None
            and (
                not contexts
                or contexts[0]["score"] < self.abstention_threshold
            )
        ):
            answer = "I don't have enough information to answer this question."
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