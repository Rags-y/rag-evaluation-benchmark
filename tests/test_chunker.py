from rag_eval.chunker import TextChunker
from rag_eval.schemas import Document


def test_chunker_creates_chunks():
    document = Document(
        document_id="test-doc",
        title="Test Document",
        source_url="https://example.com",
        text="A" * 2500,
    )

    chunker = TextChunker(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "test-doc-chunk-0000"
    assert chunks[1].chunk_id == "test-doc-chunk-0001"


def test_empty_document_returns_no_chunks():
    document = Document(
        document_id="empty",
        title="Empty",
        source_url="https://example.com",
        text="",
    )

    chunker = TextChunker()

    assert chunker.chunk_document(document) == []


def test_invalid_overlap_raises_error():
    try:
        TextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )
        assert False
    except ValueError:
        assert True