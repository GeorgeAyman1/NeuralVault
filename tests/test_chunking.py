from core.ingestion.chunking import TextChunker


def test_text_chunker_splits_paragraphs():
    text = "This is the first meaningful paragraph.\n\nThis is the second meaningful paragraph."

    chunker = TextChunker(min_chunk_chars=10)
    chunks = chunker.chunk(text)

    assert len(chunks) == 2
    assert chunks[0] == "This is the first meaningful paragraph."
    assert chunks[1] == "This is the second meaningful paragraph."


def test_text_chunker_ignores_short_chunks():
    text = "Hi.\n\nThis is a meaningful paragraph."

    chunker = TextChunker(min_chunk_chars=10)
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0] == "This is a meaningful paragraph."


def test_text_chunker_splits_long_text():
    text = " ".join(["word"] * 300)

    chunker = TextChunker(min_chunk_chars=10, max_chunk_chars=100)
    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)