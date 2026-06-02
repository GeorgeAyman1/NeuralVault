class TextChunker:
    """
    Simple paragraph-based text chunker.

    Splits long text into smaller chunks for better semantic retrieval.
    """

    def __init__(self, min_chunk_chars: int = 40, max_chunk_chars: int = 1000):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.max_chunk_chars:
                if len(paragraph) >= self.min_chunk_chars:
                    chunks.append(paragraph)
                continue

            chunks.extend(self._split_long_text(paragraph))

        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        current = []

        for word in words:
            candidate = " ".join(current + [word])

            if len(candidate) > self.max_chunk_chars:
                chunk = " ".join(current).strip()

                if len(chunk) >= self.min_chunk_chars:
                    chunks.append(chunk)

                current = [word]
            else:
                current.append(word)

        final_chunk = " ".join(current).strip()

        if len(final_chunk) >= self.min_chunk_chars:
            chunks.append(final_chunk)

        return chunks
