from pathlib import Path
from typing import Any

from core.ingestion.chunking import TextChunker


class DOCXLoader:
    """
    Extracts text from .docx files paragraph-by-paragraph and chunks them.

    Each chunk carries metadata: source_file, paragraph_index, chunk_index.
    Requires: python-docx  (pip install python-docx)
    """

    def __init__(self):
        self.chunker = TextChunker()

    def load(self, path: str) -> list[dict[str, Any]]:
        try:
            import docx
        except ImportError as e:
            raise ImportError(
                "python-docx is required for DOCX ingestion: pip install python-docx"
            ) from e

        docx_path = Path(path)
        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {path}")

        document   = docx.Document(str(docx_path))
        full_text  = "\n\n".join(
            para.text.strip()
            for para in document.paragraphs
            if para.text.strip()
        )

        chunks = []
        for chunk_index, chunk_text in enumerate(self.chunker.chunk(full_text)):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_file": str(docx_path),
                    "source_type": "docx",
                    "chunk_index": chunk_index,
                },
            })

        return chunks
