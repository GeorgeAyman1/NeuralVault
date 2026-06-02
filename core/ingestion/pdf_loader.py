from pathlib import Path
from typing import Any

from core.ingestion.chunking import TextChunker


class PDFLoader:
    """
    Extracts text from PDF files page-by-page and chunks each page.

    Each chunk carries metadata: source_file, page_number, chunk_index.
    Requires: pypdf  (pip install pypdf)
    """

    def __init__(self):
        self.chunker = TextChunker()

    def load(self, path: str) -> list[dict[str, Any]]:
        try:
            import pypdf
        except ImportError as e:
            raise ImportError("pypdf is required for PDF ingestion: pip install pypdf") from e

        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        chunks = []

        with open(pdf_path, "rb") as fh:
            reader = pypdf.PdfReader(fh)

            for page_number, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()

                if not text:
                    continue

                for chunk_index, chunk_text in enumerate(self.chunker.chunk(text)):
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source_file": str(pdf_path),
                            "source_type": "pdf",
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                        },
                    })

        return chunks
