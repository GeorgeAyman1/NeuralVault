import json
from pathlib import Path
from typing import Any

from core.ingestion.chunking import TextChunker


class NotebookLoader:
    """
    Loads Jupyter notebooks and extracts markdown/code cells as text chunks.
    """

    def __init__(self):
        self.chunker = TextChunker()

    def load(self, path: str) -> list[dict[str, Any]]:
        notebook_path = Path(path)

        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {path}")

        with open(notebook_path, "r", encoding="utf-8") as file:
            notebook = json.load(file)

        chunks = []

        for cell_index, cell in enumerate(notebook.get("cells", [])):
            cell_type = cell.get("cell_type", "unknown")
            source = cell.get("source", [])

            if isinstance(source, list):
                text = "".join(source).strip()
            else:
                text = str(source).strip()

            if not text:
                continue

            text_chunks = self.chunker.chunk(text)

            for chunk_index, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_file": str(notebook_path),
                        "cell_index": cell_index,
                        "chunk_index": chunk_index,
                        "cell_type": cell_type,
                    },
                })

        return chunks
