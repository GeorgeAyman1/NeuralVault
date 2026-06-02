from pathlib import Path
from typing import Any

from core.ingestion.chunking import TextChunker


# File types treated as plain text; code files get a slightly larger max chunk
# so function/class bodies aren't fragmented unnecessarily.
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".log"}
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".r", ".sql", ".sh",
    ".yaml", ".yml", ".toml", ".json",
}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | CODE_EXTENSIONS

# Directories that are never useful to ingest
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__",
    "node_modules", ".idea", ".vscode", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
}


class DirectoryLoader:
    """
    Recursively ingests a directory tree, extracting text from supported files.

    Supported: .txt .md .rst .csv .log + common code extensions.
    Skips: binary files, hidden directories, virtual-env / build artefacts.

    Each chunk carries: source_file, source_type, relative_path, chunk_index.
    """

    def __init__(
        self,
        extensions: set[str] | None = None,
        max_file_bytes: int = 1_000_000,  # skip files > 1 MB
    ):
        self._extensions    = extensions or SUPPORTED_EXTENSIONS
        self._max_file_bytes = max_file_bytes
        self._chunker       = TextChunker(min_chunk_chars=40, max_chunk_chars=1200)

    def load(self, path: str) -> list[dict[str, Any]]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        chunks = []
        for file_path in self._walk(root):
            file_chunks = self._load_file(file_path, root)
            chunks.extend(file_chunks)

        return chunks

    # ------------------------------------------------------------------ #

    def _walk(self, root: Path):
        for item in sorted(root.iterdir()):
            if item.is_dir():
                if item.name not in SKIP_DIRS and not item.name.startswith("."):
                    yield from self._walk(item)
            elif item.is_file() and item.suffix.lower() in self._extensions:
                if item.stat().st_size <= self._max_file_bytes:
                    yield item

    def _load_file(self, file_path: Path, root: Path) -> list[dict[str, Any]]:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return []

        if not text:
            return []

        suffix      = file_path.suffix.lower()
        source_type = "code" if suffix in CODE_EXTENSIONS else "text"
        rel_path    = str(file_path.relative_to(root))

        chunks = []
        for chunk_index, chunk_text in enumerate(self._chunker.chunk(text)):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_file":    str(file_path),
                    "source_type":    source_type,
                    "relative_path":  rel_path,
                    "file_extension": suffix,
                    "chunk_index":    chunk_index,
                },
            })

        return chunks
