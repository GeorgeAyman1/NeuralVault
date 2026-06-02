import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from uuid import uuid4


class MetadataStore:
    """
    Metadata storage for Phase 1.

    Stores memory text, user metadata, UUIDs, and timestamps.
    Supports save/load using JSON.
    """

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        record = {
            "id": str(uuid4()),
            "text": text,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
        }

        self.records.append(record)
        return len(self.records) - 1

    def get(self, index: int) -> dict[str, Any]:
        return self.records[index]

    def save(self, path: str = "data/processed/metadata.json") -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(self.records, file, indent=2, ensure_ascii=False)

    def load(self, path: str = "data/processed/metadata.json") -> None:
        load_path = Path(path)

        if not load_path.exists():
            self.records = []
            return

        with open(load_path, "r", encoding="utf-8") as file:
            self.records = json.load(file)

    def increment_retrieval_count(self, index: int) -> None:
        record = self.records[index]
        metadata = record.setdefault("metadata", {})
        metadata["retrieval_count"] = metadata.get("retrieval_count", 0) + 1

    def delete(self, index: int) -> None:
        if 0 <= index < len(self.records):
            self.records[index]["deleted"] = True

    def compact(self) -> list[int]:
        """Remove deleted records. Returns the original indices that were kept."""
        kept = [i for i, r in enumerate(self.records) if not r.get("deleted", False)]
        self.records = [self.records[i] for i in kept]
        return kept

    def count(self) -> int:
        return sum(1 for r in self.records if not r.get("deleted", False))
