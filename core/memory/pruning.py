from typing import Any

from core.memory.decay import MemoryDecay


class MemoryPruner:
    """
    Identifies memories that are safe to remove based on a composite keep-score.

    Score = recency * 0.5 + importance * 0.3 + frequency * 0.2

    Memories below `threshold` are considered prunable.
    Memories with retrieval_count > 0 or importance > 0.5 are never pruned
    regardless of threshold — they have been marked as valuable.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self._decay = MemoryDecay()

    def find_prunable(self, records: list[dict[str, Any]]) -> list[int]:
        prunable = []

        for i, record in enumerate(records):
            if record.get("deleted", False):
                continue

            meta = record.get("metadata", {})

            importance     = self._safe_float(meta.get("importance", 0.0))
            retrieval_count = self._safe_int(meta.get("retrieval_count", 0))

            # Never prune memories the user has accessed or explicitly flagged
            if retrieval_count > 0 or importance > 0.5:
                continue

            recency   = self._decay.score(record.get("created_at"), meta)
            frequency = min(retrieval_count / 10, 1.0)
            score     = recency * 0.5 + importance * 0.3 + frequency * 0.2

            if score < self.threshold:
                prunable.append(i)

        return prunable

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
