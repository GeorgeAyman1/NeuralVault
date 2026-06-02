from datetime import datetime, UTC
from math import exp
from typing import Any


class MemoryDecay:
    """
    Calculates memory decay over time.

    Important memories decay more slowly.
    """

    def __init__(
        self,
        base_half_life_days: float = 30.0,
        importance_boost: float = 2.0,
    ):
        self.base_half_life_days = base_half_life_days
        self.importance_boost = importance_boost

    def score(self, created_at: str | None, metadata: dict[str, Any] | None = None) -> float:
        if not created_at:
            return 0.0

        metadata = metadata or {}

        created_time = datetime.fromisoformat(created_at)
        now = datetime.now(UTC)

        age_days = max((now - created_time).total_seconds() / 86400, 0)

        importance = self._importance(metadata)

        effective_half_life = self.base_half_life_days * (
            1 + importance * self.importance_boost
        )

        return exp(-age_days / effective_half_life)

    def _importance(self, metadata: dict[str, Any]) -> float:
        value = metadata.get("importance", 0.0)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(value, 1.0))
