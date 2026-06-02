from typing import Any
from core.memory.decay import MemoryDecay

class MemoryRanker:
    """
    Combines semantic similarity with memory intelligence signals.

    Current scoring:
    - similarity
    - recency
    - importance
    """

    def __init__(
        self,
        similarity_weight: float = 0.6,
        recency_weight: float = 0.2,
        importance_weight: float = 0.1,
        recency_half_life_days: float = 30.0,
        frequency_weight: float = 0.1,
    ):
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.recency_half_life_days = recency_half_life_days
        self.frequency_weight = frequency_weight
        self.decay = MemoryDecay()

    def rank(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked_results = []

        for result in results:
            similarity_score = float(result.get("score", 0.0))
            recency_score = self.decay.score(
                created_at=result.get("created_at"),
                metadata=result.get("metadata", {}),
            )
            importance_score = self._importance_score(result.get("metadata", {}))
            frequency_score = self._frequency_score(result.get("metadata", {}))

            final_score = (
                similarity_score * self.similarity_weight
                + recency_score * self.recency_weight
                + importance_score * self.importance_weight
                + frequency_score * self.frequency_weight
            )

            ranked_result = {
                **result,
                "ranking": {
                    "similarity_score": round(similarity_score, 4),
                    "recency_score": round(recency_score, 4),
                    "importance_score": round(importance_score, 4),
                    "final_score": round(final_score, 4),
                    "frequency_score": round(frequency_score, 4),
                },
            }

            ranked_results.append(ranked_result)

        ranked_results.sort(
            key=lambda item: item["ranking"]["final_score"],
            reverse=True,
        )

        return ranked_results


    def _importance_score(self, metadata: dict[str, Any]) -> float:
        importance = metadata.get("importance", 0.0)

        try:
            importance = float(importance)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(importance, 1.0))

    def _frequency_score(self, metadata: dict[str, Any]) -> float:
        retrieval_count = metadata.get("retrieval_count", 0)

        try:
            retrieval_count = int(retrieval_count)
        except (TypeError, ValueError):
            return 0.0

        return min(retrieval_count / 10, 1.0)
