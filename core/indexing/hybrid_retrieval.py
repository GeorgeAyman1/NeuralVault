from typing import Any

from core.indexing.keyword_search import KeywordSearch


class HybridRetrieval:
    """
    Combines semantic similarity with keyword overlap scoring.
    """

    def __init__(
        self,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        self.keyword_search = KeywordSearch()

    def rerank(
        self,
        query: str,
        semantic_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        reranked = []

        for result in semantic_results:
            semantic_score = result.get("score", 0.0)

            keyword_score = self.keyword_search.score(
                query=query,
                text=result.get("text", ""),
            )

            hybrid_score = (
                semantic_score * self.semantic_weight
                + keyword_score * self.keyword_weight
            )

            result["ranking"]["keyword_score"] = round(keyword_score, 4)
            result["ranking"]["hybrid_score"] = round(hybrid_score, 4)

            result["score"] = hybrid_score

            reranked.append(result)

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return reranked
