import re
from typing import Any


class KeywordSearch:
    """
    Lightweight keyword-overlap search.

    Useful for exact terms, code identifiers, acronyms, and dataset names.
    """

    def __init__(self):
        pass

    def score(self, query: str, text: str) -> float:
        query_terms = self._tokenize(query)
        text_terms = self._tokenize(text)

        if not query_terms or not text_terms:
            return 0.0

        overlap = query_terms.intersection(text_terms)

        return len(overlap) / len(query_terms)

    def score_records(
        self,
        query: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scored_records = []

        for index, record in enumerate(records):
            keyword_score = self.score(
                query=query,
                text=record.get("text", ""),
            )

            scored_records.append({
                "index": index,
                "keyword_score": keyword_score,
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
            })

        scored_records.sort(
            key=lambda item: item["keyword_score"],
            reverse=True,
        )

        return scored_records

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return set(tokens)