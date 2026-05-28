from datetime import datetime, UTC, timedelta

from core.memory.ranking import MemoryRanker


def test_memory_ranker_uses_similarity_recency_and_importance():
    now = datetime.now(UTC)

    results = [
        {
            "score": 0.8,
            "text": "Older but important memory",
            "metadata": {"importance": 1.0},
            "created_at": (now - timedelta(days=10)).isoformat(),
        },
        {
            "score": 0.9,
            "text": "Recent but less important memory",
            "metadata": {"importance": 0.1},
            "created_at": now.isoformat(),
        },
    ]

    ranker = MemoryRanker()
    ranked = ranker.rank(results)

    assert len(ranked) == 2
    assert "ranking" in ranked[0]
    assert "final_score" in ranked[0]["ranking"]
    assert ranked[0]["ranking"]["final_score"] >= ranked[1]["ranking"]["final_score"]


def test_importance_score_is_clamped():
    ranker = MemoryRanker()

    results = [
        {
            "score": 0.5,
            "text": "Invalid high importance",
            "metadata": {"importance": 5},
            "created_at": datetime.now(UTC).isoformat(),
        }
    ]

    ranked = ranker.rank(results)

    assert ranked[0]["ranking"]["importance_score"] == 1.0 