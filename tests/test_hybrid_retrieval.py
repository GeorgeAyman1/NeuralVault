from core.indexing.hybrid_retrieval import HybridRetrieval


def test_hybrid_retrieval_boosts_keyword_matches():
    retrieval = HybridRetrieval()

    semantic_results = [
        {
            "score": 0.8,
            "text": "VecDB top_k retrieval system",
            "ranking": {},
        },
        {
            "score": 0.82,
            "text": "General AI semantic memory",
            "ranking": {},
        },
    ]

    results = retrieval.rerank(
        query="VecDB top_k",
        semantic_results=semantic_results,
    )

    assert results[0]["text"] == "VecDB top_k retrieval system"
    assert results[0]["ranking"]["keyword_score"] > 0.0
