from core.indexing.keyword_search import KeywordSearch


def test_keyword_search_scores_exact_overlap():
    search = KeywordSearch()

    score = search.score(
        query="VecDB top_k retrieval",
        text="The VecDB class uses top_k to retrieve nearest results.",
    )

    assert score > 0.5


def test_keyword_search_returns_zero_for_no_overlap():
    search = KeywordSearch()

    score = search.score(
        query="VecDB top_k retrieval",
        text="Pizza is a popular food.",
    )

    assert score == 0.0


def test_keyword_search_scores_records():
    search = KeywordSearch()

    records = [
        {"text": "Pizza recipe"},
        {"text": "VecDB uses top_k for retrieval"},
    ]

    results = search.score_records("VecDB top_k", records)

    assert results[0]["text"] == "VecDB uses top_k for retrieval"
    assert results[0]["keyword_score"] == 1.0
