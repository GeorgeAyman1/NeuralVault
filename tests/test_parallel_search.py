import numpy as np

from core.indexing.parallel_search import ParallelPartitionSearch


def test_parallel_partition_search_returns_top_results():
    vectors = [
        np.array([1, 0], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
    ]

    searcher = ParallelPartitionSearch(max_workers=2)

    results = searcher.search(
        vectors=vectors,
        candidate_indices=[0, 1, 2, 3],
        query_vector=np.array([1, 0], dtype=np.float32),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0] == 0
    assert results[0][1] == 1.0
    assert results[1][0] == 1
