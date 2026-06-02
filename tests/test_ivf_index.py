import numpy as np

from core.indexing.ivf_index import IVFIndex


def test_ivf_index_returns_nearest_vectors():
    vectors = [
        np.array([1, 0], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
    ]

    index = IVFIndex(n_clusters=2, n_probe=1)
    index.build(vectors)

    results = index.search(
        np.array([1, 0], dtype=np.float32),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0] == 0
    assert results[0][1] == 1.0
