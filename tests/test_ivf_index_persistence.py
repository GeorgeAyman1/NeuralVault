import numpy as np

from core.indexing.ivf_index import IVFIndex


def test_ivf_index_save_and_load():
    vectors = [
        np.array([1, 0], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
    ]

    index = IVFIndex(n_clusters=2, n_probe=1)
    index.build(vectors)
    index.save()

    loaded_index = IVFIndex(n_clusters=2, n_probe=1)
    loaded_index.load(vectors)

    results = loaded_index.search(
        np.array([1, 0], dtype=np.float32),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0] == 0
    assert results[1][0] == 1
