import numpy as np

from core.indexing.ivf_index import IVFIndex
from core.indexing.mmap_storage import MMapVectorStorage


def test_ivf_index_with_mmap_vectors():
    vectors = [
        np.array([1, 0], dtype=np.float32),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
        np.array([0.1, 0.9], dtype=np.float32),
    ]

    storage = MMapVectorStorage(
        path="data/indexes/test_ivf_vectors.mmap",
        shape_path="data/indexes/test_ivf_vectors_shape.npy",
    )

    storage.save(vectors)
    mmap_vectors = storage.load()

    index = IVFIndex(n_clusters=2, n_probe=1)
    index.build(mmap_vectors)

    results = index.search(
        np.array([1, 0], dtype=np.float32),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0] == 0
    assert results[0][1] == 1.0
    assert results[1][0] == 1