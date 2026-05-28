import numpy as np

from core.indexing.mmap_storage import MMapVectorStorage


def test_mmap_storage_save_and_load():
    vectors = [
        np.array([1, 0], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
    ]

    storage = MMapVectorStorage(
        path="data/indexes/test_vectors.mmap",
        shape_path="data/indexes/test_vectors_shape.npy",
    )

    storage.save(vectors)

    mmap = storage.load()

    assert mmap.shape == (2, 2)

    np.testing.assert_array_equal(
        mmap[0],
        np.array([1, 0], dtype=np.float32),
    )

    np.testing.assert_array_equal(
        mmap[1],
        np.array([0, 1], dtype=np.float32),
    )

    assert storage.exists()