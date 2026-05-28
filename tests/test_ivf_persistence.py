import numpy as np

from core.indexing.ivf_persistence import IVFPersistence


def test_ivf_persistence_save_and_load():
    centroids = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )

    partitions = {
        0: [0, 1],
        1: [2, 3],
    }

    persistence = IVFPersistence(
        centroids_path="data/indexes/test_ivf_centroids.npy",
        partitions_path="data/indexes/test_ivf_partitions.json",
    )

    persistence.save(centroids, partitions)

    loaded_centroids, loaded_partitions = persistence.load()

    np.testing.assert_array_equal(loaded_centroids, centroids)
    assert loaded_partitions == partitions
    assert persistence.exists()