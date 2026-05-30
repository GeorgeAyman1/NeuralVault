import numpy as np

from vec_db import VecDB


def test_vec_db_retrieve_returns_nearest_ids(tmp_path):
    vectors = np.array(
        [
            [1, 0],
            [0.9, 0.1],
            [0, 1],
        ],
        dtype=np.float32,
    )

    vector_path = tmp_path / "vectors.npy"
    np.save(vector_path, vectors)

    db = VecDB(
        database_file_path=str(vector_path),
        index_file_path=None,
        new_db=False,
        db_size=3,
    )

    results = db.retrieve(
        np.array([1, 0], dtype=np.float32),
        top_k=2,
    )

    assert results == [0, 1]