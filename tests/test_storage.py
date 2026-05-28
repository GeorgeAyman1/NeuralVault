import numpy as np

from core.storage.vector_store import VectorStore
from core.storage.metadata_store import MetadataStore


def test_vector_store_add_and_search():
    store = VectorStore()

    store.add(np.array([1, 0], dtype=np.float32))
    store.add(np.array([0, 1], dtype=np.float32))

    results = store.search(np.array([1, 0], dtype=np.float32), top_k=2)

    assert store.count() == 2
    assert results[0][0] == 0
    assert results[0][1] == 1.0


def test_metadata_store_add_and_get():
    store = MetadataStore()

    index = store.add("NeuralVault stores memory", {"source": "test"})
    record = store.get(index)

    assert index == 0
    assert store.count() == 1
    assert record["text"] == "NeuralVault stores memory"
    assert record["metadata"]["source"] == "test"
    assert "id" in record
    assert "created_at" in record