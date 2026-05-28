import numpy as np

from core.indexing.retrieval import SemanticRetriever
from core.storage.vector_store import VectorStore
from core.storage.metadata_store import MetadataStore


class FakeEncoder:
    def encode(self, text: str):
        vectors = {
            "ai": np.array([1, 0], dtype=np.float32),
            "pizza": np.array([0, 1], dtype=np.float32),
        }

        return vectors.get(text, np.array([1, 0], dtype=np.float32))


def test_semantic_retriever_returns_best_match():
    encoder = FakeEncoder()
    vector_store = VectorStore()
    metadata_store = MetadataStore()

    metadata_store.add("ai", {"category": "tech"})
    vector_store.add(np.array([1, 0], dtype=np.float32))

    metadata_store.add("pizza", {"category": "food"})
    vector_store.add(np.array([0, 1], dtype=np.float32))

    retriever = SemanticRetriever(
        encoder=encoder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    results = retriever.search("ai", top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "ai"
    assert results[0]["metadata"]["category"] == "tech"