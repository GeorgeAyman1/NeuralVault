"""
Sprint 0 tests: VecDBStore wired into MemoryService.

Covers:
- Add and search (brute-force)
- Save / load persistence
- IVF index build and activation
- IVF search matches brute-force results
- SemanticRetriever returns ranked results with metadata
"""
import numpy as np
import pytest

from core.storage.vecdb_store import VecDBStore
from core.storage.metadata_store import MetadataStore
from core.indexing.retrieval import SemanticRetriever


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

TEXTS = [
    "Neural networks learn from data",
    "Deep learning is a subset of machine learning",
    "The cat sat on the mat",
    "Dogs are loyal companions",
    "Python is great for data science",
    "Transformers changed natural language processing",
]

DIM = 16  # small dim for fast tests — no real encoder needed


def make_unit(rng, dim=DIM) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def populated_store(tmp_path, n=len(TEXTS)):
    rng   = np.random.default_rng(0)
    store = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    vectors = [make_unit(rng) for _ in range(n)]
    for v in vectors:
        store.add(v)
    return store, vectors


# --------------------------------------------------------------------------- #
# VecDBStore: add / count / search                                             #
# --------------------------------------------------------------------------- #

def test_vecdbstore_add_and_count(tmp_path):
    store, _ = populated_store(tmp_path)
    assert store.count() == len(TEXTS)


def test_vecdbstore_search_returns_correct_top_k(tmp_path):
    store, vectors = populated_store(tmp_path)
    query   = vectors[2]   # exact match for index 2
    results = store.search(query, top_k=1)
    assert len(results) == 1
    assert results[0][0] == 2
    assert abs(results[0][1] - 1.0) < 1e-4


def test_vecdbstore_search_top_k_ordering(tmp_path):
    store, vectors = populated_store(tmp_path)
    query   = vectors[0]
    results = store.search(query, top_k=3)
    scores  = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# VecDBStore: save / load                                                      #
# --------------------------------------------------------------------------- #

def test_vecdbstore_save_and_reload(tmp_path):
    store, vectors = populated_store(tmp_path)
    store.save()

    store2 = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    store2.load()

    assert store2.count() == len(TEXTS)
    results = store2.search(vectors[1], top_k=1)
    assert results[0][0] == 1


# --------------------------------------------------------------------------- #
# VecDBStore: IVF index                                                        #
# --------------------------------------------------------------------------- #

def test_vecdbstore_build_index_activates_ivf(tmp_path):
    rng   = np.random.default_rng(1)
    store = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    for _ in range(150):
        store.add(make_unit(rng))

    assert not store._index_valid
    store.save()
    store.build_index(n_clusters=8)
    assert store._index_valid


def test_vecdbstore_ivf_matches_brute_force(tmp_path):
    rng   = np.random.default_rng(2)
    store = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    vectors = [make_unit(rng) for _ in range(200)]
    for v in vectors:
        store.add(v)

    query = make_unit(rng)
    r_bf  = [i for i, _ in store.search(query, top_k=5)]

    store.save()
    store.build_index(n_clusters=8)
    r_ivf = [i for i, _ in store.search(query, top_k=5)]

    assert set(r_bf) == set(r_ivf), f"BF={r_bf}  IVF={r_ivf}"


def test_vecdbstore_ivf_persists_across_reload(tmp_path):
    rng   = np.random.default_rng(3)
    store = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    vectors = [make_unit(rng) for _ in range(150)]
    for v in vectors:
        store.add(v)

    store.save()
    store.build_index(n_clusters=8)

    store2 = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    store2.load()
    assert store2._index_valid


# --------------------------------------------------------------------------- #
# SemanticRetriever                                                            #
# --------------------------------------------------------------------------- #

class FakeEncoder:
    """Returns unit vectors keyed by text for deterministic tests."""
    def __init__(self, mapping: dict):
        self._map = mapping

    def encode(self, text: str) -> np.ndarray:
        v = self._map.get(text, np.ones(DIM, dtype=np.float32))
        return (v / np.linalg.norm(v)).astype(np.float32)


def test_retriever_returns_correct_match(tmp_path):
    rng  = np.random.default_rng(4)
    vecs = {t: make_unit(rng) for t in TEXTS}

    mstore = MetadataStore()
    vstore = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )

    for t in TEXTS:
        mstore.add(t)
        vstore.add(vecs[t])

    encoder   = FakeEncoder(vecs)
    retriever = SemanticRetriever(encoder=encoder, vector_store=vstore, metadata_store=mstore)

    results = retriever.search(TEXTS[0], top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == TEXTS[0]
    assert "ranking" in results[0]


def test_retriever_result_has_required_fields(tmp_path):
    rng   = np.random.default_rng(5)
    vecs  = {t: make_unit(rng) for t in TEXTS[:3]}

    mstore = MetadataStore()
    vstore = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    for t in TEXTS[:3]:
        mstore.add(t, {"source": "test"})
        vstore.add(vecs[t])

    encoder   = FakeEncoder(vecs)
    retriever = SemanticRetriever(encoder=encoder, vector_store=vstore, metadata_store=mstore)

    result = retriever.search(TEXTS[0], top_k=1)[0]
    for field in ("score", "text", "metadata", "created_at", "id", "ranking"):
        assert field in result, f"Missing field: {field}"


def test_retriever_increments_retrieval_count(tmp_path):
    rng   = np.random.default_rng(6)
    vecs  = {t: make_unit(rng) for t in TEXTS[:2]}

    mstore = MetadataStore()
    vstore = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    for t in TEXTS[:2]:
        mstore.add(t)
        vstore.add(vecs[t])

    encoder   = FakeEncoder(vecs)
    retriever = SemanticRetriever(encoder=encoder, vector_store=vstore, metadata_store=mstore)

    retriever.search(TEXTS[0], top_k=1)
    count = mstore.records[0]["metadata"].get("retrieval_count", 0)
    assert count == 1
