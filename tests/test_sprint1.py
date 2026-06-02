"""
Sprint 1 tests: memory consolidation merge, pruning, summarization, compaction.

Covers:
- Soft delete: deleted memories excluded from search and count
- Merge: creates new combined memory, deletes both originals
- Prune: removes low-score memories, preserves accessed/important ones
- Summarize: creates extractive summary, deletes originals
- Compact: permanently removes soft-deleted entries, renumbers indices
- Consolidation find_candidates (existing, regression)
"""
import numpy as np
import pytest
from datetime import datetime, UTC, timedelta

from core.storage.vecdb_store import VecDBStore
from core.storage.metadata_store import MetadataStore
from core.memory.consolidation import MemoryConsolidator
from core.memory.pruning import MemoryPruner
from core.memory.summarization import MemorySummarizer


DIM = 16


def make_unit(rng, dim=DIM) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_stores(tmp_path, n=6):
    rng    = np.random.default_rng(42)
    vstore = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    mstore  = MetadataStore()
    vectors = []
    texts   = [f"memory {i}" for i in range(n)]
    for t in texts:
        v = make_unit(rng)
        vectors.append(v)
        mstore.add(t)
        vstore.add(v)
    return vstore, mstore, vectors, texts


# --------------------------------------------------------------------------- #
# Soft delete                                                                  #
# --------------------------------------------------------------------------- #

def test_delete_reduces_count(tmp_path):
    vstore, mstore, _, _ = make_stores(tmp_path)
    before = vstore.count()
    vstore.delete(0)
    mstore.delete(0)
    assert vstore.count() == before - 1
    assert mstore.count() == before - 1


def test_deleted_memory_excluded_from_search(tmp_path):
    rng    = np.random.default_rng(0)
    vstore = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    v0 = make_unit(rng)
    vstore.add(v0)
    vstore.add(make_unit(rng))

    results_before = [i for i, _ in vstore.search(v0, top_k=2)]
    assert 0 in results_before

    vstore.delete(0)
    results_after = [i for i, _ in vstore.search(v0, top_k=2)]
    assert 0 not in results_after


def test_sync_deleted_restores_state(tmp_path):
    vstore, mstore, _, _ = make_stores(tmp_path)
    vstore.delete(1)
    mstore.delete(1)

    vstore2 = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    vstore2._matrix = vstore._matrix
    vstore2.sync_deleted(mstore.records)
    assert 1 in vstore2._deleted
    assert vstore2.count() == vstore.count()


# --------------------------------------------------------------------------- #
# MetadataStore compact                                                        #
# --------------------------------------------------------------------------- #

def test_metadata_compact_removes_deleted(tmp_path):
    _, mstore, _, _ = make_stores(tmp_path)
    mstore.delete(0)
    mstore.delete(3)
    kept = mstore.compact()
    assert len(mstore.records) == 4
    assert all(not r.get("deleted", False) for r in mstore.records)
    assert 0 not in kept
    assert 3 not in kept


def test_vecdbstore_compact_rebuilds_matrix(tmp_path):
    vstore, mstore, _, _ = make_stores(tmp_path)
    vstore.delete(1)
    mstore.delete(1)
    kept = mstore.compact()
    vstore.compact(kept)

    assert vstore.count() == 5
    assert vstore._matrix is not None
    assert len(vstore._matrix) == 5
    assert len(vstore._deleted) == 0


# --------------------------------------------------------------------------- #
# Consolidation: find_candidates (regression)                                  #
# --------------------------------------------------------------------------- #

def test_consolidation_finds_similar_pair():
    v = np.array([1.0, 0.0], dtype=np.float32)
    w = np.array([0.99, 0.01], dtype=np.float32)
    w /= np.linalg.norm(w)
    x = np.array([0.0, 1.0], dtype=np.float32)

    consolidator = MemoryConsolidator(similarity_threshold=0.95)
    candidates   = consolidator.find_candidates(
        [v, w, x],
        [{"text": "a"}, {"text": "b"}, {"text": "c"}],
    )
    assert len(candidates) == 1
    assert {candidates[0]["memory_a_index"], candidates[0]["memory_b_index"]} == {0, 1}


def test_consolidation_respects_threshold():
    rng  = np.random.default_rng(99)
    vecs = [make_unit(rng) for _ in range(5)]
    recs = [{"text": f"m{i}"} for i in range(5)]
    assert MemoryConsolidator(similarity_threshold=0.999).find_candidates(vecs, recs) == []


# --------------------------------------------------------------------------- #
# Pruning                                                                      #
# --------------------------------------------------------------------------- #

def test_pruner_marks_old_unaccessed_memories():
    old = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    now = datetime.now(UTC).isoformat()
    records = [
        {"text": "old untouched",  "created_at": old, "metadata": {}},
        {"text": "recent",         "created_at": now, "metadata": {}},
        {"text": "old accessed",   "created_at": old, "metadata": {"retrieval_count": 5}},
        {"text": "old important",  "created_at": old, "metadata": {"importance": 0.9}},
    ]
    prunable = MemoryPruner(threshold=0.3).find_prunable(records)

    assert 0 in prunable
    assert 1 not in prunable
    assert 2 not in prunable
    assert 3 not in prunable


def test_pruner_skips_already_deleted():
    old = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    records = [{"text": "old", "created_at": old, "metadata": {}, "deleted": True}]
    assert MemoryPruner(threshold=0.9).find_prunable(records) == []


def test_prune_then_compact_removes_entries(tmp_path):
    old = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    vstore, mstore, _, _ = make_stores(tmp_path, n=4)

    for i in (0, 1):
        mstore.records[i]["created_at"] = old

    prunable = MemoryPruner(threshold=0.3).find_prunable(mstore.records)
    for idx in prunable:
        mstore.delete(idx)
        vstore.delete(idx)

    assert mstore.count() == 4 - len(prunable)
    kept = mstore.compact()
    vstore.compact(kept)
    assert len(mstore.records) == mstore.count()


# --------------------------------------------------------------------------- #
# Summarization                                                                #
# --------------------------------------------------------------------------- #

def test_summarizer_returns_string():
    rng     = np.random.default_rng(7)
    records = [{"text": f"memory {i}"} for i in range(3)]
    vectors = [make_unit(rng) for _ in range(3)]
    summary = MemorySummarizer().summarize(records, vectors)
    assert isinstance(summary, str) and len(summary) > 0


def test_summarizer_single_record_returns_text():
    v = np.ones(DIM, dtype=np.float32) / DIM ** 0.5
    assert MemorySummarizer().summarize([{"text": "only"}], [v]) == "only"


def test_summarizer_includes_central_text():
    v_center = np.ones(DIM, dtype=np.float32)
    v_center /= np.linalg.norm(v_center)
    rng     = np.random.default_rng(8)
    v_other = make_unit(rng)

    summary = MemorySummarizer().summarize(
        [{"text": "central"}, {"text": "peripheral"}],
        [v_center, v_other],
    )
    assert "central" in summary


def test_summarizer_raises_on_empty():
    with pytest.raises(ValueError):
        MemorySummarizer().summarize([], [])


# --------------------------------------------------------------------------- #
# Merge simulation (store-level)                                               #
# --------------------------------------------------------------------------- #

def test_merge_creates_new_and_deletes_originals(tmp_path):
    vstore, mstore, vectors, texts = make_stores(tmp_path, n=4)
    count_before = mstore.count()

    merged_text = f"{texts[0]}\n{texts[1]}"
    merged_meta = {"merged_from": [mstore.records[0]["id"], mstore.records[1]["id"]]}

    mstore.add(merged_text, merged_meta)
    vstore.add(vectors[0] + vectors[1])
    mstore.delete(0)
    mstore.delete(1)
    vstore.delete(0)
    vstore.delete(1)

    assert mstore.count() == count_before - 1

    kept = mstore.compact()
    vstore.compact(kept)
    assert mstore.count() == 3
    assert any(texts[0] in r["text"] for r in mstore.records)
