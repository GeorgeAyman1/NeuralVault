import numpy as np

from core.memory.consolidation import MemoryConsolidator


def test_memory_consolidator_finds_similar_memories():
    vectors = [
        np.array([1, 0], dtype=np.float32),
        np.array([0.95, 0.05], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
    ]

    records = [
        {"text": "AI memory systems"},
        {"text": "Artificial intelligence memory systems"},
        {"text": "Pizza recipe"},
    ]

    consolidator = MemoryConsolidator(similarity_threshold=0.9)
    candidates = consolidator.find_candidates(vectors, records)

    assert len(candidates) == 1
    assert candidates[0]["memory_a_index"] == 0
    assert candidates[0]["memory_b_index"] == 1
    assert candidates[0]["similarity"] >= 0.9
