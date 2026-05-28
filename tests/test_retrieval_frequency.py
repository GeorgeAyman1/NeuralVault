from core.memory.service import MemoryService


def test_retrieval_count_increments_after_search():
    memory = MemoryService(auto_load=False)

    memory.add_memory("AI memory systems are useful")

    memory.search_memory("AI memory")
    results = memory.search_memory("AI memory")

    metadata = results[0]["metadata"]

    assert metadata["retrieval_count"] == 2
    assert results[0]["ranking"]["frequency_score"] == 0.2