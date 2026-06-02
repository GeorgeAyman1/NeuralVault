"""
Live end-to-end test of Sprint 3 chat() against the real Anthropic API.

Requires ANTHROPIC_API_KEY in the environment. Uses an isolated temp store
so it does not touch your real data/ directory.

Run:
    python scripts/live_chat_test.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory.service import MemoryService
from core.storage.vecdb_store import VecDBStore
from core.indexing.retrieval import SemanticRetriever


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set in the environment.")
        return 1

    tmp = tempfile.mkdtemp()
    svc = MemoryService(auto_load=False)  # real LLMClient (reads ANTHROPIC_API_KEY)
    svc.vector_store = VecDBStore(
        db_path=str(Path(tmp) / "vecs.npy"),
        index_path=str(Path(tmp) / "idx"),
    )
    svc.retriever = SemanticRetriever(
        encoder=svc.encoder,
        vector_store=svc.vector_store,
        metadata_store=svc.metadata_store,
    )

    # Seed a few memories
    facts = [
        "The NeuralVault project deadline is June 15th, 2026.",
        "Alice is the team lead and owns the retrieval subsystem.",
        "The IVF index uses 4096 clusters for the 20M-vector database.",
        "Bob is responsible for the ingestion pipeline (PDF, DOCX, directory).",
        "The 20M brute-force baseline was 2.58s; IVF brought it to ~0.04s.",
    ]
    svc.add_memories(facts)
    print(f"Seeded {len(facts)} memories.\n")

    # --- Turn 1 ---
    print("=" * 60)
    q1 = "Who is the team lead and when is the deadline?"
    print(f"Q1: {q1}")
    r1 = svc.chat(q1, top_k=3)
    print(f"\nA1: {r1['answer']}")
    print(f"\nMemories used: {[m['index'] for m in r1['memories_used']]}")
    print(f"Usage: {r1['usage']}")

    # --- Turn 2 (follow-up — tests conversation memory) ---
    print("\n" + "=" * 60)
    q2 = "What subsystem does that person own?"
    print(f"Q2 (follow-up): {q2}")
    r2 = svc.chat(q2, top_k=3)
    print(f"\nA2: {r2['answer']}")
    print(f"Usage: {r2['usage']}")
    # cache_read should be > 0 on turn 2 — the frozen instruction prefix is cached
    cache_read = r2["usage"].get("cache_read_input_tokens")
    print(f"\nCache read tokens on turn 2: {cache_read}")

    # --- Turn 3 (grounding check — answer not in memories) ---
    print("\n" + "=" * 60)
    q3 = "What is the company's stock price?"
    print(f"Q3 (not in memory): {q3}")
    r3 = svc.chat(q3, top_k=3)
    print(f"\nA3: {r3['answer']}")

    print("\n" + "=" * 60)
    print(f"Conversation turns retained: {svc.conversation.turn_count()}")
    print("Live test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
