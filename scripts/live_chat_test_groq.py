"""
Live end-to-end test of chat() against a FREE Groq LLM (OpenAI-compatible).

Requires GROQ_API_KEY in the environment (free at https://console.groq.com).
Uses an isolated temp store so it does not touch your real data/ directory.

Run:
    # PowerShell:  $env:GROQ_API_KEY="gsk-..."; python scripts/live_chat_test_groq.py
    # bash:        GROQ_API_KEY=gsk-... python scripts/live_chat_test_groq.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from core.llm.llm_client import LLMClient
from core.memory.service import MemoryService
from core.storage.vecdb_store import VecDBStore
from core.indexing.retrieval import SemanticRetriever

MODEL = os.environ.get("NEURALVAULT_LLM_MODEL", "llama-3.3-70b-versatile")


def chat_with_retry(svc, query, top_k=3, retries=4):
    """Call chat(), backing off on Groq free-tier 429 rate limits."""
    for attempt in range(retries):
        try:
            return svc.chat(query, top_k=top_k)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 429 or attempt == retries - 1:
                raise
            wait = float(e.response.headers.get("retry-after", 2 ** attempt))
            print(f"  (rate-limited, retrying in {wait:.0f}s...)")
            time.sleep(wait)


def main() -> int:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("ERROR: GROQ_API_KEY is not set in the environment.")
        return 1

    tmp = tempfile.mkdtemp()
    groq = LLMClient(provider="groq", api_key=key, model=MODEL)
    svc = MemoryService(auto_load=False, llm_client=groq)
    svc.vector_store = VecDBStore(
        db_path=str(Path(tmp) / "vecs.npy"),
        index_path=str(Path(tmp) / "idx"),
    )
    svc.retriever = SemanticRetriever(
        encoder=svc.encoder,
        vector_store=svc.vector_store,
        metadata_store=svc.metadata_store,
    )

    facts = [
        "The NeuralVault project deadline is June 15th, 2026.",
        "Alice is the team lead and owns the retrieval subsystem.",
        "The IVF index uses 4096 clusters for the 20M-vector database.",
        "Bob is responsible for the ingestion pipeline (PDF, DOCX, directory).",
        "The 20M brute-force baseline was 2.58s; IVF brought it to ~0.04s.",
    ]
    svc.add_memories(facts)
    print(f"Provider: groq | model: {MODEL}")
    print(f"Seeded {len(facts)} memories.\n")

    print("=" * 60)
    q1 = "Who is the team lead and when is the deadline?"
    print(f"Q1: {q1}")
    r1 = chat_with_retry(svc, q1)
    print(f"\nA1: {r1['answer']}")
    print(f"\nMemories used: {[m['index'] for m in r1['memories_used']]}")
    print(f"Usage: {r1['usage']}")
    time.sleep(2)  # ease the free-tier rate limit between turns

    print("\n" + "=" * 60)
    q2 = "What subsystem does that person own?"
    print(f"Q2 (follow-up): {q2}")
    r2 = chat_with_retry(svc, q2)
    print(f"\nA2: {r2['answer']}")
    time.sleep(2)

    print("\n" + "=" * 60)
    q3 = "What is the company's stock price?"
    print(f"Q3 (not in memory): {q3}")
    r3 = chat_with_retry(svc, q3)
    print(f"\nA3: {r3['answer']}")

    print("\n" + "=" * 60)
    print(f"Conversation turns retained: {svc.conversation.turn_count()}")
    print("Live Groq test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
