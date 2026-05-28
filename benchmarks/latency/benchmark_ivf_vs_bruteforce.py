import csv
import time
from pathlib import Path

from core.memory.service import MemoryService


BASE_TEXTS = [
    "Artificial intelligence helps machines learn from data",
    "Neural networks are used in deep learning",
    "Vector databases store embeddings for semantic search",
    "Large language models use transformers",
    "Semantic memory helps AI recall information",
    "Approximate nearest neighbor search improves retrieval speed",
    "Pizza is a popular Italian food",
    "Football is played by two teams",
    "Cloud systems scale distributed applications",
    "Databases store and retrieve structured data",
]


def generate_texts(size: int) -> list[str]:
    return [
        f"{BASE_TEXTS[i % len(BASE_TEXTS)]} sample_id={i}"
        for i in range(size)
    ]


def load_memory(memory: MemoryService, texts: list[str]) -> None:
    memory.add_memories(texts)


def measure_search(memory: MemoryService, query: str, top_k: int = 5) -> tuple[float, list[dict]]:
    start = time.perf_counter()
    results = memory.search_memory(query, top_k=top_k)
    end = time.perf_counter()

    return (end - start) * 1000, results


def benchmark(size: int) -> dict:
    query = "AI semantic search using vectors"
    texts = generate_texts(size)

    brute_force = MemoryService(auto_load=False, use_ivf=False)
    ivf = MemoryService(auto_load=False, use_ivf=True)

    load_memory(brute_force, texts)
    load_memory(ivf, texts)

    # Build IVF index ONCE after insertion
    ivf.build_index()

    brute_ms, brute_results = measure_search(brute_force, query)
    ivf_ms, ivf_results = measure_search(ivf, query)

    brute_top_texts = {result["text"] for result in brute_results}
    ivf_top_texts = {result["text"] for result in ivf_results}

    overlap = len(brute_top_texts.intersection(ivf_top_texts))
    recall_at_5 = overlap / max(len(brute_top_texts), 1)

    speedup = brute_ms / ivf_ms if ivf_ms > 0 else 0

    print(f"\nDataset size: {size}")
    print(f"Brute-force latency: {brute_ms:.4f} ms")
    print(f"IVF latency: {ivf_ms:.4f} ms")
    print(f"Recall@5 vs brute force: {recall_at_5:.2f}")
    print(f"Speedup: {speedup:.2f}x")

    return {
        "dataset_size": size,
        "brute_force_latency_ms": round(brute_ms, 4),
        "ivf_latency_ms": round(ivf_ms, 4),
        "recall_at_5": round(recall_at_5, 4),
        "speedup": round(speedup, 4),
    }


def save_results(results: list[dict]) -> None:
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "ivf_vs_bruteforce.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "dataset_size",
                "brute_force_latency_ms",
                "ivf_latency_ms",
                "recall_at_5",
                "speedup",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved benchmark results to: {output_file}")


def main():
    results = []

    for size in [100, 500, 1000, 5000]:
        results.append(benchmark(size))

    save_results(results)


if __name__ == "__main__":
    main()