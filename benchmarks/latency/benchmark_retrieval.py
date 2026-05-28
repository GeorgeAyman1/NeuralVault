import csv
import time
from pathlib import Path

from core.memory.service import MemoryService


BASE_TEXTS = [
    "Artificial intelligence helps machines learn from data",
    "Neural networks are used in deep learning",
    "Vector databases store embeddings for semantic search",
    "Pizza is a popular Italian food",
    "Football is played by two teams",
    "Large language models use transformers",
    "Semantic memory helps AI recall information",
    "Cloud systems scale distributed applications",
    "Databases store and retrieve structured data",
    "Approximate nearest neighbor search improves retrieval speed",
]


def generate_texts(size: int) -> list[str]:
    return [
        f"{BASE_TEXTS[i % len(BASE_TEXTS)]} sample_id={i}"
        for i in range(size)
    ]


def run_benchmark(size: int, query: str = "AI semantic search using vectors"):
    memory = MemoryService(auto_load=False)

    texts = generate_texts(size)

    insert_start = time.perf_counter()

    for text in texts:
        memory.add_memory(text)

    insert_end = time.perf_counter()

    search_start = time.perf_counter()
    results = memory.search_memory(query, top_k=5)
    search_end = time.perf_counter()

    insert_ms = (insert_end - insert_start) * 1000
    search_ms = (search_end - search_start) * 1000

    print(f"\nDataset size: {size}")
    print(f"Insert latency: {insert_ms:.2f} ms")
    print(f"Search latency: {search_ms:.4f} ms")
    print(f"Top result: {results[0]['text'] if results else 'None'}")

    return {
        "dataset_size": size,
        "insert_latency_ms": round(insert_ms, 4),
        "search_latency_ms": round(search_ms, 4),
    }


def save_results(results: list[dict]):
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "retrieval_benchmark.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "dataset_size",
                "insert_latency_ms",
                "search_latency_ms",
            ],
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"\nSaved benchmark results to: {output_file}")


def main():
    benchmark_results = []

    for size in [10, 100, 500, 1000]:
        result = run_benchmark(size)
        benchmark_results.append(result)

    save_results(benchmark_results)


if __name__ == "__main__":
    main()