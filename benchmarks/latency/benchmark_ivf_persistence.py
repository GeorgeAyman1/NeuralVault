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


def benchmark(size: int) -> dict:
    texts = generate_texts(size)

    memory = MemoryService(auto_load=False, use_ivf=True)
    memory.add_memories(texts)

    build_start = time.perf_counter()
    memory.build_index()
    build_end = time.perf_counter()

    save_start = time.perf_counter()
    memory.retriever.ivf_index.save()
    save_end = time.perf_counter()

    load_start = time.perf_counter()
    memory.retriever.ivf_index.load(memory.vector_store.vectors)
    load_end = time.perf_counter()

    build_ms = (build_end - build_start) * 1000
    save_ms = (save_end - save_start) * 1000
    load_ms = (load_end - load_start) * 1000

    print(f"\nDataset size: {size}")
    print(f"Build IVF: {build_ms:.4f} ms")
    print(f"Save IVF: {save_ms:.4f} ms")
    print(f"Load IVF: {load_ms:.4f} ms")

    return {
        "dataset_size": size,
        "build_ivf_ms": round(build_ms, 4),
        "save_ivf_ms": round(save_ms, 4),
        "load_ivf_ms": round(load_ms, 4),
    }


def save_results(results: list[dict]) -> None:
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "ivf_persistence_benchmark.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "dataset_size",
                "build_ivf_ms",
                "save_ivf_ms",
                "load_ivf_ms",
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