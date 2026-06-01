"""
Benchmark VecDB brute-force block-scan at different DB sizes and block sizes.

Measures:
  - avg / p50 / p95 retrieve latency
  - recall@5 vs exact search
  - RAM delta (requires psutil: pip install psutil)

Results → benchmarks/results/vecdb_block_benchmark.csv
"""
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vec_db import VecDB

try:
    import os
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = ROOT / "data" / "tmp_benchmark"
TMP_DIR.mkdir(parents=True, exist_ok=True)

DB_SIZES   = [10_000, 100_000, 500_000, 1_000_000]
BLOCK_SIZES = [10_000, 50_000, 100_000, 250_000, 500_000]
DIM        = 100
N_QUERIES  = 20
TOP_K      = 5
SEED       = 42


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1.0, norms)


def gen_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _normalize(rng.standard_normal((n, dim)).astype(np.float32))


def exact_top_k(vectors: np.ndarray, query: np.ndarray, k: int) -> set[int]:
    scores = vectors @ query
    return set(int(i) for i in np.argpartition(scores, -k)[-k:])


def ram_mb() -> float:
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return 0.0


def percentile(data: list[float], p: int) -> float:
    idx = max(0, int(len(data) * p / 100) - 1)
    return sorted(data)[idx]


def main():
    rows: list[dict] = []

    for db_size in DB_SIZES:
        print(f"\n{'='*60}")
        print(f"DB size: {db_size:,}")

        vectors = gen_vectors(db_size, DIM, SEED)
        queries = gen_vectors(N_QUERIES, DIM, SEED + 1)

        print(f"  Computing exact ground truth ...")
        ground_truth = [exact_top_k(vectors, q, TOP_K) for q in queries]

        db_path = TMP_DIR / f"db_{db_size}.npy"
        np.save(db_path, vectors)
        del vectors  # free RAM; VecDB will mmap

        for block_size in BLOCK_SIZES:
            ram_before = ram_mb()
            db = VecDB(database_file_path=str(db_path), block_size=block_size)

            # warmup
            db.retrieve(queries[0], TOP_K)

            latencies: list[float] = []
            recalls: list[float] = []

            for i, q in enumerate(queries):
                t0 = time.perf_counter()
                result = db.retrieve(q, TOP_K)
                latencies.append((time.perf_counter() - t0) * 1000)
                recalls.append(len(set(result) & ground_truth[i]) / TOP_K)

            ram_after = ram_mb()

            row = {
                "db_size":        db_size,
                "block_size":     block_size,
                "avg_ms":         round(sum(latencies) / len(latencies), 2),
                "p50_ms":         round(percentile(latencies, 50), 2),
                "p95_ms":         round(percentile(latencies, 95), 2),
                "recall_at_5":    round(sum(recalls) / len(recalls), 3),
                "ram_delta_mb":   round(ram_after - ram_before, 1) if HAS_PSUTIL else "N/A",
            }
            rows.append(row)
            print(
                f"  block={block_size:>8,}  "
                f"avg={row['avg_ms']:6.1f}ms  "
                f"p95={row['p95_ms']:6.1f}ms  "
                f"recall={row['recall_at_5']:.3f}  "
                f"ram_delta={row['ram_delta_mb']}MB"
            )

        # Close mmap before unlinking (required on Windows)
        del db
        gc.collect()
        try:
            db_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # file still locked; will be overwritten next run

    out = RESULTS_DIR / "vecdb_block_benchmark.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
