"""
Benchmark VecDB with IVF index vs brute-force at multiple DB sizes.

For each DB size, this script:
  1. Generates synthetic unit-normalized vectors
  2. Builds a VecDB IVF index (CSR format)
  3. Benchmarks brute-force retrieve
  4. Benchmarks IVF retrieve at several n_probe values
  5. Reports latency, recall@5, and speedup

Results → benchmarks/results/vecdb_ivf_benchmark.csv

n_clusters is set to sqrt(db_size) per the standard IVF rule of thumb.
"""
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vec_db import VecDB

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = ROOT / "data" / "tmp_benchmark"
TMP_DIR.mkdir(parents=True, exist_ok=True)

DB_SIZES    = [10_000, 100_000, 500_000, 1_000_000]
N_PROBE_VALUES = [4, 8, 16, 32, 64]
DIM         = 100
N_QUERIES   = 20
TOP_K       = 5
SEED        = 42


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1.0, norms)


def gen_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _normalize(rng.standard_normal((n, dim)).astype(np.float32))


def exact_top_k(vectors: np.ndarray, query: np.ndarray, k: int) -> set[int]:
    scores = vectors @ query
    return set(int(i) for i in np.argpartition(scores, -k)[-k:])


def build_ivf_index(vectors: np.ndarray, index_dir: Path, n_clusters: int) -> float:
    """Build CSR IVF index. Returns build time in seconds."""
    n, dim = vectors.shape
    n_clusters = min(n_clusters, n)

    t0 = time.perf_counter()

    sample_size = min(200_000, n)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample = vectors[sample_idx].copy()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=SEED,
        batch_size=min(10_000, sample_size),
        n_init=3,
        max_iter=100,
        verbose=0,
    )
    kmeans.fit(sample)

    centroids = kmeans.cluster_centers_.astype(np.float32)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.where(centroid_norms == 0, 1.0, centroid_norms)

    # Assign all vectors at once (fits in RAM for our benchmark sizes)
    scores = vectors @ centroids.T     # (n, n_clusters)
    labels = np.argmax(scores, axis=1) # (n,)

    partition_lists: list[list[int]] = [[] for _ in range(n_clusters)]
    for idx, cid in enumerate(labels):
        partition_lists[int(cid)].append(idx)

    counts = np.array([len(p) for p in partition_lists], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    non_empty = [np.array(p, dtype=np.int64) for p in partition_lists if p]
    ids = np.concatenate(non_empty) if non_empty else np.array([], dtype=np.int64)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "centroids.npy", centroids)
    np.save(index_dir / "partition_offsets.npy", offsets)
    np.save(index_dir / "partition_ids.npy", ids)

    return time.perf_counter() - t0


def bench_retrieve(db: VecDB, queries: np.ndarray, top_k: int) -> list[float]:
    """Return per-query latency in ms (with one warmup pass)."""
    db.retrieve(queries[0], top_k)  # warmup
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        db.retrieve(q, top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def main():
    rows: list[dict] = []

    for db_size in DB_SIZES:
        print(f"\n{'='*64}")
        print(f"DB size: {db_size:,}  dim={DIM}")

        vectors = gen_vectors(db_size, DIM, SEED)
        queries = gen_vectors(N_QUERIES, DIM, SEED + 1)

        # Ground truth
        print("  Computing exact ground truth ...")
        ground_truth = [exact_top_k(vectors, q, TOP_K) for q in queries]

        db_path   = TMP_DIR / f"db_{db_size}.npy"
        index_dir = TMP_DIR / f"index_{db_size}"
        np.save(db_path, vectors)

        # ── Brute-force baseline ──────────────────────────────────────
        db_bf  = VecDB(database_file_path=str(db_path))
        bf_lat = bench_retrieve(db_bf, queries, TOP_K)
        bf_avg = sum(bf_lat) / len(bf_lat)

        print(f"  Brute-force:  {bf_avg:7.2f}ms avg")
        rows.append({
            "db_size": db_size, "method": "brute_force",
            "n_clusters": None, "n_probe": None,
            "index_build_s": None,
            "avg_ms": round(bf_avg, 2),
            "p95_ms": round(sorted(bf_lat)[int(len(bf_lat) * 0.95)], 2),
            "recall_at_5": 1.0,
            "speedup": 1.0,
        })

        # ── Build IVF index ───────────────────────────────────────────
        n_clusters = max(8, int(db_size ** 0.5))  # sqrt(n) rule of thumb
        print(f"  Building IVF index  n_clusters={n_clusters} ...")
        build_s = build_ivf_index(vectors, index_dir, n_clusters)
        print(f"  Index built in {build_s:.1f}s")

        del vectors  # VecDB will mmap from disk

        # ── IVF at different n_probe values ───────────────────────────
        for n_probe in N_PROBE_VALUES:
            if n_probe > n_clusters:
                continue

            db_ivf = VecDB(
                database_file_path=str(db_path),
                index_file_path=str(index_dir),
                n_probe=n_probe,
            )

            ivf_lat = bench_retrieve(db_ivf, queries, TOP_K)
            ivf_avg = sum(ivf_lat) / len(ivf_lat)

            recalls = []
            for i, q in enumerate(queries):
                result = db_ivf.retrieve(q, TOP_K)
                recalls.append(len(set(result) & ground_truth[i]) / TOP_K)
            avg_recall = sum(recalls) / len(recalls)
            speedup = bf_avg / ivf_avg if ivf_avg > 0 else 0

            print(
                f"  IVF n_probe={n_probe:>3}:  "
                f"{ivf_avg:7.2f}ms avg  "
                f"recall@5={avg_recall:.3f}  "
                f"speedup={speedup:.2f}x"
            )
            rows.append({
                "db_size": db_size, "method": "ivf",
                "n_clusters": n_clusters, "n_probe": n_probe,
                "index_build_s": round(build_s, 2),
                "avg_ms": round(ivf_avg, 2),
                "p95_ms": round(sorted(ivf_lat)[int(len(ivf_lat) * 0.95)], 2),
                "recall_at_5": round(avg_recall, 3),
                "speedup": round(speedup, 2),
            })

        # Release mmaps before unlinking (required on Windows)
        del db_bf
        if "db_ivf" in dir():
            del db_ivf
        gc.collect()
        for path in [db_path, *index_dir.glob("*.npy")]:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                pass
        try:
            index_dir.rmdir()
        except (PermissionError, OSError):
            pass

    out = RESULTS_DIR / "vecdb_ivf_benchmark.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
