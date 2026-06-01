"""
IVF recall benchmark on CLUSTERED (realistic) data.

Synthetic random vectors are the worst case for IVF — no cluster structure.
Real sentence embeddings cluster around semantic topics. This benchmark
simulates that by generating vectors from a fixed set of Gaussian cluster
centers with small noise, which approximates actual embedding distributions.

Tests recall@5 and latency at multiple (n_clusters, n_probe) combinations.

Results -> benchmarks/results/ivf_recall_benchmark.csv
"""
import csv
import gc
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from vec_db import VecDB

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DB_SIZES        = [100_000, 500_000, 1_000_000]
N_TRUE_CLUSTERS = 200     # simulates distinct semantic topics
NOISE           = 0.20    # intra-cluster spread (0 = perfect clusters, 1 = near-random)
DIM             = 100
N_QUERIES       = 50
TOP_K           = 5
SEED            = 42

# (n_clusters, n_probe) pairs to test
GRID = [
    (256,  4), (256,  8), (256, 16), (256, 32),
    (512,  4), (512,  8), (512, 16), (512, 32),
    (1024, 4), (1024, 8), (1024, 16), (1024, 32),
    (2048, 4), (2048, 8), (2048, 16), (2048, 32),
]


def gen_clustered(n: int, dim: int, n_true: int, noise: float, seed: int) -> np.ndarray:
    """Gaussian mixture on the unit sphere — simulates semantic embeddings."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_true, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    vecs = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        c = centers[i % n_true]
        vecs[i] = c + rng.standard_normal(dim).astype(np.float32) * noise

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def exact_top_k(vecs: np.ndarray, q: np.ndarray, k: int) -> set[int]:
    scores = vecs @ q
    return set(int(i) for i in np.argpartition(scores, -k)[-k:])


def build_index(vecs: np.ndarray, n_clusters: int, index_dir: Path) -> float:
    n, dim = vecs.shape
    nc = min(n_clusters, n)
    t0 = time.perf_counter()

    sample_size = min(200_000, n)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample = vecs[sample_idx].copy()

    km = MiniBatchKMeans(
        n_clusters=nc, random_state=SEED,
        batch_size=min(10_000, sample_size),
        n_init=3, max_iter=100, verbose=0,
    )
    km.fit(sample)

    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    scores = vecs @ centroids.T
    labels = np.argmax(scores, axis=1)

    partitions: list[list[int]] = [[] for _ in range(nc)]
    for idx, cid in enumerate(labels):
        partitions[int(cid)].append(idx)

    counts = np.array([len(p) for p in partitions], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    non_empty = [np.array(p, dtype=np.int64) for p in partitions if p]
    ids = np.concatenate(non_empty) if non_empty else np.array([], dtype=np.int64)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "centroids.npy", centroids)
    np.save(index_dir / "partition_offsets.npy", offsets)
    np.save(index_dir / "partition_ids.npy", ids)

    return time.perf_counter() - t0


def main():
    rows: list[dict] = []

    for db_size in DB_SIZES:
        print(f"\n{'='*68}")
        print(f"DB size: {db_size:,}  dim={DIM}  noise={NOISE}  true_clusters={N_TRUE_CLUSTERS}")

        vecs = gen_clustered(db_size, DIM, N_TRUE_CLUSTERS, NOISE, SEED)
        queries = gen_clustered(N_QUERIES, DIM, N_TRUE_CLUSTERS, NOISE, SEED + 1)

        print("  Computing exact ground truth ...")
        gt = [exact_top_k(vecs, q, TOP_K) for q in queries]

        # Brute-force baseline
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "db.npy"
            np.save(db_path, vecs)

            db_bf = VecDB(str(db_path))
            db_bf.retrieve(queries[0], TOP_K)  # warmup
            bf_lat = []
            for q in queries:
                t0 = time.perf_counter()
                db_bf.retrieve(q, TOP_K)
                bf_lat.append((time.perf_counter() - t0) * 1000)
            bf_avg = sum(bf_lat) / len(bf_lat)
            del db_bf; gc.collect()

            print(f"  Brute-force: {bf_avg:.2f}ms avg  recall@5=1.000")
            rows.append({
                "db_size": db_size, "method": "brute_force",
                "n_clusters": None, "n_probe": None, "build_s": None,
                "avg_ms": round(bf_avg, 2), "recall_at_5": 1.0,
                "speedup": 1.0,
            })

            # IVF grid search
            built: dict[int, float] = {}  # n_clusters -> build_s

            for n_clusters, n_probe in GRID:
                if n_clusters > db_size // 10:
                    continue
                if n_probe > n_clusters:
                    continue

                index_dir = tmp / f"idx_{n_clusters}"
                if n_clusters not in built:
                    build_s = build_index(vecs, n_clusters, index_dir)
                    built[n_clusters] = build_s
                build_s = built[n_clusters]

                db_ivf = VecDB(str(db_path), str(index_dir), n_probe=n_probe)
                db_ivf.retrieve(queries[0], TOP_K)  # warmup

                latencies = []
                recalls = []
                for i, q in enumerate(queries):
                    t0 = time.perf_counter()
                    result = db_ivf.retrieve(q, TOP_K)
                    latencies.append((time.perf_counter() - t0) * 1000)
                    recalls.append(len(set(result) & gt[i]) / TOP_K)

                del db_ivf; gc.collect()

                avg_ms = sum(latencies) / len(latencies)
                avg_recall = sum(recalls) / len(recalls)
                speedup = bf_avg / avg_ms if avg_ms > 0 else 0

                print(
                    f"  n_clusters={n_clusters:>5}  n_probe={n_probe:>3}:  "
                    f"{avg_ms:7.2f}ms  recall@5={avg_recall:.3f}  speedup={speedup:.2f}x"
                )
                rows.append({
                    "db_size": db_size, "method": "ivf",
                    "n_clusters": n_clusters, "n_probe": n_probe,
                    "build_s": round(build_s, 2),
                    "avg_ms": round(avg_ms, 2),
                    "recall_at_5": round(avg_recall, 3),
                    "speedup": round(speedup, 2),
                })

    out = RESULTS_DIR / "ivf_recall_benchmark.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
