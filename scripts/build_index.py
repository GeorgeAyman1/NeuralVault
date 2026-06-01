"""
Build a compact IVF index for VecDB from a vector database.

The index format (CSR layout) is stored in a directory:
  centroids.npy          — (n_clusters, dim) float32, unit-normalized
  partition_offsets.npy  — (n_clusters+1,) int64, start/end of each partition
  partition_ids.npy      — (n_total,) int64, all vector IDs concatenated

Supports .npy files and raw binary .dat files (float32, shape N x dim).

Usage:
    python scripts/build_index.py --db-path data/vectors.npy \
        --index-dir data/indexes/my_index

    # Raw binary .dat file (ADB format):
    python scripts/build_index.py --db-path OpenSubtitles_en_1M_emb_64.dat \
        --index-dir ivf_1m --dim 64 --db-size 1000000 \
        --n-clusters 4096 --sample-size 500000

    # Large DB recommended settings:
    python scripts/build_index.py --db-path data/vectors.npy \
        --index-dir data/indexes/my_index \
        --n-clusters 4096 --sample-size 500000 --block-size 200000
"""
import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def build_index(
    db_path: Path,
    index_dir: Path,
    n_clusters: int = 4096,
    sample_size: int = 500_000,
    block_size: int = 100_000,
    seed: int = 42,
    dim: int = 64,
    db_size: int | None = None,
) -> None:
    print(f"Loading vectors from {db_path} ...")
    if db_path.suffix in (".npy", ".npz"):
        vectors = np.load(db_path, mmap_mode="r")
        if db_size is not None:
            vectors = vectors[:db_size]
        n, dim = vectors.shape
    else:
        # Raw binary memmap (e.g. .dat) — float32, shape (N, dim)
        if db_size is None:
            file_bytes = db_path.stat().st_size
            db_size = file_bytes // (dim * 4)
        n = db_size
        vectors = np.memmap(db_path, dtype="float32", mode="r", shape=(n, dim))
    print(f"  {n:,} vectors  dim={dim}  dtype={vectors.dtype}")

    # --- Sample for KMeans training ---
    actual_sample = min(sample_size, n)
    n_clusters = min(n_clusters, n)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=actual_sample, replace=False)
    sample = vectors[sample_idx].astype(np.float32)

    # Normalize sample so cosine ≡ dot product
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    sample /= norms

    # --- Train MiniBatchKMeans ---
    print(f"\nTraining MiniBatchKMeans  n_clusters={n_clusters}  sample={actual_sample:,} ...")
    t0 = time.perf_counter()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(10_000, actual_sample),
        n_init=3,
        max_iter=100,
        verbose=0,
    )
    kmeans.fit(sample)

    centroids = kmeans.cluster_centers_.astype(np.float32)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_norms = np.where(centroid_norms == 0, 1.0, centroid_norms)
    centroids /= centroid_norms  # unit-normalize for dot product assignment

    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # --- Assign all vectors in blocks ---
    print(f"\nAssigning {n:,} vectors in blocks of {block_size:,} ...")
    partition_lists: list[list[int]] = [[] for _ in range(n_clusters)]
    t0 = time.perf_counter()
    prev_log = 0

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = vectors[start:end].astype(np.float32)

        block_norms = np.linalg.norm(block, axis=1, keepdims=True)
        block_norms = np.where(block_norms == 0, 1.0, block_norms)
        block /= block_norms

        # (block_size, n_clusters) — nearest centroid = argmax of dot product
        scores = block @ centroids.T
        labels = np.argmax(scores, axis=1)

        for local_i, cid in enumerate(labels):
            partition_lists[int(cid)].append(start + local_i)

        pct = end / n * 100
        if pct - prev_log >= 10:
            elapsed = time.perf_counter() - t0
            print(f"  {pct:.0f}%  ({end:,}/{n:,})  {elapsed:.1f}s")
            prev_log = pct

    print(f"  Assignment done in {time.perf_counter() - t0:.1f}s")

    # --- Build CSR arrays ---
    counts = np.array([len(p) for p in partition_lists], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    non_empty = [np.array(p, dtype=np.int64) for p in partition_lists if p]
    ids = np.concatenate(non_empty) if non_empty else np.array([], dtype=np.int64)

    # --- Save ---
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "centroids.npy", centroids)
    np.save(index_dir / "partition_offsets.npy", offsets)
    np.save(index_dir / "partition_ids.npy", ids)

    print(f"\nIndex saved to: {index_dir}/")
    print(f"  centroids.npy          {centroids.shape}  {centroids.nbytes / 1e6:.1f} MB")
    print(f"  partition_offsets.npy  {offsets.shape}")
    print(f"  partition_ids.npy      {ids.shape}  {ids.nbytes / 1e6:.1f} MB")

    sizes = [len(p) for p in partition_lists]
    n_empty = sum(1 for s in sizes if s == 0)
    print(f"\nPartition stats:")
    print(f"  non-empty: {n_clusters - n_empty}/{n_clusters}")
    print(f"  avg size:  {np.mean(sizes):.0f}")
    print(f"  min/max:   {min(sizes)}/{max(sizes)}")


def main():
    parser = argparse.ArgumentParser(description="Build IVF index for VecDB")
    parser.add_argument("--db-path", required=True, help="Path to .npy vector database")
    parser.add_argument("--index-dir", required=True, help="Directory to write index files")
    parser.add_argument("--n-clusters", type=int, default=4096)
    parser.add_argument("--sample-size", type=int, default=500_000)
    parser.add_argument("--block-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=64, help="Vector dimension (for raw binary files)")
    parser.add_argument("--db-size", type=int, default=None, help="Number of vectors to load (for raw binary files)")
    args = parser.parse_args()

    build_index(
        db_path=Path(args.db_path),
        index_dir=Path(args.index_dir),
        n_clusters=args.n_clusters,
        sample_size=args.sample_size,
        block_size=args.block_size,
        seed=args.seed,
        dim=args.dim,
        db_size=args.db_size,
    )


if __name__ == "__main__":
    main()
