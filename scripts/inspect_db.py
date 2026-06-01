"""
Inspect an ADB-format vector database before building an index.

Prints shape, dtype, memory footprint, normalization status, and
recommended IVF settings (n_clusters, n_probe, expected latency).

Usage:
    python scripts/inspect_db.py --db-path data/raw/database_file.npy
    python scripts/inspect_db.py --db-path data/raw/database_file.npy --sample 5000
"""
import argparse
import time
from pathlib import Path

import numpy as np


def recommend_ivf(n: int, dim: int) -> None:
    print("\n--- IVF Recommendations ---")

    configs = [
        (512,   16,  32),
        (1024,  16,  32),
        (2048,  16,  32),
        (4096,  32,  64),
        (8192,  32,  64),
    ]

    print(f"{'n_clusters':>12} {'n_probe':>8} {'candidates':>12} {'pct_scanned':>12}  notes")
    for n_clusters, n_probe_lo, n_probe_hi in configs:
        if n_clusters > n:
            continue
        cluster_size = n / n_clusters
        cands_lo = int(n_probe_lo * cluster_size)
        cands_hi = int(n_probe_hi * cluster_size)
        pct_lo = cands_lo / n * 100
        pct_hi = cands_hi / n * 100
        note = ""
        if pct_hi > 10:
            note = "⚠ many candidates"
        elif pct_lo < 0.1:
            note = "⚠ too few candidates (low recall risk)"
        print(
            f"{n_clusters:>12,} {n_probe_lo:>4}–{n_probe_hi:<2}  "
            f"{cands_lo:>6,}–{cands_hi:<6,}  {pct_lo:.2f}%–{pct_hi:.2f}%  {note}"
        )

    print()
    best_nc = min(4096, max(256, int(4 * n ** 0.5)))
    best_nc = min(best_nc, n)
    print(f"Suggested starting point:  --n-clusters {best_nc}  n_probe=32")
    print(
        f"  Build command:\n"
        f"    python scripts/build_index.py \\\n"
        f"        --db-path <your_db.npy> \\\n"
        f"        --index-dir data/indexes/adb_ivf \\\n"
        f"        --n-clusters {best_nc} \\\n"
        f"        --sample-size {min(500_000, n):,}"
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect ADB vector database")
    parser.add_argument("--db-path", required=True, help="Path to .npy vector database")
    parser.add_argument(
        "--sample", type=int, default=10_000,
        help="Number of vectors to sample for statistics (default: 10000)"
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: file not found: {db_path}")
        return

    print(f"Loading (mmap) {db_path} ...")
    t0 = time.perf_counter()
    vectors = np.load(db_path, mmap_mode="r")
    load_ms = (time.perf_counter() - t0) * 1000

    n, dim = vectors.shape
    dtype = vectors.dtype
    size_gb = vectors.nbytes / 1e9

    print(f"\n--- Database Info ---")
    print(f"  Shape:         {vectors.shape}")
    print(f"  Dtype:         {dtype}")
    print(f"  Size on disk:  {size_gb:.2f} GB  ({vectors.nbytes / 1e6:.0f} MB)")
    print(f"  Mmap open:     {load_ms:.1f} ms")

    # Sample-based statistics
    sample_size = min(args.sample, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=sample_size, replace=False)

    print(f"\nSampling {sample_size:,} vectors for statistics ...")
    sample = vectors[idx].astype(np.float32)

    norms = np.linalg.norm(sample, axis=1)
    norm_mean = float(np.mean(norms))
    norm_std = float(np.std(norms))
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))

    val_mean = float(np.mean(sample))
    val_std = float(np.std(sample))
    val_min = float(np.min(sample))
    val_max = float(np.max(sample))

    print(f"\n--- Vector Statistics ---")
    print(f"  Norms:         mean={norm_mean:.4f}  std={norm_std:.4f}  "
          f"min={norm_min:.4f}  max={norm_max:.4f}")
    print(f"  Values:        mean={val_mean:.4f}  std={val_std:.4f}  "
          f"min={val_min:.4f}  max={val_max:.4f}")

    unit_norm = abs(norm_mean - 1.0) < 0.01 and norm_std < 0.01
    print(f"  Normalized:    {'YES (unit norm)' if unit_norm else 'NO — build_index.py will normalize before clustering'}")

    # Retrieval latency estimate (brute-force)
    query = sample[0]
    trials = 5
    t0 = time.perf_counter()
    for _ in range(trials):
        _ = vectors[:min(n, 1_000_000)] @ query
    bf_1m_ms = (time.perf_counter() - t0) / trials * 1000
    bf_full_ms = bf_1m_ms * (n / min(n, 1_000_000))

    print(f"\n--- Brute-Force Estimates ---")
    print(f"  At 1M vectors:      ~{bf_1m_ms:.1f} ms/query")
    print(f"  At {n:,} vectors:  ~{bf_full_ms:.1f} ms/query")

    recommend_ivf(n, dim)


if __name__ == "__main__":
    main()
