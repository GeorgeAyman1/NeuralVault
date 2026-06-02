from pathlib import Path

import numpy as np


class VecDB:
    def __init__(
        self,
        database_file_path: str,
        index_file_path: str | None = None,
        new_db: bool = False,
        db_size: int | None = None,
        block_size: int = 500_000,
        n_probe: int = 32,
        min_candidates: int = 0,
        dim: int = 64,
    ):
        self.database_file_path = Path(database_file_path)
        self.index_file_path = Path(index_file_path) if index_file_path else None
        self.new_db = new_db
        self.db_size = db_size
        self.block_size = block_size
        self.n_probe = n_probe
        self.min_candidates = min_candidates
        self.dim = dim

        self.vectors = self._load_vectors()
        self.ivf_index = self._load_ivf_index()

    def _load_vectors(self) -> np.ndarray:
        if not self.database_file_path.exists():
            raise FileNotFoundError(
                f"Database file not found: {self.database_file_path}"
            )

        if self.database_file_path.suffix == ".npy":
            vectors = np.load(self.database_file_path, mmap_mode="r")
            if self.db_size is not None:
                vectors = vectors[: self.db_size]
        else:
            # Raw binary memmap (e.g. .dat) — must specify shape explicitly
            if self.db_size is None:
                file_bytes = self.database_file_path.stat().st_size
                self.db_size = file_bytes // (self.dim * 4)
            vectors = np.memmap(
                self.database_file_path,
                dtype=np.float32,
                mode="r",
                shape=(self.db_size, self.dim),
            )

        return vectors

    def _load_ivf_index(self) -> dict | None:
        """Load CSR-format IVF index from index_file_path directory."""
        if self.index_file_path is None:
            return None

        index_dir = self.index_file_path
        centroids_path = index_dir / "centroids.npy"
        offsets_path = index_dir / "partition_offsets.npy"
        ids_path = index_dir / "partition_ids.npy"

        if not (centroids_path.exists() and offsets_path.exists() and ids_path.exists()):
            return None

        return {
            "centroids": np.load(centroids_path),              # (n_clusters, dim) float32
            "offsets": np.load(offsets_path),                  # (n_clusters+1,) int64
            "ids": np.load(ids_path).astype(np.int32, copy=False),  # (total_assigned,) int32
        }

    def retrieve(self, query_vector, top_k: int) -> list[int]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        if self.ivf_index is not None:
            return self._retrieve_ivf(query, top_k)

        return self._retrieve_brute_force(query, top_k)

    def _retrieve_ivf(self, query: np.ndarray, top_k: int) -> list[int]:
        centroids = self.ivf_index["centroids"]
        offsets = self.ivf_index["offsets"]
        ids = self.ivf_index["ids"]

        # Score all centroids (single matmul, fast for any n_clusters)
        centroid_scores = centroids @ query
        n_probe = min(self.n_probe, len(centroids))

        # Gather candidate IDs from the nearest n_probe partitions.
        # When min_candidates > 0, expand past n_probe until the floor is met.
        if self.min_candidates > 0:
            # Sort once; iterate in descending similarity order
            sorted_clusters = np.argsort(centroid_scores)[::-1]
            parts: list[np.ndarray] = []
            candidate_count = 0
            for rank, c in enumerate(sorted_clusters):
                if rank >= n_probe and candidate_count >= self.min_candidates:
                    break
                part = ids[offsets[c]: offsets[c + 1]]
                parts.append(part)
                candidate_count += len(part)
        else:
            # argpartition is faster than argsort when we don't need order
            nearest = np.argpartition(centroid_scores, -n_probe)[-n_probe:]
            parts = [ids[offsets[c]: offsets[c + 1]] for c in nearest]

        non_empty = [p for p in parts if len(p) > 0]
        if not non_empty:
            return self._retrieve_brute_force(query, top_k)

        candidates = np.concatenate(non_empty)

        # Sort candidate IDs before mmap access — turns random I/O into sequential reads
        sorted_candidates = np.sort(candidates)

        candidate_vecs = self.vectors[sorted_candidates].astype(np.float32)
        # Normalize candidates for cosine similarity
        norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        candidate_vecs /= norms
        scores = candidate_vecs @ query

        k = min(top_k, len(sorted_candidates))
        if len(sorted_candidates) > k:
            top_local = np.argpartition(scores, -k)[-k:]
            order = top_local[np.argsort(scores[top_local])[::-1]]
        else:
            order = np.argsort(scores)[::-1]

        return [int(sorted_candidates[i]) for i in order[:top_k]]

    def _retrieve_brute_force(self, query: np.ndarray, top_k: int) -> list[int]:
        block_size = self.block_size
        best_scores = np.full(top_k, -np.inf, dtype=np.float32)
        best_indices = np.full(top_k, -1, dtype=np.int64)
        total_vectors = self.vectors.shape[0]

        for start in range(0, total_vectors, block_size):
            end = min(start + block_size, total_vectors)
            block = self.vectors[start:end].astype(np.float32)
            # Normalize each vector for cosine similarity
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            block /= norms
            scores = block @ query

            k = min(top_k, len(scores))
            local_top = np.argpartition(scores, -k)[-k:]
            local_scores = scores[local_top]
            local_indices = local_top + start

            combined_scores = np.concatenate([best_scores, local_scores])
            combined_indices = np.concatenate([best_indices, local_indices])

            top = np.argpartition(combined_scores, -top_k)[-top_k:]
            best_scores = combined_scores[top]
            best_indices = combined_indices[top]

        order = np.argsort(best_scores)[::-1]
        best_indices = best_indices[order]
        return [int(i) for i in best_indices if i >= 0]
