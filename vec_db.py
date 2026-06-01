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
        n_probe: int = 16,
        min_candidates: int = 0,
    ):
        self.database_file_path = Path(database_file_path)
        self.index_file_path = Path(index_file_path) if index_file_path else None
        self.new_db = new_db
        self.db_size = db_size
        self.block_size = block_size
        self.n_probe = n_probe
        self.min_candidates = min_candidates

        self.vectors = self._load_vectors()
        self.ivf_index = self._load_ivf_index()

    def _load_vectors(self) -> np.ndarray:
        if not self.database_file_path.exists():
            raise FileNotFoundError(
                f"Database file not found: {self.database_file_path}"
            )

        if self.database_file_path.suffix == ".npy":
            vectors = np.load(self.database_file_path, mmap_mode="r")
        else:
            if self.db_size is None:
                raise ValueError(
                    "db_size is required when loading raw memmap vector files."
                )
            vectors = np.memmap(
                self.database_file_path,
                dtype=np.float32,
                mode="r",
            )

        if self.db_size is not None:
            vectors = vectors[: self.db_size]

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
            "centroids": np.load(centroids_path),   # (n_clusters, dim) float32
            "offsets": np.load(offsets_path),        # (n_clusters+1,) int64
            "ids": np.load(ids_path),                # (total_assigned,) int64
        }

    def retrieve(self, query_vector, top_k: int) -> list[int]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)

        if self.ivf_index is not None:
            return self._retrieve_ivf(query, top_k)

        return self._retrieve_brute_force(query, top_k)

    def _retrieve_ivf(self, query: np.ndarray, top_k: int) -> list[int]:
        centroids = self.ivf_index["centroids"]
        offsets = self.ivf_index["offsets"]
        ids = self.ivf_index["ids"]

        # Find n_probe nearest centroids by dot product
        centroid_scores = centroids @ query
        n_probe = min(self.n_probe, len(centroids))
        nearest = np.argpartition(centroid_scores, -n_probe)[-n_probe:]

        # Gather candidate IDs from selected partitions (CSR lookup).
        # Expand n_probe if we haven't hit min_candidates yet.
        expanded_probe = n_probe
        parts = [ids[offsets[c]: offsets[c + 1]] for c in nearest]
        if self.min_candidates > 0:
            candidate_count = sum(len(p) for p in parts)
            while candidate_count < self.min_candidates and expanded_probe < len(centroids):
                expanded_probe += 1
                extra = int(np.argpartition(centroid_scores, -expanded_probe)[-expanded_probe])
                extra_ids = ids[offsets[extra]: offsets[extra + 1]]
                parts.append(extra_ids)
                candidate_count += len(extra_ids)

        if not parts or all(len(p) == 0 for p in parts):
            return self._retrieve_brute_force(query, top_k)

        candidates = np.concatenate(parts)
        if len(candidates) == 0:
            return self._retrieve_brute_force(query, top_k)

        # Sort candidate IDs for sequential memmap access (avoids random I/O)
        sort_order = np.argsort(candidates)
        sorted_candidates = candidates[sort_order]

        candidate_vecs = self.vectors[sorted_candidates]
        scores = np.dot(candidate_vecs, query)

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
            block = self.vectors[start:end]
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
