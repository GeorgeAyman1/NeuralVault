from pathlib import Path

import numpy as np


class VecDB:
    def __init__(
        self,
        database_file_path: str,
        index_file_path: str | None = None,
        new_db: bool = False,
        db_size: int | None = None,
        block_size: int = 100_000,
    ):
        self.database_file_path = Path(database_file_path)
        self.index_file_path = Path(index_file_path) if index_file_path else None
        self.new_db = new_db
        self.db_size = db_size
        self.block_size = block_size

        self.vectors = self._load_vectors()

    def _load_vectors(self):
        if not self.database_file_path.exists():
            raise FileNotFoundError(
                f"Database file not found: {self.database_file_path}"
            )

        if self.database_file_path.suffix == ".npy":
            vectors = np.load(
                self.database_file_path,
                mmap_mode="r",
            )
        else:
            vectors = np.memmap(
                self.database_file_path,
                dtype=np.float32,
                mode="r",
            )

            if self.db_size is None:
                raise ValueError(
                    "db_size is required when loading raw memmap vector files."
                )

        if self.db_size is not None:
            vectors = vectors[: self.db_size]

        return vectors

    def retrieve(self, query_vector, top_k: int):
        query = np.asarray(query_vector, dtype=np.float32)

        if query.ndim > 1:
            query = query.reshape(-1)

        block_size = self.block_size
        best_scores = np.full(top_k, -np.inf, dtype=np.float32)
        best_indices = np.full(top_k, -1, dtype=np.int64)

        total_vectors = self.vectors.shape[0]

        for start in range(0, total_vectors, block_size):
            end = min(start + block_size, total_vectors)

            block = self.vectors[start:end]
            scores = block @ query

            local_top = np.argpartition(scores, -top_k)[-top_k:]
            local_scores = scores[local_top]
            local_indices = local_top + start

            combined_scores = np.concatenate([best_scores, local_scores])
            combined_indices = np.concatenate([best_indices, local_indices])

            top = np.argpartition(combined_scores, -top_k)[-top_k:]

            best_scores = combined_scores[top]
            best_indices = combined_indices[top]

        order = np.argsort(best_scores)[::-1]
        best_indices = best_indices[order]

        return [int(index) for index in best_indices]