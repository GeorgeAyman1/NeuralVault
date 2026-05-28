from pathlib import Path

import numpy as np


class VectorStore:
    """
    Vector storage for Phase 1.

    Stores normalized embeddings in memory and supports save/load using NumPy.
    """

    def __init__(self):
        self.vectors: list[np.ndarray] = []

    def add(self, vector: np.ndarray) -> int:
        if vector.ndim != 1:
            raise ValueError("Vector must be 1-dimensional.")

        self.vectors.append(vector.astype(np.float32))
        return len(self.vectors) - 1

    def get(self, index: int) -> np.ndarray:
        return self.vectors[index]

    def count(self) -> int:
        return len(self.vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        if not self.vectors:
            return []

        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1-dimensional.")

        matrix = np.vstack(self.vectors)
        scores = matrix @ query_vector

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (int(index), float(scores[index]))
            for index in top_indices
        ]

    def save(self, path: str = "data/embeddings/vectors.npy") -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.vectors:
            np.save(save_path, np.empty((0,), dtype=np.float32))
            return

        matrix = np.vstack(self.vectors).astype(np.float32)
        np.save(save_path, matrix)

    def load(self, path: str = "data/embeddings/vectors.npy") -> None:
        load_path = Path(path)

        if not load_path.exists():
            self.vectors = []
            return

        matrix = np.load(load_path)

        if matrix.size == 0:
            self.vectors = []
            return

        self.vectors = [
            matrix[i].astype(np.float32)
            for i in range(matrix.shape[0])
        ]