from pathlib import Path

import numpy as np


class MMapVectorStorage:
    """
    Disk-backed vector storage using NumPy memmap.

    This allows large vector matrices to be accessed without loading
    the full array into RAM.
    """

    def __init__(
        self,
        path: str = "data/indexes/vectors.mmap",
        shape_path: str = "data/indexes/vectors_shape.npy",
        dtype=np.float32,
    ):
        self.path = Path(path)
        self.shape_path = Path(shape_path)
        self.dtype = dtype

        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, vectors: list[np.ndarray]) -> None:
        if not vectors:
            raise ValueError("Vectors list cannot be empty.")

        matrix = np.vstack(vectors).astype(self.dtype)
        shape = np.array(matrix.shape, dtype=np.int64)

        mmap = np.memmap(
            self.path,
            dtype=self.dtype,
            mode="w+",
            shape=matrix.shape,
        )

        mmap[:] = matrix[:]
        mmap.flush()

        np.save(self.shape_path, shape)

    def load(self) -> np.memmap:
        if not self.path.exists():
            raise FileNotFoundError(f"MMap file not found: {self.path}")

        if not self.shape_path.exists():
            raise FileNotFoundError(f"Shape file not found: {self.shape_path}")

        shape = tuple(np.load(self.shape_path))

        return np.memmap(
            self.path,
            dtype=self.dtype,
            mode="r",
            shape=shape,
        )

    def exists(self) -> bool:
        return self.path.exists() and self.shape_path.exists()