from pathlib import Path

import numpy as np

from vec_db import VecDB
from scripts.build_index import build_index as _build_ivf_index


class VecDBStore:
    """
    Drop-in replacement for VectorStore backed by VecDB.

    Write path: vectors accumulate in an in-memory numpy matrix and are
    saved to a .npy file on every save() call.

    Read path:
    - If an IVF index has been built (via build_index()), loads VecDB once
      and uses it for all subsequent searches — IVF-backed, fast, low RAM.
    - Otherwise falls back to direct cosine similarity on the in-memory
      matrix (correct and fast for up to ~100k memories).

    The switch between brute-force and IVF is automatic and transparent
    to callers.
    """

    DEFAULT_DB_PATH    = "data/embeddings/vectors.npy"
    DEFAULT_INDEX_PATH = "data/indexes/memory_ivf"

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        index_path: str = DEFAULT_INDEX_PATH,
    ):
        self.db_path    = Path(db_path)
        self.index_path = Path(index_path)
        self._matrix: np.ndarray | None = None   # (n, dim) float32, unit-normalized
        self._vecdb:  VecDB | None = None
        self._index_valid = False  # True only when VecDB is in sync with _matrix

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def add(self, vector: np.ndarray) -> int:
        v = vector.reshape(1, -1).astype(np.float32)
        self._matrix = v if self._matrix is None else np.vstack([self._matrix, v])
        self._index_valid = False   # in-memory data ahead of any saved index
        self._vecdb = None
        return len(self._matrix) - 1

    def count(self) -> int:
        return 0 if self._matrix is None else len(self._matrix)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | None = None) -> None:
        save_path = Path(path) if path else self.db_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        matrix = self._matrix if self._matrix is not None else np.empty((0,), dtype=np.float32)
        np.save(save_path, matrix)

    def load(self, path: str | None = None) -> None:
        load_path = Path(path) if path else self.db_path

        self._matrix = None
        self._vecdb  = None
        self._index_valid = False

        if not load_path.exists():
            return

        data = np.load(load_path)
        if data.ndim < 2 or data.size == 0:
            return

        self._matrix = data.astype(np.float32)

        # Eagerly load VecDB if a valid index already exists on disk,
        # so the first search after startup uses IVF immediately.
        self._try_load_vecdb()

    # ------------------------------------------------------------------ #
    # Index building                                                       #
    # ------------------------------------------------------------------ #

    def build_index(
        self,
        n_clusters: int | None = None,
        sample_size: int = 100_000,
    ) -> None:
        if self._matrix is None or len(self._matrix) < 100:
            return

        self.save()

        n   = len(self._matrix)
        dim = self._matrix.shape[1]
        clusters = n_clusters or max(16, min(4096, int(4 * n ** 0.5)))

        _build_ivf_index(
            db_path=self.db_path,
            index_dir=self.index_path,
            n_clusters=clusters,
            sample_size=min(sample_size, n),
            block_size=min(100_000, n),
            dim=dim,
            db_size=n,
        )

        self._vecdb = None
        self._try_load_vecdb()

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        if self._matrix is None:
            return []

        q = query_vector.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(q))
        q_norm = q / norm if norm > 0 else q

        if self._index_valid and self._vecdb is not None:
            ids = self._vecdb.retrieve(q_norm, top_k)
            # Vectors from encoder are already unit-normalized → dot = cosine sim
            return [(int(i), float(self._matrix[i] @ q_norm)) for i in ids]

        # Brute-force cosine similarity on the in-memory matrix
        # (encoder normalizes embeddings so no per-vector norm needed)
        scores  = self._matrix @ q_norm
        k       = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(int(i), float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------ #
    # Compatibility shim for MemoryConsolidator (expects list[np.ndarray]) #
    # ------------------------------------------------------------------ #

    @property
    def vectors(self) -> list[np.ndarray]:
        if self._matrix is None:
            return []
        return [self._matrix[i] for i in range(len(self._matrix))]

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _try_load_vecdb(self) -> bool:
        """Load VecDB from disk if index is present and matches saved file."""
        has_index = (self.index_path / "centroids.npy").exists()
        if not has_index or not self.db_path.exists():
            return False

        try:
            saved = np.load(self.db_path, mmap_mode="r")
            if saved.ndim < 2 or saved.shape[0] == 0:
                return False

            n, dim = int(saved.shape[0]), int(saved.shape[1])

            # Only load VecDB if saved file is in sync with in-memory matrix
            if self._matrix is not None and n != len(self._matrix):
                return False

            self._vecdb = VecDB(
                database_file_path=str(self.db_path),
                index_file_path=str(self.index_path),
                db_size=n,
                dim=dim,   # inferred from saved file, not hardcoded 64
            )
            self._index_valid = True
            return True
        except Exception:
            return False
