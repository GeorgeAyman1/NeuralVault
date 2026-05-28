import numpy as np

from core.indexing.partitioning import VectorPartitioner
from core.indexing.ivf_persistence import IVFPersistence
from core.indexing.parallel_search import ParallelPartitionSearch


class IVFIndex:
    """
    IVF-style approximate nearest-neighbor index.

    Supports both:
    - in-memory list[np.ndarray]
    - disk-backed np.memmap / np.ndarray matrix
    """

    def __init__(self, n_clusters: int = 8, n_probe: int = 2):
        if n_probe < 1:
            raise ValueError("n_probe must be at least 1.")

        self.partitioner = VectorPartitioner(n_clusters=n_clusters)
        self.persistence = IVFPersistence()
        self.n_probe = n_probe
        self.vectors = None
        self.is_fitted = False
        self.parallel_search = ParallelPartitionSearch(max_workers=4)

    def build(self, vectors) -> None:
        if vectors is None:
            raise ValueError("Vectors cannot be None.")

        if isinstance(vectors, list):
            if not vectors:
                raise ValueError("Vectors list cannot be empty.")

            self.vectors = vectors
            fit_vectors = vectors

        else:
            if vectors.shape[0] == 0:
                raise ValueError("Vector matrix cannot be empty.")

            self.vectors = vectors
            fit_vectors = [
                vectors[index]
                for index in range(vectors.shape[0])
            ]

        self.partitioner.fit(fit_vectors)
        self.is_fitted = True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        if not self.is_fitted:
            raise ValueError("IVF index has not been built yet.")

        candidate_cluster_ids = self.partitioner.nearest_clusters(
            query_vector=query_vector,
            n_probe=self.n_probe,
        )

        candidate_indices = []

        for cluster_id in candidate_cluster_ids:
            candidate_indices.extend(
                self.partitioner.get_partition(cluster_id)
            )

        if not candidate_indices:
            return []

        return self.parallel_search.search(
            vectors=self.vectors,
            candidate_indices=candidate_indices,
            query_vector=query_vector,
            top_k=top_k,
        )

    def _get_vector(self, index: int) -> np.ndarray:
        if isinstance(self.vectors, list):
            return self.vectors[index]

        return self.vectors[index]
    
    def save(self) -> None:
        if not self.is_fitted:
            raise ValueError("Cannot save IVF index before it is built.")

        self.persistence.save(
            centroids=self.partitioner.centroids,
            partitions=self.partitioner.partitions,
        )


    def load(self, vectors) -> None:
        if vectors is None:
            raise ValueError("Vectors cannot be None.")

        centroids, partitions = self.persistence.load()

        self.vectors = vectors
        self.partitioner.centroids = centroids
        self.partitioner.partitions = partitions
        self.is_fitted = True