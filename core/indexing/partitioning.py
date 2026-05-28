import numpy as np
from sklearn.cluster import KMeans


class VectorPartitioner:
    """
    Partitions vectors into clusters using KMeans.

    This is the foundation for IVF-style ANN retrieval.
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: KMeans | None = None
        self.centroids: np.ndarray | None = None
        self.partitions: dict[int, list[int]] = {}

    def fit(self, vectors: list[np.ndarray]) -> None:
        if not vectors:
            raise ValueError("Vectors list cannot be empty.")

        matrix = np.vstack(vectors).astype(np.float32)

        actual_clusters = min(self.n_clusters, len(vectors))

        self.kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=self.random_state,
            n_init="auto",
        )

        labels = self.kmeans.fit_predict(matrix)

        self.centroids = self.kmeans.cluster_centers_.astype(np.float32)
        self.partitions = {cluster_id: [] for cluster_id in range(actual_clusters)}

        for vector_index, cluster_id in enumerate(labels):
            self.partitions[int(cluster_id)].append(vector_index)

    def assign(self, vector: np.ndarray) -> int:
        if self.kmeans is None:
            raise ValueError("Partitioner has not been fitted yet.")

        if vector.ndim != 1:
            raise ValueError("Vector must be 1-dimensional.")

        label = self.kmeans.predict(vector.reshape(1, -1))[0]
        return int(label)

    def nearest_clusters(self, query_vector: np.ndarray, n_probe: int = 1) -> list[int]:
        if self.centroids is None:
            raise ValueError("Partitioner has not been fitted yet.")

        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1-dimensional.")

        if n_probe < 1:
            raise ValueError("n_probe must be at least 1.")

        scores = self.centroids @ query_vector
        top_clusters = np.argsort(scores)[::-1][:n_probe]

        return [int(cluster_id) for cluster_id in top_clusters]

    def get_partition(self, cluster_id: int) -> list[int]:
        return self.partitions.get(cluster_id, [])

    def count_partitions(self) -> int:
        return len(self.partitions)