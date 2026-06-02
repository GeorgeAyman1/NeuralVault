import json
from pathlib import Path

import numpy as np


class IVFPersistence:
    """
    Saves and loads IVF index state.

    Stores:
    - centroids as .npy
    - partitions as .json
    """

    def __init__(
        self,
        centroids_path: str = "data/indexes/ivf_centroids.npy",
        partitions_path: str = "data/indexes/ivf_partitions.json",
    ):
        self.centroids_path = Path(centroids_path)
        self.partitions_path = Path(partitions_path)

        self.centroids_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, centroids: np.ndarray, partitions: dict[int, list[int]]) -> None:
        if centroids is None:
            raise ValueError("Centroids cannot be None.")

        np.save(self.centroids_path, centroids.astype(np.float32))

        json_partitions = {
            str(cluster_id): vector_ids
            for cluster_id, vector_ids in partitions.items()
        }

        with open(self.partitions_path, "w", encoding="utf-8") as file:
            json.dump(json_partitions, file, indent=2)

    def load(self) -> tuple[np.ndarray, dict[int, list[int]]]:
        if not self.exists():
            raise FileNotFoundError("IVF persistence files not found.")

        centroids = np.load(self.centroids_path)

        with open(self.partitions_path, "r", encoding="utf-8") as file:
            raw_partitions = json.load(file)

        partitions = {
            int(cluster_id): vector_ids
            for cluster_id, vector_ids in raw_partitions.items()
        }

        return centroids.astype(np.float32), partitions

    def exists(self) -> bool:
        return self.centroids_path.exists() and self.partitions_path.exists()
