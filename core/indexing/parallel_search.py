from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ParallelPartitionSearch:
    """
    Scores IVF candidate partitions in parallel.
    """

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def search(
        self,
        vectors,
        candidate_indices: list[int],
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        if not candidate_indices:
            return []

        chunks = self._chunk_indices(candidate_indices)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            partial_results = list(
                executor.map(
                    lambda chunk: self._score_chunk(vectors, chunk, query_vector, top_k),
                    chunks,
                )
            )

        merged_results = [
            result
            for chunk_results in partial_results
            for result in chunk_results
        ]

        merged_results.sort(key=lambda item: item[1], reverse=True)

        return merged_results[:top_k]

    def _score_chunk(
        self,
        vectors,
        indices: list[int],
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[tuple[int, float]]:
        candidate_vectors = np.vstack([
            vectors[index]
            for index in indices
        ])

        scores = candidate_vectors @ query_vector
        top_local_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (
                int(indices[local_index]),
                float(scores[local_index]),
            )
            for local_index in top_local_indices
        ]

    def _chunk_indices(self, indices: list[int]) -> list[list[int]]:
        if self.max_workers is None or self.max_workers <= 1:
            return [indices]

        chunk_size = max(1, len(indices) // self.max_workers)

        return [
            indices[start:start + chunk_size]
            for start in range(0, len(indices), chunk_size)
        ]