import numpy as np


class MemoryConsolidator:
    """
    Finds highly similar memories that may be duplicates or related memories.

    Phase 1 behavior:
    - detects candidate memory pairs
    - does not merge automatically
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def find_candidates(
        self,
        vectors: list[np.ndarray],
        records: list[dict],
    ) -> list[dict]:
        if len(vectors) != len(records):
            raise ValueError("Vectors and records must have the same length.")

        candidates = []

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarity = float(vectors[i] @ vectors[j])

                if similarity >= self.similarity_threshold:
                    candidates.append({
                        "memory_a_index": i,
                        "memory_b_index": j,
                        "similarity": round(similarity, 4),
                        "memory_a_text": records[i]["text"],
                        "memory_b_text": records[j]["text"],
                    })

        candidates.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        return candidates