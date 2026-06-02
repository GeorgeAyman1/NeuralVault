import numpy as np
from typing import Any


class MemorySummarizer:
    """
    Extracts a representative summary from a group of related memories.

    Sprint 1: extractive only — selects the memory closest to the group
    centroid and appends a combined context string.

    Sprint 3 will replace this with LLM-generated abstractive summaries.
    """

    def summarize(
        self,
        records: list[dict[str, Any]],
        vectors: list[np.ndarray],
    ) -> str:
        if not records:
            raise ValueError("Cannot summarize an empty list of memories.")

        if len(records) == 1:
            return records[0]["text"]

        # Compute centroid of the group
        matrix   = np.vstack([v.reshape(1, -1) for v in vectors]).astype(np.float32)
        centroid = matrix.mean(axis=0)
        norm     = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        # Pick the memory whose vector is closest to the centroid
        scores      = matrix @ centroid
        central_idx = int(np.argmax(scores))
        central_text = records[central_idx]["text"]

        # Append a compact context line from the remaining memories
        other_texts = [
            r["text"] for i, r in enumerate(records) if i != central_idx
        ]
        context = " | ".join(other_texts)

        return f"{central_text}\n[Related: {context}]"
