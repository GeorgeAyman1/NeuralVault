from sentence_transformers import SentenceTransformer
import numpy as np


class TextEncoder:
    """
    Converts text into normalized vector embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Texts list cannot be empty.")

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)