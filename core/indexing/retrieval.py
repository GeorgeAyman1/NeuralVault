from core.embeddings.encoder import TextEncoder
from core.storage.vector_store import VectorStore
from core.storage.metadata_store import MetadataStore


class SemanticRetriever:
    """
    Handles semantic retrieval using vector similarity search.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ):
        self.encoder = encoder
        self.vector_store = vector_store
        self.metadata_store = metadata_store

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_vector = self.encoder.encode(query)

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        output = []

        for index, score in results:
            metadata = self.metadata_store.get(index)

            output.append({
                "score": round(score, 4),
                "text": metadata["text"],
                "metadata": metadata["metadata"],
                "created_at": metadata["created_at"],
                "id": metadata["id"],
            })

        return output