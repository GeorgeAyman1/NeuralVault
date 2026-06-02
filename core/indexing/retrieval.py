from core.embeddings.encoder import TextEncoder
from core.storage.vecdb_store import VecDBStore
from core.storage.metadata_store import MetadataStore
from core.memory.ranking import MemoryRanker
from core.indexing.hybrid_retrieval import HybridRetrieval


class SemanticRetriever:
    """
    Retrieves memories using VecDBStore (brute-force or IVF, automatic).

    Post-processes results through MemoryRanker (similarity + recency +
    importance + frequency) and optionally through HybridRetrieval.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        vector_store: VecDBStore,
        metadata_store: MetadataStore,
        use_hybrid: bool = False,
    ):
        self.encoder        = encoder
        self.vector_store   = vector_store
        self.metadata_store = metadata_store
        self.use_hybrid     = use_hybrid
        self.ranker         = MemoryRanker()
        self.hybrid         = HybridRetrieval()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_vector = self.encoder.encode(query)
        raw_results  = self.vector_store.search(query_vector, top_k=top_k)

        output = []
        for index, score in raw_results:
            self.metadata_store.increment_retrieval_count(index)
            metadata = self.metadata_store.get(index)
            output.append({
                "score":      round(score, 4),
                "text":       metadata["text"],
                "metadata":   metadata["metadata"],
                "created_at": metadata["created_at"],
                "id":         metadata["id"],
            })

        ranked = self.ranker.rank(output)

        if self.use_hybrid:
            return self.hybrid.rerank(query=query, semantic_results=ranked)

        return ranked
