from core.embeddings.encoder import TextEncoder
from core.storage.vector_store import VectorStore
from core.storage.metadata_store import MetadataStore
from core.indexing.ivf_index import IVFIndex
from core.memory.ranking import MemoryRanker


class SemanticRetriever:
    """
    Handles semantic retrieval using either brute-force or IVF search.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        use_ivf: bool = False,
        n_clusters: int = 8,
        n_probe: int = 2,
    ):
        self.encoder = encoder
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.use_ivf = use_ivf
        self.ranker = MemoryRanker()

        self.ivf_index = IVFIndex(
            n_clusters=n_clusters,
            n_probe=n_probe,
        )

        self.ivf_ready = False

    def build_ivf(self) -> None:
        if self.vector_store.count() == 0:
            self.ivf_ready = False
            return

        self.ivf_index.build(self.vector_store.vectors)
        self.ivf_ready = True

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_vector = self.encoder.encode(query)

        if self.use_ivf:
            if not self.ivf_ready:
                self.build_ivf()

            results = self.ivf_index.search(
                query_vector=query_vector,
                top_k=top_k,
            )
        else:
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
            )

        output = []

        for index, score in results:
            self.metadata_store.increment_retrieval_count(index)
            metadata = self.metadata_store.get(index)
            output.append({
                "score": round(score, 4),
                "text": metadata["text"],
                "metadata": metadata["metadata"],
                "created_at": metadata["created_at"],
                "id": metadata["id"],
            })

        return self.ranker.rank(output)