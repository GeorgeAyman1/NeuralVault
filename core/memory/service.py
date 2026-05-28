from core.embeddings.encoder import TextEncoder
from core.storage.vector_store import VectorStore
from core.storage.metadata_store import MetadataStore
from core.indexing.retrieval import SemanticRetriever
from core.utils.logging import get_logger


class MemoryService:
    """
    High-level service for adding, saving, loading, and searching semantic memories.
    """

    def __init__(self, auto_load: bool = True, use_ivf: bool = False):
        self.encoder = TextEncoder()
        self.vector_store = VectorStore()
        self.metadata_store = MetadataStore()
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MemoryService")

        if auto_load:
            self.load()

        self.retriever = SemanticRetriever(
            encoder=self.encoder,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
            use_ivf=use_ivf,
        )

    def add_memory(self, text: str, metadata: dict | None = None) -> dict:
        vector = self.encoder.encode(text)

        metadata_index = self.metadata_store.add(
            text=text,
            metadata=metadata,
        )

        vector_index = self.vector_store.add(vector)

        self.save()
        self.logger.info("Stored memory: metadata_index=%s vector_index=%s", metadata_index, vector_index)

        return {
            "status": "stored",
            "memory_index": metadata_index,
            "vector_index": vector_index,
        }

    def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
        self.logger.info("Searching memory: query='%s' top_k=%s", query, top_k)
        return self.retriever.search(query=query, top_k=top_k)

    def save(self) -> None:
        self.vector_store.save()
        self.metadata_store.save()
        self.logger.info("Saving memory stores")

    def load(self) -> None:
        self.vector_store.load()
        self.metadata_store.load()
        self.logger.info("Loading memory stores")

        if self.vector_store.count() != self.metadata_store.count():
            raise ValueError(
                "Vector store and metadata store are out of sync. "
                f"Vectors: {self.vector_store.count()}, "
                f"Metadata: {self.metadata_store.count()}"
            )

    def count(self) -> int:
        return self.metadata_store.count()
    
    def add_memories(self, texts: list[str], metadata_list: list[dict] | None = None) -> dict:
        if not texts:
            raise ValueError("Texts list cannot be empty.")

        if metadata_list is not None and len(metadata_list) != len(texts):
            raise ValueError("metadata_list must match texts length.")

        vectors = self.encoder.encode_batch(texts)

        stored_indices = []

        for index, text in enumerate(texts):
            metadata = metadata_list[index] if metadata_list else None

            metadata_index = self.metadata_store.add(
                text=text,
                metadata=metadata,
            )

            vector_index = self.vector_store.add(vectors[index])

            stored_indices.append({
                "memory_index": metadata_index,
                "vector_index": vector_index,
            })

        self.save()
        self.logger.info("Stored batch memories: count=%s", len(texts))

        return {
            "status": "stored",
            "count": len(texts),
            "items": stored_indices,
        }

    def build_index(self) -> None:
        if self.retriever.use_ivf:
            self.logger.info("Building IVF index")
            self.retriever.build_ivf()