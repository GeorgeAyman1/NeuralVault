from core.embeddings.encoder import TextEncoder
from core.storage.vecdb_store import VecDBStore
from core.storage.metadata_store import MetadataStore
from core.indexing.retrieval import SemanticRetriever
from core.utils.logging import get_logger
from core.memory.consolidation import MemoryConsolidator
from core.ingestion.notebook_loader import NotebookLoader


class MemoryService:
    """
    High-level service for adding, searching, and managing semantic memories.

    Storage backend is VecDBStore, which uses brute-force cosine similarity
    for small collections and IVF-backed VecDB after build_index() is called.
    """

    def __init__(self, auto_load: bool = True):
        self.logger         = get_logger(__name__)
        self.encoder        = TextEncoder()
        self.vector_store   = VecDBStore()
        self.metadata_store = MetadataStore()
        self.consolidator   = MemoryConsolidator()
        self.notebook_loader = NotebookLoader()

        if auto_load:
            self.load()

        self.retriever = SemanticRetriever(
            encoder=self.encoder,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
        )

        self.logger.info("MemoryService ready — %d memories loaded", self.vector_store.count())

    # ------------------------------------------------------------------ #
    # Core memory operations                                               #
    # ------------------------------------------------------------------ #

    def add_memory(self, text: str, metadata: dict | None = None) -> dict:
        vector = self.encoder.encode(text)

        metadata_index = self.metadata_store.add(text=text, metadata=metadata)
        vector_index   = self.vector_store.add(vector)

        self.save()
        self.logger.info("Stored memory idx=%d", metadata_index)

        return {
            "status":        "stored",
            "memory_index":  metadata_index,
            "vector_index":  vector_index,
        }

    def add_memories(self, texts: list[str], metadata_list: list[dict] | None = None) -> dict:
        if not texts:
            raise ValueError("Texts list cannot be empty.")
        if metadata_list is not None and len(metadata_list) != len(texts):
            raise ValueError("metadata_list must match texts length.")

        vectors = self.encoder.encode_batch(texts)
        stored  = []

        for i, text in enumerate(texts):
            meta           = metadata_list[i] if metadata_list else None
            metadata_index = self.metadata_store.add(text=text, metadata=meta)
            vector_index   = self.vector_store.add(vectors[i])
            stored.append({"memory_index": metadata_index, "vector_index": vector_index})

        self.save()
        self.logger.info("Stored batch of %d memories", len(texts))

        return {"status": "stored", "count": len(texts), "items": stored}

    def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
        self.logger.info("Search: query='%s' top_k=%d", query, top_k)
        return self.retriever.search(query=query, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Index                                                                #
    # ------------------------------------------------------------------ #

    def build_index(self, n_clusters: int | None = None) -> dict:
        n = self.vector_store.count()
        self.logger.info("Building IVF index for %d memories", n)
        self.vector_store.build_index(n_clusters=n_clusters)
        status = "built" if self.vector_store._index_valid else "skipped (too few memories)"
        self.logger.info("Index status: %s", status)
        return {"status": status, "memory_count": n}

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        self.vector_store.save()
        self.metadata_store.save()

    def load(self) -> None:
        self.vector_store.load()
        self.metadata_store.load()

        if self.vector_store.count() != self.metadata_store.count():
            raise ValueError(
                f"Store mismatch — vectors: {self.vector_store.count()}, "
                f"metadata: {self.metadata_store.count()}"
            )

    def count(self) -> int:
        return self.metadata_store.count()

    # ------------------------------------------------------------------ #
    # Memory intelligence                                                  #
    # ------------------------------------------------------------------ #

    def find_consolidation_candidates(self, similarity_threshold: float = 0.85) -> list[dict]:
        self.logger.info("Finding consolidation candidates (threshold=%.2f)", similarity_threshold)
        consolidator = MemoryConsolidator(similarity_threshold=similarity_threshold)
        return consolidator.find_candidates(
            vectors=self.vector_store.vectors,
            records=self.metadata_store.records,
        )

    def ingest_notebook(self, path: str) -> dict:
        chunks = self.notebook_loader.load(path)
        self.logger.info("Ingesting notebook %s (%d chunks)", path, len(chunks))
        return self.add_memories(
            texts=[c["text"] for c in chunks],
            metadata_list=[c["metadata"] for c in chunks],
        )
