from core.embeddings.encoder import TextEncoder
from core.storage.vecdb_store import VecDBStore
from core.storage.metadata_store import MetadataStore
from core.indexing.retrieval import SemanticRetriever
from core.utils.logging import get_logger
from core.memory.consolidation import MemoryConsolidator
from core.memory.pruning import MemoryPruner
from core.memory.summarization import MemorySummarizer
from core.ingestion.notebook_loader import NotebookLoader
from core.ingestion.pdf_loader import PDFLoader
from core.ingestion.docx_loader import DOCXLoader
from core.ingestion.directory_loader import DirectoryLoader
from core.llm.context_builder import ContextBuilder
from core.llm.conversation import ConversationMemory
from core.llm.llm_client import LLMClient


class MemoryService:
    """
    High-level service for adding, searching, and managing semantic memories.

    Storage backend is VecDBStore, which uses brute-force cosine similarity
    for small collections and IVF-backed VecDB after build_index() is called.
    """

    def __init__(self, auto_load: bool = True, llm_client: LLMClient | None = None):
        self.logger          = get_logger(__name__)
        self.encoder         = TextEncoder()
        self.vector_store    = VecDBStore()
        self.metadata_store  = MetadataStore()
        self.consolidator         = MemoryConsolidator()
        self.pruner               = MemoryPruner()
        self.summarizer           = MemorySummarizer()
        self.notebook_loader      = NotebookLoader()
        self.pdf_loader           = PDFLoader()
        self.docx_loader          = DOCXLoader()
        self.directory_loader     = DirectoryLoader()
        self.context_builder      = ContextBuilder()
        self.llm                  = llm_client or LLMClient()
        self.conversation         = ConversationMemory()

        if auto_load:
            self.load()

        self.retriever = SemanticRetriever(
            encoder=self.encoder,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
        )

        self.logger.info("MemoryService ready — %d memories loaded", self.vector_store.count())

    # ------------------------------------------------------------------ #
    # Core CRUD                                                            #
    # ------------------------------------------------------------------ #

    def add_memory(self, text: str, metadata: dict | None = None) -> dict:
        vector         = self.encoder.encode(text)
        metadata_index = self.metadata_store.add(text=text, metadata=metadata)
        vector_index   = self.vector_store.add(vector)
        self.save()
        self.logger.info("Stored memory idx=%d", metadata_index)
        return {"status": "stored", "memory_index": metadata_index, "vector_index": vector_index}

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

    def delete_memory(self, index: int) -> dict:
        if index < 0 or index >= len(self.metadata_store.records):
            raise ValueError(f"Index {index} out of range.")
        if self.metadata_store.records[index].get("deleted", False):
            raise ValueError(f"Memory at index {index} is already deleted.")

        self.metadata_store.delete(index)
        self.vector_store.delete(index)
        self.save()
        self.logger.info("Soft-deleted memory idx=%d", index)
        return {"status": "deleted", "index": index}

    def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
        self.logger.info("Search: query='%s' top_k=%d", query, top_k)
        return self.retriever.search(query=query, top_k=top_k)

    def count(self) -> int:
        return self.metadata_store.count()

    # ------------------------------------------------------------------ #
    # Index                                                                #
    # ------------------------------------------------------------------ #

    def build_index(self, n_clusters: int | None = None) -> dict:
        n = self.vector_store.count()
        self.logger.info("Building IVF index for %d memories", n)
        self.vector_store.build_index(n_clusters=n_clusters)
        status = "built" if self.vector_store._index_valid else "skipped (too few memories)"
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
        # Sync deleted indices from metadata (source of truth)
        self.vector_store.sync_deleted(self.metadata_store.records)

        if self.vector_store.count() != self.metadata_store.count():
            raise ValueError(
                f"Store mismatch — vectors: {self.vector_store.count()}, "
                f"metadata: {self.metadata_store.count()}"
            )

    # ------------------------------------------------------------------ #
    # Sprint 1: Memory intelligence                                        #
    # ------------------------------------------------------------------ #

    def find_consolidation_candidates(self, similarity_threshold: float = 0.85) -> list[dict]:
        self.logger.info("Finding consolidation candidates (threshold=%.2f)", similarity_threshold)
        consolidator = MemoryConsolidator(similarity_threshold=similarity_threshold)
        return consolidator.find_candidates(
            vectors=self.vector_store.vectors,
            records=self.metadata_store.records,
        )

    def merge_memories(self, index_a: int, index_b: int) -> dict:
        """
        Merge two memories into one re-encoded combined memory, delete both originals.
        """
        records = self.metadata_store.records
        if index_a == index_b:
            raise ValueError("Cannot merge a memory with itself.")
        for idx in (index_a, index_b):
            if idx < 0 or idx >= len(records):
                raise ValueError(f"Index {idx} out of range.")
            if records[idx].get("deleted", False):
                raise ValueError(f"Memory at index {idx} is already deleted.")

        record_a = records[index_a]
        record_b = records[index_b]

        merged_text = f"{record_a['text']}\n{record_b['text']}"
        merged_meta = {
            **record_a.get("metadata", {}),
            "merged_from": [record_a["id"], record_b["id"]],
        }

        result = self.add_memory(merged_text, metadata=merged_meta)

        # Delete originals (add_memory may have grown the records list, so indices stable)
        self.metadata_store.delete(index_a)
        self.vector_store.delete(index_a)
        self.metadata_store.delete(index_b)
        self.vector_store.delete(index_b)

        self.save()
        self.logger.info("Merged memories %d + %d → new idx %d", index_a, index_b,
                         result["memory_index"])
        return {**result, "merged_from": [index_a, index_b]}

    def prune_memories(self, threshold: float = 0.1) -> dict:
        """
        Soft-delete memories whose keep-score is below threshold.
        """
        pruner    = MemoryPruner(threshold=threshold)
        prunable  = pruner.find_prunable(self.metadata_store.records)

        for idx in prunable:
            self.metadata_store.delete(idx)
            self.vector_store.delete(idx)

        self.save()
        self.logger.info("Pruned %d memories (threshold=%.2f)", len(prunable), threshold)
        return {"status": "pruned", "pruned_count": len(prunable), "pruned_indices": prunable}

    def summarize_memories(self, indices: list[int]) -> dict:
        """
        Create a summary memory from the given indices, then delete the originals.
        Sprint 1: extractive. Sprint 3: LLM-generated.
        """
        if len(indices) < 2:
            raise ValueError("Need at least 2 memories to summarize.")

        records = self.metadata_store.records
        vstore  = self.vector_store

        for idx in indices:
            if idx < 0 or idx >= len(records):
                raise ValueError(f"Index {idx} out of range.")
            if records[idx].get("deleted", False):
                raise ValueError(f"Memory at index {idx} is already deleted.")

        source_records = [records[i] for i in indices]
        source_vectors = [vstore._matrix[i] for i in indices]

        summary_text = self.summarizer.summarize(source_records, source_vectors)
        source_ids   = [r["id"] for r in source_records]

        result = self.add_memory(summary_text, metadata={"summarized_from": source_ids})

        for idx in indices:
            self.metadata_store.delete(idx)
            self.vector_store.delete(idx)

        self.save()
        self.logger.info("Summarized %d memories → new idx %d", len(indices),
                         result["memory_index"])
        return {**result, "summarized_count": len(indices), "summarized_from": source_ids}

    def compact(self) -> dict:
        """
        Permanently remove all soft-deleted entries from both stores.
        Rebuilds contiguous indices. Invalidates any existing IVF index.
        """
        before = len(self.metadata_store.records)
        kept   = self.metadata_store.compact()
        self.vector_store.compact(kept)
        self.save()
        after  = self.metadata_store.count()
        self.logger.info("Compacted: %d → %d memories", before, after)
        return {"status": "compacted", "before": before, "after": after, "removed": before - after}

    # ------------------------------------------------------------------ #
    # Sprint 3: LLM integration                                            #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        query: str,
        top_k: int = 5,
        use_conversation: bool = True,
    ) -> dict:
        """
        Answer a query grounded in retrieved memories.

        Retrieves the top_k most relevant memories, builds a cached-prefix
        system prompt from them, and asks the LLM. When use_conversation is
        True, the exchange is appended to the running ConversationMemory so
        follow-up questions retain context.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        memories      = self.search_memory(query, top_k=top_k)
        system_blocks = self.context_builder.build_system_blocks(memories)

        conversation = self.conversation if use_conversation else ConversationMemory()
        conversation.add_user(query)

        result = self.llm.complete(
            system_blocks=system_blocks,
            messages=conversation.get_messages(),
        )
        answer = result["text"]
        conversation.add_assistant(answer)

        self.logger.info("Chat: query='%s' memories=%d", query, len(memories))
        return {
            "query":        query,
            "answer":       answer,
            "memories_used": [
                {"index": i, "text": m["text"], "score": m.get("score")}
                for i, m in enumerate(memories)
            ],
            "usage":        result.get("usage", {}),
        }

    def reset_conversation(self) -> dict:
        self.conversation.clear()
        return {"status": "reset"}

    # ------------------------------------------------------------------ #
    # Ingestion                                                            #
    # ------------------------------------------------------------------ #

    def ingest_notebook(self, path: str) -> dict:
        chunks = self.notebook_loader.load(path)
        self.logger.info("Ingesting notebook %s (%d chunks)", path, len(chunks))
        return self._ingest_chunks(chunks, path)

    def ingest_pdf(self, path: str) -> dict:
        chunks = self.pdf_loader.load(path)
        self.logger.info("Ingesting PDF %s (%d chunks)", path, len(chunks))
        return self._ingest_chunks(chunks, path)

    def ingest_docx(self, path: str) -> dict:
        chunks = self.docx_loader.load(path)
        self.logger.info("Ingesting DOCX %s (%d chunks)", path, len(chunks))
        return self._ingest_chunks(chunks, path)

    def ingest_directory(self, path: str, extensions: list[str] | None = None) -> dict:
        loader = DirectoryLoader(
            extensions=set(extensions) if extensions else None
        )
        chunks = loader.load(path)
        self.logger.info("Ingesting directory %s (%d chunks)", path, len(chunks))
        return self._ingest_chunks(chunks, path)

    def _ingest_chunks(self, chunks: list[dict], source: str) -> dict:
        if not chunks:
            return {"status": "empty", "source": source, "count": 0}
        return {
            **self.add_memories(
                texts=[c["text"] for c in chunks],
                metadata_list=[c["metadata"] for c in chunks],
            ),
            "source": source,
        }
