from fastapi import APIRouter, HTTPException

from api.schemas.memory import (
    MemoryCreateRequest,
    MemorySearchRequest,
    MemoryBatchCreateRequest,
    ConsolidationRequest,
    MergeRequest,
    PruneRequest,
    SummarizeRequest,
    BuildIndexRequest,
)
from core.memory.service import MemoryService


router = APIRouter(prefix="/memory", tags=["Memory"])

# Single shared instance — not recreated per request
memory_service = MemoryService()


@router.post("/add")
def add_memory(request: MemoryCreateRequest):
    return memory_service.add_memory(text=request.text, metadata=request.metadata)


@router.post("/add-batch")
def add_memories(request: MemoryBatchCreateRequest):
    return memory_service.add_memories(texts=request.texts, metadata_list=request.metadata_list)


@router.post("/search")
def search_memory(request: MemorySearchRequest):
    return memory_service.search_memory(query=request.query, top_k=request.top_k)


@router.delete("/{index}")
def delete_memory(index: int):
    try:
        return memory_service.delete_memory(index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/build-index")
def build_index(request: BuildIndexRequest | None = None):
    n_clusters = request.n_clusters if request else None
    return memory_service.build_index(n_clusters=n_clusters)


@router.get("/count")
def count_memories():
    return {"count": memory_service.count()}


@router.post("/save")
def save_memory():
    memory_service.save()
    return {"status": "saved", "count": memory_service.count()}


@router.post("/load")
def load_memory():
    try:
        memory_service.load()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "loaded", "count": memory_service.count()}


@router.post("/compact")
def compact():
    return memory_service.compact()


# ------------------------------------------------------------------ #
# Sprint 1: Memory intelligence                                        #
# ------------------------------------------------------------------ #

@router.post("/consolidation/candidates")
def find_consolidation_candidates(request: ConsolidationRequest):
    return memory_service.find_consolidation_candidates(
        similarity_threshold=request.similarity_threshold,
    )


@router.post("/consolidation/merge")
def merge_memories(request: MergeRequest):
    try:
        return memory_service.merge_memories(request.index_a, request.index_b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/prune")
def prune_memories(request: PruneRequest):
    return memory_service.prune_memories(threshold=request.threshold)


@router.post("/summarize")
def summarize_memories(request: SummarizeRequest):
    try:
        return memory_service.summarize_memories(indices=request.indices)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
