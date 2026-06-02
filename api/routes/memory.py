from fastapi import APIRouter, HTTPException

from api.schemas.memory import (
    MemoryCreateRequest,
    MemorySearchRequest,
    MemoryBatchCreateRequest,
    ConsolidationRequest,
    BuildIndexRequest,
)
from core.memory.service import MemoryService


router = APIRouter(prefix="/memory", tags=["Memory"])

# Single shared instance — not recreated per request
memory_service = MemoryService()


@router.post("/add")
def add_memory(request: MemoryCreateRequest):
    return memory_service.add_memory(
        text=request.text,
        metadata=request.metadata,
    )


@router.post("/add-batch")
def add_memories(request: MemoryBatchCreateRequest):
    return memory_service.add_memories(
        texts=request.texts,
        metadata_list=request.metadata_list,
    )


@router.post("/search")
def search_memory(request: MemorySearchRequest):
    return memory_service.search_memory(
        query=request.query,
        top_k=request.top_k,
    )


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


@router.post("/consolidation/candidates")
def find_consolidation_candidates(request: ConsolidationRequest):
    return memory_service.find_consolidation_candidates(
        similarity_threshold=request.similarity_threshold,
    )
