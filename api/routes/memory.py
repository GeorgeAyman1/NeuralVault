from fastapi import APIRouter

from api.schemas.memory import (
    MemoryCreateRequest,
    MemorySearchRequest,
)

from core.memory.service import MemoryService


router = APIRouter(prefix="/memory", tags=["Memory"])

memory_service = MemoryService()


@router.post("/add")
def add_memory(request: MemoryCreateRequest):
    return memory_service.add_memory(
        text=request.text,
        metadata=request.metadata,
    )


@router.post("/search")
def search_memory(request: MemorySearchRequest):
    return memory_service.search_memory(
        query=request.query,
        top_k=request.top_k,
    )

@router.get("/count")
def count_memories():
    return {
        "count": memory_service.count()
    }


@router.post("/save")
def save_memory():
    memory_service.save()

    return {
        "status": "saved",
        "count": memory_service.count()
    }


@router.post("/load")
def load_memory():
    memory_service.load()

    return {
        "status": "loaded",
        "count": memory_service.count()
    }