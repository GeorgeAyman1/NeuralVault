from pydantic import BaseModel, Field
from typing import Any


class MemoryCreateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)

class MemoryBatchCreateRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    metadata_list: list[dict[str, Any]] | None = None

class ConsolidationRequest(BaseModel):
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    use_hybrid: bool = False
    use_ivf: bool = False