from pydantic import BaseModel, Field
from typing import Any


class MemoryCreateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class MemoryBatchCreateRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    metadata_list: list[dict[str, Any]] | None = None


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    use_hybrid: bool = False


class ConsolidationRequest(BaseModel):
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class MergeRequest(BaseModel):
    index_a: int = Field(..., ge=0)
    index_b: int = Field(..., ge=0)


class PruneRequest(BaseModel):
    threshold: float = Field(default=0.1, ge=0.0, le=1.0)


class SummarizeRequest(BaseModel):
    indices: list[int] = Field(..., min_length=2)


class BuildIndexRequest(BaseModel):
    n_clusters: int | None = Field(default=None, ge=16, le=65536)


class IngestPathRequest(BaseModel):
    path: str = Field(..., min_length=1)


class IngestDirectoryRequest(BaseModel):
    path: str = Field(..., min_length=1)
    extensions: list[str] | None = None
