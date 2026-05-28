from pydantic import BaseModel, Field
from typing import Any


class MemoryCreateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)