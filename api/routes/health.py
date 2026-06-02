from fastapi import APIRouter

from core.utils.config import get_settings
from core.utils.metrics import get_metrics

router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    """Liveness probe — the process is up."""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness():
    """
    Readiness probe — reports whether optional dependencies are configured.
    Always 200 (the service runs without an LLM key); the body flags what's available.
    """
    settings = get_settings()
    return {
        "status": "ready",
        "llm_available": settings.llm_available,
        "model": settings.llm_model,
    }


@router.get("/metrics")
def metrics():
    """In-process request/latency/error metrics."""
    return get_metrics().snapshot()
