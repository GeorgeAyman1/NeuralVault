import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes.memory import router as memory_router
from api.routes.health import router as health_router
from core.utils.config import get_settings
from core.utils.logging import get_logger
from core.utils.metrics import get_metrics

logger  = get_logger("neuralvault.api")
metrics = get_metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "NeuralVault starting — model=%s llm_available=%s",
        settings.llm_model, settings.llm_available,
    )
    yield
    logger.info("NeuralVault shutting down")


app = FastAPI(
    title="NeuralVault",
    version="1.0.0",
    description="Semantic long-term memory engine for LLM systems.",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(memory_router)


@app.middleware("http")
async def timing_and_metrics(request: Request, call_next):
    """Time every request, record metrics, and add an X-Process-Time header."""
    start = time.perf_counter()
    errored = False
    try:
        response = await call_next(request)
        errored = response.status_code >= 500
        return response
    except Exception:
        errored = True
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        op = f"{request.method} {request.url.path}"
        metrics.record(op, elapsed_ms, error=errored)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a structured 500 instead of leaking a stack trace to clients."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.get("/", tags=["Root"])
def root():
    settings = get_settings()
    return {
        "project": "NeuralVault",
        "version": "1.0.0",
        "status": "running",
        "llm_available": settings.llm_available,
        "docs": "/docs",
        "ui": "/ui",
    }


# Serve the demo UI at /ui (mounted last so it never shadows API routes).
_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
