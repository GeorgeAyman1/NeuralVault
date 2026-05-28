from fastapi import FastAPI

from api.routes.memory import router as memory_router


app = FastAPI(
    title="NeuralVault",
    version="0.1.0",
)


app.include_router(memory_router)


@app.get("/")
def root():
    return {
        "project": "NeuralVault",
        "status": "running",
    }