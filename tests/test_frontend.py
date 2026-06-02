"""
Tests for the frontend-supporting backend: GET /memory/list and the /ui mount.

The list_memories unit tests run offline against an isolated store. The
integration test boots the real app via TestClient (encoder loads once).
"""
import pytest
from fastapi.testclient import TestClient

from core.memory.service import MemoryService
from core.storage.vecdb_store import VecDBStore
from core.storage.metadata_store import MetadataStore
from core.indexing.retrieval import SemanticRetriever


@pytest.fixture
def service(tmp_path):
    svc = MemoryService(auto_load=False)
    # Redirect BOTH stores to tmp so nothing leaks into the real data dir.
    svc.vector_store = VecDBStore(
        db_path=str(tmp_path / "vecs.npy"),
        index_path=str(tmp_path / "idx"),
    )
    svc.metadata_store = MetadataStore(path=str(tmp_path / "meta.json"))
    svc.retriever = SemanticRetriever(
        encoder=svc.encoder,
        vector_store=svc.vector_store,
        metadata_store=svc.metadata_store,
    )
    return svc


# --------------------------------------------------------------------------- #
# list_memories                                                               #
# --------------------------------------------------------------------------- #

def test_list_memories_returns_items_with_fields(service):
    service.add_memory("first memory", {"importance": 0.5})
    service.add_memory("second memory")

    out = service.list_memories()
    assert out["total"] == 2
    assert len(out["items"]) == 2
    item = out["items"][0]
    for field in ("index", "id", "text", "metadata", "created_at"):
        assert field in item
    assert item["text"] == "first memory"
    assert item["metadata"]["importance"] == 0.5


def test_list_memories_excludes_deleted(service):
    service.add_memory("keep")
    service.add_memory("remove")
    service.delete_memory(1)

    out = service.list_memories()
    assert out["total"] == 1
    assert out["items"][0]["text"] == "keep"
    assert out["items"][0]["index"] == 0


def test_list_memories_pagination(service):
    for i in range(10):
        service.add_memory(f"memory {i}")

    page = service.list_memories(offset=3, limit=4)
    assert page["total"] == 10
    assert page["offset"] == 3 and page["limit"] == 4
    assert len(page["items"]) == 4
    assert page["items"][0]["text"] == "memory 3"


# --------------------------------------------------------------------------- #
# App integration: /ui mount + /memory/list route                            #
# --------------------------------------------------------------------------- #

def test_ui_is_served():
    from main import app
    with TestClient(app) as client:
        r = client.get("/ui/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "NeuralVault" in r.text
        # Root advertises the UI
        assert client.get("/").json()["ui"] == "/ui"


def test_list_route_returns_structure():
    from main import app
    with TestClient(app) as client:
        r = client.get("/memory/list?limit=5")
        assert r.status_code == 200
        body = r.json()
        for field in ("total", "offset", "limit", "items"):
            assert field in body
        assert isinstance(body["items"], list)
