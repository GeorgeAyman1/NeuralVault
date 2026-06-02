"""
Sprint 4 tests: config, metrics, health/metrics endpoints, graceful LLM-unavailable.

Health-endpoint tests mount only the health router on a fresh app, so they
don't construct the heavy MemoryService (encoder + data load).
"""
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.utils.config import Settings
from core.utils.metrics import Metrics
from core.llm.llm_client import LLMClient, LLMUnavailableError
from api.routes.health import router as health_router


# --------------------------------------------------------------------------- #
# Settings                                                                    #
# --------------------------------------------------------------------------- #

def test_settings_defaults(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "NEURALVAULT_LLM_MODEL", "NEURALVAULT_DEFAULT_TOP_K"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.llm_model == "claude-opus-4-8"
    assert s.default_top_k == 5
    assert s.llm_available is False


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("NEURALVAULT_LLM_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("NEURALVAULT_DEFAULT_TOP_K", "12")
    s = Settings.from_env()
    assert s.anthropic_api_key == "sk-ant-test"
    assert s.llm_model == "claude-haiku-4-5"
    assert s.default_top_k == 12
    assert s.llm_available is True


def test_settings_bad_int_falls_back(monkeypatch):
    monkeypatch.setenv("NEURALVAULT_DEFAULT_TOP_K", "not-a-number")
    s = Settings.from_env()
    assert s.default_top_k == 5  # falls back to default


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def test_metrics_records_count_and_latency():
    m = Metrics()
    m.record("op_a", 10.0)
    m.record("op_a", 20.0)
    snap = m.snapshot()
    assert snap["operations"]["op_a"]["count"] == 2
    assert snap["operations"]["op_a"]["avg_latency_ms"] == 15.0
    assert snap["total_requests"] == 2


def test_metrics_records_errors():
    m = Metrics()
    m.record("op_b", 5.0, error=True)
    m.record("op_b", 5.0, error=False)
    snap = m.snapshot()
    assert snap["operations"]["op_b"]["errors"] == 1
    assert snap["total_errors"] == 1


def test_metrics_timer_records_and_propagates_error():
    m = Metrics()
    with m.timer("timed_ok"):
        time.sleep(0.001)
    with pytest.raises(ValueError):
        with m.timer("timed_err"):
            raise ValueError("boom")
    snap = m.snapshot()
    assert snap["operations"]["timed_ok"]["count"] == 1
    assert snap["operations"]["timed_ok"]["errors"] == 0
    assert snap["operations"]["timed_err"]["errors"] == 1


def test_metrics_reset():
    m = Metrics()
    m.record("op", 1.0)
    m.reset()
    assert m.snapshot()["total_requests"] == 0


# --------------------------------------------------------------------------- #
# Health / metrics endpoints                                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(health_router)
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readiness_endpoint(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "llm_available" in body
    assert "model" in body


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "uptime_seconds" in body
    assert "total_requests" in body
    assert "operations" in body


# --------------------------------------------------------------------------- #
# Graceful LLM-unavailable                                                    #
# --------------------------------------------------------------------------- #

def test_llm_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # No injected client, no key → clear error rather than opaque SDK crash
    client = LLMClient(api_key=None)
    client._api_key = None  # ensure settings default didn't pick one up
    with pytest.raises(LLMUnavailableError):
        client.complete(
            system_blocks=[{"type": "text", "text": "sys"}],
            messages=[{"role": "user", "content": "q"}],
        )


def test_llm_client_uses_injected_client_without_key():
    # An injected client bypasses the key requirement entirely
    from types import SimpleNamespace

    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                      cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0),
            )

    fake = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: _Stream()))
    result = LLMClient(client=fake).complete(
        system_blocks=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": "q"}],
    )
    assert result["text"] == "ok"
