"""API web con el recomendador simulado (sin Ollama ni Gemini)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from restmex.app import main as app_main
from restmex.llm import LLMProvider
from restmex.recommender import Recommender

from conftest import FakeEncoder, FakeOllama, good_json


@pytest.fixture
def client(index, cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("RESTMEX_ARTIFACTS", str(tmp_path))      # sin índice en disco
    with TestClient(app_main.app) as c:
        fake = FakeOllama([good_json()])
        cfg.ranking.top_k = cfg.ranking.bottom_k = 2          # el índice de prueba tiene 4 pueblos
        app_main.app.state.recommender = Recommender(
            cfg, index, encoder=FakeEncoder(), client=fake,
            providers=[LLMProvider("gemini", "gemini-test", fake)],
        )
        yield c


def test_home_and_static(client) -> None:
    assert "Pueblos Mágicos" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_health(client) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok" and data["index"]["archetypes"] == 8
    assert data["llm"]["chain"][0] == {"provider": "gemini", "model": "gemini-test", "available": True}
    assert len(data["examples"]) == 4


def test_recommend(client) -> None:
    data = client.post("/recommend", json={"query": "ruinas tranquilas", "lambda": 0.0}).json()
    assert data["llm_provider"] == "gemini" and data["fallback"] is False
    assert data["top"][0]["town"] == "Arqueo"            # el HyDE simulado habla de ruinas
    assert data["bottom"] and data["hyde"] and data["anti_hyde"]
    assert not {t["town"] for t in data["top"]} & {b["town"] for b in data["bottom"]}
    assert data["timings_ms"]["busqueda"] < 50


def test_recommend_validation_and_missing_index(client) -> None:
    assert client.post("/recommend", json={"query": "a"}).status_code == 422
    app_main.app.state.recommender = None
    app_main.app.state.index_error = "sin índice"
    resp = client.post("/recommend", json={"query": "playa y fiesta"})
    assert resp.status_code == 503 and "sin índice" in resp.json()["detail"]
    assert client.get("/health").json()["status"] == "degraded"
