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
    assert client.get("/static/styles.css").headers["cache-control"] == "no-cache"   # sin CSS viejo en caché


def test_health(client) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok" and data["index"]["archetypes"] == 8
    assert data["llm"]["chain"][0] == {"provider": "gemini", "model": "gemini-test", "available": True, "exhausted": False}
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


def test_towns(client) -> None:
    data = client.get("/towns").json()
    assert [t["town"] for t in data["towns"]] == ["Arqueo", "Frio", "Grande", "Playa"] and data["n_max"] == 10000
    first = data["towns"][0]
    assert first["n_reviews"] == 100 and first["archetypes"] == ["Arqueo-0", "Arqueo-1"]
    assert first["lat"] is None and first["lon"] is None      # el índice de prueba no está en la tabla de coordenadas


def test_recommend_keeps_contract_and_adds_towns(client) -> None:
    data = client.post("/recommend", json={"query": "ruinas tranquilas", "lambda": 0.0}).json()
    previous = {"query", "hyde", "anti_hyde", "fallback", "fallback_reason", "degraded", "llm_provider", "llm_model",
                "llm_errors", "llm_attempts", "cached", "lambda", "top", "bottom", "timings_ms"}
    assert previous <= set(data) and set(data) - previous == {"towns"}
    towns = {t["town"]: t for t in data["towns"]}
    assert len(towns) == 4
    for kind in ("top", "bottom"):
        for r in data[kind]:
            t = towns[r["town"]]
            assert (t["group"], t["rank"]) == (kind, r["rank"])
            assert t["sim_hyde"] == pytest.approx(r["sim_hyde"]) and t["score"] == pytest.approx(r["score"])
    assert all(t["group"] is None for name, t in towns.items()
               if name not in {r["town"] for r in data["top"] + data["bottom"]})


def test_recommend_stream(client) -> None:
    import json

    resp = client.post("/recommend/stream", json={"query": "ruinas tranquilas", "lambda": 0.0})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert [e["event"] for e in events] == ["expansion", "result"]
    assert events[0]["hyde"] and events[0]["anti_hyde"] and events[0]["llm_provider"] == "gemini"
    result = events[1]["data"]
    assert result["top"][0]["town"] == "Arqueo" and result["hyde"] == events[0]["hyde"] and len(result["towns"]) == 4


def test_recommend_stream_errors(client) -> None:
    from restmex.llm import LLMError

    assert client.post("/recommend/stream", json={"query": "a"}).status_code == 422
    assert client.post("/recommend/stream", json={"query": "playa"}).status_code == 422   # una sola palabra

    def boom(texts):
        raise LLMError("ollama apagado")

    app_main.app.state.recommender.encoder.encode_documents = boom
    resp = client.post("/recommend/stream", json={"query": "ruinas tranquilas"})
    last = [line for line in resp.text.splitlines() if line.strip()][-1]
    assert resp.status_code == 200 and '"event": "error"' in last and "Ollama" in last
    assert client.post("/recommend", json={"query": "ruinas tranquilas"}).status_code == 503

    app_main.app.state.recommender = None
    app_main.app.state.index_error = "sin índice"
    assert client.post("/recommend/stream", json={"query": "playa y fiesta"}).status_code == 503
    assert client.get("/towns").status_code == 503
