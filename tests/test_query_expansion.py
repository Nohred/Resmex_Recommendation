"""HyDE / Anti-HyDE con Ollama simulado: validación, reintento y fallback."""
from __future__ import annotations

import pytest

from restmex.config import LLMConfig
from restmex.ollama_client import OllamaError
from restmex.query_expansion import antonym_anti_hyde, expand_query, parse_expansion
from restmex.recommender import Recommender

from conftest import GOOD_ANTI, GOOD_HYDE, FakeEncoder, FakeOllama, good_json

QUERY = "Quiero un lugar tranquilo con ruinas"


def test_happy_path() -> None:
    client = FakeOllama([good_json()])
    out = expand_query(QUERY, client, LLMConfig())
    assert (out.hyde, out.anti_hyde) == (GOOD_HYDE, GOOD_ANTI)
    assert not out.fallback and out.attempts == 1
    assert client.calls[0]["json_format"] is True
    assert QUERY in client.calls[0]["prompt"]


def test_retries_once_after_malformed_output() -> None:
    client = FakeOllama(["esto no es json", good_json()])
    out = expand_query(QUERY, client, LLMConfig(retries=1))
    assert not out.fallback and out.attempts == 2


def test_fallback_after_two_bad_outputs() -> None:
    client = FakeOllama(['{"hyde": "corto"}', "{"])
    out = expand_query(QUERY, client, LLMConfig(retries=1))
    assert out.fallback and out.attempts == 2
    assert out.hyde == QUERY
    assert "ruidoso" in out.anti_hyde            # opuesto de "tranquilo"
    assert "Salida mal formada" in out.fallback_reason
    assert len(client.calls) == 2


def test_fallback_when_ollama_is_down() -> None:
    out = expand_query(QUERY, FakeOllama([OllamaError("timeout"), OllamaError("timeout")]), LLMConfig())
    assert out.fallback and "Ollama no respondió" in out.fallback_reason


@pytest.mark.parametrize("raw", [
    "[]", '{"hyde": 3, "anti_hyde": "x"}', '{"hyde": "a b c", "anti_hyde": "d e f"}',
    '{"hyde": "%s", "anti_hyde": "%s"}' % (GOOD_HYDE, GOOD_HYDE),
])
def test_parse_rejects_bad_output(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_expansion(raw)


def test_antonym_template() -> None:
    text = antonym_anti_hyde("Busco fiesta y playa")
    assert "sin bares" in text and "sierra" in text
    assert "sin identidad" in antonym_anti_hyde("algo distinto")


def test_recommender_keeps_working_in_fallback(index, cfg) -> None:
    rec = Recommender(cfg, index, encoder=FakeEncoder(), client=FakeOllama([]))
    out = rec.recommend("Quiero ruinas, muchas ruinas", lam=0.0)
    assert out.expansion.fallback
    assert out.top[0].town == "Arqueo"
    assert len(out.top) + len(out.bottom) == 4
    data = out.to_dict()
    assert data["fallback"] is True and set(data["timings_ms"]) == {"llm", "embedding", "busqueda", "total"}
