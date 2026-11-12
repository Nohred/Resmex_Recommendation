"""HyDE / Anti-HyDE con LLM simulados: validación, reintento y cadena de respaldo."""
from __future__ import annotations

import json

import pytest

from restmex.archetypes import label_archetypes
from restmex.config import Config, LLMConfig
from restmex.llm import GeminiClient, GeminiError, LLMProvider, build_llm_chain, parse_retry_after
from restmex.ollama_client import OllamaError
from restmex.query_expansion import antonym_anti_hyde, expand_query, parse_expansion
from restmex.recommender import Recommender

from conftest import GOOD_ANTI, GOOD_HYDE, FakeEncoder, FakeOllama, good_json

QUERY = "Quiero un lugar tranquilo con ruinas"


def chain(gemini: FakeOllama, ollama: FakeOllama) -> list[LLMProvider]:
    return [
        LLMProvider("gemini", "gemini-test", gemini),
        LLMProvider("ollama", "qwen-test", ollama),
    ]


def only(client: FakeOllama) -> list[LLMProvider]:
    return [LLMProvider("ollama", "qwen-test", client)]


def test_happy_path_uses_first_provider() -> None:
    gemini, ollama = FakeOllama([good_json()]), FakeOllama([good_json()])
    out = expand_query(QUERY, chain(gemini, ollama), LLMConfig())
    assert (out.hyde, out.anti_hyde) == (GOOD_HYDE, GOOD_ANTI)
    assert out.provider == "gemini" and out.model == "gemini-test"
    assert not out.fallback and not out.degraded and out.attempts == 1
    assert gemini.calls[0]["json_format"] is True
    assert QUERY in gemini.calls[0]["prompt"]
    assert ollama.calls == []


def test_retries_once_after_malformed_output() -> None:
    client = FakeOllama(["esto no es json", good_json()])
    out = expand_query(QUERY, only(client), LLMConfig(retries=1))
    assert not out.fallback and out.attempts == 2


def test_gemini_down_falls_back_to_ollama() -> None:
    gemini, ollama = FakeOllama([GeminiError("503 UNAVAILABLE")]), FakeOllama([good_json()])
    providers = chain(gemini, ollama)
    out = expand_query(QUERY, providers, LLMConfig())
    assert out.provider == "ollama" and out.degraded and not out.fallback
    assert "gemini/gemini-test" in out.fallback_reason
    assert len(gemini.calls) == 1                # un error de conexión no se reintenta
    # Gemini queda en enfriamiento: la siguiente consulta va directo a Ollama.
    ollama.responses.append(good_json())
    expand_query(QUERY, providers, LLMConfig())
    assert len(gemini.calls) == 1 and not providers[0].available


def test_gemini_malformed_twice_then_ollama() -> None:
    gemini, ollama = FakeOllama(['{"hyde": "corto"}', "{"]), FakeOllama([good_json()])
    out = expand_query(QUERY, chain(gemini, ollama), LLMConfig(retries=1))
    assert out.provider == "ollama" and out.attempts == 3
    assert len(gemini.calls) == 2


def test_template_fallback_when_everything_fails() -> None:
    gemini = FakeOllama([GeminiError("sin red")])
    ollama = FakeOllama(['{"hyde": "corto"}', OllamaError("timeout")])
    out = expand_query(QUERY, chain(gemini, ollama), LLMConfig(retries=1))
    assert out.fallback and out.degraded and out.provider == "plantilla"
    assert out.hyde == QUERY
    assert "ruidoso" in out.anti_hyde            # opuesto de "tranquilo"
    assert len(out.errors) == 3


def test_empty_chain_uses_template() -> None:
    out = expand_query(QUERY, [], LLMConfig())
    assert out.fallback and out.provider == "plantilla"


@pytest.mark.parametrize("raw", [
    "[]", '{"hyde": 3, "anti_hyde": "x"}', '{"hyde": "a b c", "anti_hyde": "d e f"}',
    '{"hyde": "%s", "anti_hyde": "%s"}' % (GOOD_HYDE, GOOD_HYDE),
])
def test_parse_rejects_bad_output(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_expansion(raw)


def test_parse_ignores_extra_keys() -> None:
    raw = json.dumps({"busca": ["a"], "opuesto": ["b"], "hyde": GOOD_HYDE, "anti_hyde": GOOD_ANTI})
    assert parse_expansion(raw) == (GOOD_HYDE, GOOD_ANTI)


def test_antonym_template() -> None:
    text = antonym_anti_hyde("Busco fiesta y playa")
    assert "sin bares" in text and "sierra" in text
    assert "sin identidad" in antonym_anti_hyde("algo distinto")


def test_gemini_client_requires_key_and_hides_it() -> None:
    with pytest.raises(GeminiError):
        GeminiClient(None)
    assert "secreta" not in repr(GeminiClient("clave-secreta"))


def test_chain_without_key_is_ollama_only(monkeypatch) -> None:
    cfg = Config()
    cfg.llm.api_key_env = "RESTMEX_TEST_NO_KEY"
    monkeypatch.delenv("RESTMEX_TEST_NO_KEY", raising=False)
    assert [p.name for p in build_llm_chain(cfg)] == ["ollama"]
    cfg.llm.provider = "ollama"
    assert [p.model for p in build_llm_chain(cfg)] == [cfg.llm.ollama_model]


def test_chain_with_key_orders_gemini_first(monkeypatch) -> None:
    cfg = Config()
    cfg.llm.api_key_env = "RESTMEX_TEST_KEY"
    cfg.llm.gemini_fallback_models = ["gemini-b"]
    monkeypatch.setenv("RESTMEX_TEST_KEY", "x")
    names = [(p.name, p.model) for p in build_llm_chain(cfg)]
    assert names == [("gemini", cfg.llm.gemini_model), ("gemini", "gemini-b"), ("ollama", cfg.llm.ollama_model)]


def test_label_archetypes_batch_local_and_keywords(index, cfg) -> None:
    meta = index.meta                                # 4 pueblos × 2 arquetipos
    for m in meta:
        m["keywords"], m["type_mix"] = ["tacos", "pescado", "ceviche"], {"Restaurant": 1.0}
    two = json.dumps({"etiquetas": ["Mariscos", "Zona arqueológica"]})
    one = json.dumps({"etiquetas": ["Fondas"]})
    bad = json.dumps({"etiquetas": ["solo una"]})
    gemini = FakeOllama([two, bad, bad, GeminiError("503")])       # pueblo 1 ok, 2 mal formado, 3 caído
    ollama = FakeOllama([one, "mal", "mal", one, one])              # pueblo 2: 1 ok + 1 mal formado; pueblo 3: 2 ok
    n = label_archetypes(meta, chain(gemini, ollama), cfg)
    assert n == 2 + 1 + 2 + 0
    assert meta[0]["label"] == "Mariscos" and meta[0]["label_source"] == "gemini:gemini-test"
    assert meta[2]["label"] == "Fondas" and meta[2]["label_source"] == "ollama:qwen-test"
    assert meta[3]["label"] == "Tacos, pescado, ceviche" and meta[3]["label_source"] == "keywords"
    assert meta[4]["label_source"] == "ollama:qwen-test"
    assert meta[6]["label_source"] == "keywords"     # sin respuestas: palabras clave


def test_quota_error_sets_cooldown_from_retry_after() -> None:
    assert parse_retry_after("Please retry in 5h31m22.6s.") == pytest.approx(5 * 3600 + 31 * 60 + 22.6)
    assert parse_retry_after("Please retry in 41.5s.") == pytest.approx(41.5)
    assert parse_retry_after("otro error") is None
    err = GeminiError("429 RESOURCE_EXHAUSTED")
    err.retry_after = 7200.0
    providers = chain(FakeOllama([err]), FakeOllama([good_json()]))
    out = expand_query(QUERY, providers, LLMConfig())
    assert out.provider == "ollama"
    assert providers[0].seconds_until_available > 7000   # cuota diaria: no se reintenta en horas


def test_recommender_keeps_working_in_fallback(index, cfg) -> None:
    rec = Recommender(cfg, index, encoder=FakeEncoder(), client=FakeOllama([]), providers=only(FakeOllama([])))
    out = rec.recommend("Quiero ruinas, muchas ruinas", lam=0.0)
    assert out.expansion.fallback
    assert out.top[0].town == "Arqueo"
    assert len(out.top) + len(out.bottom) == 4
    data = out.to_dict()
    assert data["fallback"] is True and data["llm_provider"] == "plantilla"
    assert set(data["timings_ms"]) == {"llm", "embedding", "busqueda", "total"}
