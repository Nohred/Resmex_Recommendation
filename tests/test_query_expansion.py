"""HyDE / Anti-HyDE con LLM simulados: validación, reintento y cadena de respaldo."""
from __future__ import annotations

import json

import pytest

from restmex.archetypes import label_archetypes
from restmex.config import Config, LLMConfig
from restmex.llm import (
    GeminiClient, GeminiError, GroqClient, GroqError, LLMCache, LLMProvider, build_llm_chain, parse_retry_after,
)
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


def test_clients_require_key_and_hide_it() -> None:
    with pytest.raises(GeminiError):
        GeminiClient(None)
    with pytest.raises(GroqError):
        GroqClient("")
    assert "secreta" not in repr(GeminiClient("clave-secreta"))
    assert "secreta" not in repr(GroqClient("clave-secreta"))


def test_chain_follows_config_and_skips_missing_keys(monkeypatch) -> None:
    cfg = Config()
    cfg.llm.gemini_api_key_env, cfg.llm.groq_api_key_env = "RESTMEX_T_GEM", "RESTMEX_T_GROQ"
    monkeypatch.delenv("RESTMEX_T_GEM", raising=False)
    monkeypatch.setenv("RESTMEX_T_GROQ", "x")
    names = [(p.name, p.model, p.rpm) for p in build_llm_chain(cfg)]
    assert names == [("groq", cfg.llm.groq_model, cfg.llm.groq_rpm), ("ollama", cfg.llm.ollama_model, 0)]
    monkeypatch.setenv("RESTMEX_T_GEM", "x")
    assert [p.name for p in build_llm_chain(cfg)] == ["gemini", "groq", "ollama"]
    assert [p.name for p in build_llm_chain(cfg, names=cfg.llm.eval_chain)] == ["groq", "ollama"]
    with pytest.raises(ValueError):
        build_llm_chain(cfg, names=["otro"])


def test_groq_client_request_and_errors() -> None:
    class Resp:
        def __init__(self, status, body, headers=None):
            self.status_code, self._body, self.headers, self.text = status, body, headers or {}, str(body)

        def json(self):
            return self._body

    class Session:
        def __init__(self, resp):
            self.resp, self.sent = resp, None

        def post(self, url, json, timeout, headers):
            self.sent = {"url": url, "json": json, "headers": headers}
            return self.resp

    client = GroqClient("k")
    client._session = Session(Resp(200, {"choices": [{"message": {"content": "{}"}}]}))
    assert client.generate("m", "hola", system="sys", json_format=True) == "{}"
    sent = client._session.sent
    assert sent["url"].endswith("/openai/v1/chat/completions")
    assert sent["json"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in sent["json"]["messages"]] == ["system", "user"]

    client._session = Session(Resp(429, {"error": {"message": "Rate limit reached"}}, {"retry-after": "7"}))
    with pytest.raises(GroqError) as e:
        client.generate("m", "hola")
    assert e.value.status == 429 and e.value.retry_after == 7.0 and not e.value.daily

    msg = "Rate limit reached on tokens per day (TPD): Limit 200000. Please try again in 7m12s."
    client._session = Session(Resp(429, {"error": {"message": msg}}))
    with pytest.raises(GroqError) as e:
        client.generate("m", "hola")
    assert e.value.daily and e.value.retry_after == pytest.approx(432)


def test_parse_retry_after() -> None:
    assert parse_retry_after("Please retry in 5h31m22.6s.") == pytest.approx(5 * 3600 + 31 * 60 + 22.6)
    assert parse_retry_after("Please retry in 41.5s.") == pytest.approx(41.5)
    assert parse_retry_after("otro error") is None


def quota_error(daily: bool = False, retry_after: float | None = None, status: int = 429) -> GeminiError:
    err = GeminiError(f"{status}")
    err.status, err.daily, err.retry_after = status, daily, retry_after
    return err


def test_daily_quota_marks_provider_exhausted_for_session() -> None:
    gemini, ollama = FakeOllama([quota_error(daily=True)]), FakeOllama([good_json(), good_json()])
    providers = chain(gemini, ollama)
    out = expand_query(QUERY, providers, LLMConfig())
    assert out.provider == "ollama" and providers[0].exhausted and not providers[0].available
    assert "cuota diaria agotada" in out.fallback_reason
    expand_query(QUERY, providers, LLMConfig())
    assert len(gemini.calls) == 1                # agotado: no se reintenta


def test_transient_503_is_retried_with_backoff(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("restmex.llm.time.sleep", sleeps.append)
    gemini = FakeOllama([quota_error(status=503), good_json()])
    out = expand_query(QUERY, chain(gemini, FakeOllama([])), LLMConfig())
    assert out.provider == "gemini" and sleeps == [1.5]
    # En modo lote se respeta el retry_after de un 429 por minuto.
    sleeps.clear()
    gemini = FakeOllama([quota_error(retry_after=20.0), good_json()])
    out = expand_query(QUERY, chain(gemini, FakeOllama([])), LLMConfig(), batch=True)
    assert out.provider == "gemini" and sleeps == [20.0]
    # En la web no se espera 20 s: se pasa al siguiente y Gemini se enfría ese tiempo.
    providers = chain(FakeOllama([quota_error(retry_after=20.0)]), FakeOllama([good_json()]))
    out = expand_query(QUERY, providers, LLMConfig())
    assert out.provider == "ollama" and not providers[0].available and not providers[0].exhausted


def test_rate_limiter(monkeypatch) -> None:
    prov = LLMProvider("groq", "m", FakeOllama([good_json()] * 3), rpm=2)
    assert prov.acquire(wait=False) and prov.acquire(wait=False)
    assert not prov.acquire(wait=False)          # tercer llamado en el mismo minuto
    out = expand_query(QUERY, [prov, LLMProvider("ollama", "q", FakeOllama([good_json()]))], LLMConfig())
    assert out.provider == "ollama" and "límite de solicitudes" in out.fallback_reason
    prov._calls[0] -= 61                         # pasa el tiempo: se libera un cupo
    assert prov.acquire(wait=False)


def test_cache_avoids_second_call(tmp_path) -> None:
    cache = LLMCache(tmp_path / "cache")
    gemini = FakeOllama([good_json()])
    providers = chain(gemini, FakeOllama([]))
    first = expand_query(QUERY, providers, LLMConfig(), cache)
    second = expand_query(QUERY, providers, LLMConfig(), cache)
    assert not first.cached and second.cached and second.attempts == 0
    assert second.hyde == first.hyde and second.provider == "gemini"
    assert len(gemini.calls) == 1
    # Otra consulta sí llama; una salida mal formada nunca se guarda.
    other = expand_query("otra consulta distinta", chain(FakeOllama(["{"]), FakeOllama([])), LLMConfig(retries=0), cache)
    assert other.fallback and len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_label_archetypes_batch_local_and_keywords(index, cfg) -> None:
    meta = index.meta                                # 4 pueblos × 2 arquetipos
    for m in meta:
        m["keywords"], m["type_mix"] = ["tacos", "pescado", "ceviche"], {"Restaurant": 1.0}
    two = json.dumps({"etiquetas": ["Mariscos", "Zona arqueológica"]})
    one = json.dumps({"etiquetas": ["Fondas"]})
    bad = json.dumps({"etiquetas": ["solo una"]})
    gemini = FakeOllama([two, bad, bad, GeminiError("sin red")])    # pueblo 1 ok, 2 mal formado, 3 caído
    ollama = FakeOllama([one, "mal", "mal", one, one])              # pueblo 2: 1 ok + 1 mal formado; pueblo 3: 2 ok
    n = label_archetypes(meta, chain(gemini, ollama), cfg)
    assert n == 2 + 1 + 2 + 0
    assert meta[0]["label"] == "Mariscos" and meta[0]["label_source"] == "gemini:gemini-test"
    assert meta[2]["label"] == "Fondas" and meta[2]["label_source"] == "ollama:qwen-test"
    assert meta[3]["label"] == "Tacos, pescado, ceviche" and meta[3]["label_source"] == "keywords"
    assert meta[4]["label_source"] == "ollama:qwen-test"
    assert meta[6]["label_source"] == "keywords"     # sin respuestas: palabras clave


def test_recommender_keeps_working_in_fallback(index, cfg) -> None:
    rec = Recommender(cfg, index, encoder=FakeEncoder(), client=FakeOllama([]), providers=only(FakeOllama([])))
    out = rec.recommend("Quiero ruinas, muchas ruinas", lam=0.0)
    assert out.expansion.fallback
    assert out.top[0].town == "Arqueo"
    assert len(out.top) + len(out.bottom) == 4
    data = out.to_dict()
    assert data["fallback"] is True and data["llm_provider"] == "plantilla"
    assert set(data["timings_ms"]) == {"llm", "embedding", "busqueda", "total"}
