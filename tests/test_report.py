"""Reporte con el protocolo del equipo: métricas portadas, enmascarado, validación y suites."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from restmex import report as rp
from restmex.app import main as app_main
from restmex.llm import LLMCache, LLMProvider
from restmex.ranking import rank_towns
from restmex.recommender import Recommender
from restmex.validation import QueryError, validate_query

from conftest import FakeEncoder, FakeOllama, good_json


class EndlessLLM(FakeOllama):
    """LLM simulado que siempre responde el mismo JSON válido."""

    def generate(self, model: str, prompt: str, **kwargs) -> str:
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        return good_json()


def make_pipeline(index, cfg, cache=None, **kwargs) -> tuple[rp.Pipeline, EndlessLLM]:
    cfg.ranking.top_k = cfg.ranking.bottom_k = 2          # el índice de prueba tiene 4 pueblos
    fake = EndlessLLM()
    rec = Recommender(cfg, index, encoder=FakeEncoder(), client=fake,
                      providers=[LLMProvider("groq", "modelo-test", fake)], cache=cache)
    centroids = np.eye(4, dtype=np.float32)
    return rp.Pipeline(rec, centroids, sleep=lambda _s: None, **kwargs), fake


GOLD = [
    {"id": "ruinas", "category": "arqueologia", "query": "ruinas tranquilas y comida", "relevant": ["Arqueo"]},
    {"id": "frio", "category": "montana", "query": "ruinas con clima frío, sin playa",
     "relevant": ["Frio", "Arqueo"], "excluded": ["Playa"]},
]


# ---------------------------------------------------------------- métricas del equipo
def test_ranking_metrics_matches_team_definitions() -> None:
    ranking = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    m = rp.ranking_metrics(ranking, relevant=["b", "j", "z"], excluded=["c"])
    assert m["precision_at_5"] == pytest.approx(0.2)
    assert m["capped_recall_at_5"] == pytest.approx(1 / 3)      # hits / min(5, |R|)
    assert (m["hit_at_1"], m["hit_at_5"], m["mrr"]) == (0.0, 1.0, 0.5)
    dcg, idcg = 1 / np.log2(3), 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert m["ndcg_at_5"] == pytest.approx(dcg / idcg)
    assert m["relevant_in_bottom"] == 1.0 and m["exclusion_violation"] == 1.0
    # Con un solo relevante, acertarlo vale recall acotado 1.0.
    assert rp.ranking_metrics(ranking, ["a"])["capped_recall_at_5"] == 1.0


def test_aggregate_wide_precision_and_exclusion_rate() -> None:
    wide = {**rp.ranking_metrics(list("abcdefghij"), list("abcxy")), "has_exclusion": False}
    narrow = {**rp.ranking_metrics(list("abcdefghij"), ["j"], ["a"]), "has_exclusion": True}
    agg = rp.aggregate([wide, narrow])
    assert agg["wide_queries"] == 1 and agg["precision_at_5_wide"] == 0.6   # solo consultas con ≥5 relevantes
    assert agg["precision_at_5"] == 0.3
    assert agg["bottom_error_rate"] == 0.5 and agg["exclusion_violation_rate"] == 1.0


def test_compose_ranking_matches_rank_towns(index) -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        hyde, anti = rng.normal(size=(2, 4)).astype(np.float32)
        top, bottom = rank_towns(index, hyde, anti, lam=0.05, top_k=2, bottom_k=2)
        sims, _ = index.town_scores(np.vstack([hyde, anti]))
        adjusted = sims[0] - rp.popularity_penalty(index.n_reviews, index.n_max, 0.05)
        order = [index.towns[t] for t in rp.compose_ranking(adjusted, sims[1], 2, 2)]
        assert order[:2] == [t.town for t in top]
        assert order[-2:][::-1] == [t.town for t in bottom]
        assert sorted(order) == sorted(index.towns)
    assert rp.compose_ranking(np.array([0.1, 0.9, 0.5]), None) == [1, 2, 0]


# ---------------------------------------------------------------- holdout
def test_mask_entities_removes_towns_regions_and_nicknames() -> None:
    terms = rp.mask_terms([("TodosSantos", "Baja_CaliforniaSur"), ("Tequisquiapan", "Queretaro"),
                           ("San_Cristobal_de_las_Casas", "Chiapas")])
    assert {"todossantos", "todos", "santos", "queretaro", "tequis", "sancris", "valladolid", "cristobal"} <= set(terms)
    assert "de" not in terms                                  # tokens de menos de 3 letras no se enmascaran
    masked = rp.mask_entities("Fuimos a Todos Santos y a Tequis, en Querétaro. ¡Sancris y Tequisquiapan!", terms)
    assert masked == "fuimos a y a en y"
    assert rp.mask_entities("El viñedo estuvo increíble", terms) == "el vinedo estuvo increible"


def test_holdout_sample_is_stratified_and_deterministic() -> None:
    rng = np.random.default_rng(1)
    val = pd.DataFrame({
        "town": np.repeat(["A", "B", "C"], 50),
        "polarity": rng.choice([1.0, 3.0, 4.0, 5.0], size=150),
        "review_len": rng.integers(20, 900, size=150),
        "text": [f"reseña {i}" for i in range(150)],
    })
    a, b = rp.holdout_sample(val, per_town=5, seed=2025), rp.holdout_sample(val, per_town=5, seed=2025)
    assert a["text"].tolist() == b["text"].tolist()
    assert a["town"].value_counts().to_dict() == {"A": 5, "B": 5, "C": 5}
    assert (a["polarity"] >= 4).all() and a["review_len"].between(80, 600).all()
    assert rp.holdout_sample(val, per_town=5, seed=7)["text"].tolist() != a["text"].tolist()
    assert rp.balanced_subset(["A", "A", "A", "B", "B", "C"], 4) == [0, 1, 3, 5]


# ---------------------------------------------------------------- validación
@pytest.mark.parametrize("query", ["", "   ", "playa", "sin playa, sin bosque, sin comida", "de la que el en", "12345 67890"])
def test_validate_query_rejects(query: str) -> None:
    with pytest.raises(QueryError):
        validate_query(query)


@pytest.mark.parametrize("query", [
    "🏖️ 🌊 playa mar", "<script>alert(1)</script> playa con snorkel", "cenotes y comida yucateca sin Valladolid",
    "Sin duda quiero la mejor playa", "Naturaleza y aventura en la sierra, sin playa ni mar",
])
def test_validate_query_accepts(query: str) -> None:
    assert validate_query(f"  {query} ") == " ".join(query.split())


def test_recommender_rejects_before_calling_the_llm(index, cfg) -> None:
    pipe, fake = make_pipeline(index, cfg)
    with pytest.raises(QueryError, match="dos palabras"):
        pipe.rec.recommend("playa")
    with pytest.raises(QueryError, match="exclusiones"):
        pipe.rec.recommend("sin playa, sin bosque")
    assert fake.calls == []


def test_api_rejects_invalid_queries(index, cfg, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RESTMEX_ARTIFACTS", str(tmp_path))
    with TestClient(app_main.app) as client:
        pipe, fake = make_pipeline(index, cfg)
        app_main.app.state.recommender = pipe.rec
        for query, text in [("playa", "dos palabras"), ("sin playa, sin bosque", "exclusiones"), ("12345 67890", "rasgo")]:
            resp = client.post("/recommend", json={"query": query})
            assert resp.status_code == 422 and text in resp.json()["detail"]
        assert fake.calls == []
        assert client.post("/recommend", json={"query": "ruinas tranquilas"}).status_code == 200


# ---------------------------------------------------------------- suites
def test_gold_suite_rubric_and_markdown(index, cfg, tmp_path) -> None:
    pipe, fake = make_pipeline(index, cfg, cache=LLMCache(tmp_path))
    gold, runs = rp.gold_suite(pipe, GOLD, {"ruinas": "dev", "frio": "test"})
    assert gold["contract_pass_rate"] == 1.0 and gold["pipeline_consistency"] == 1.0
    assert gold["latency_s"]["cold"]["n"] == 2 and gold["latency_s"]["warm"]["n"] == 2
    assert len(fake.calls) == 2 and pipe.new_calls == 2          # una llamada por consulta; el resto, caché
    assert gold["queries"][0]["top"][0] == "Arqueo" and gold["aggregate"]["hit_at_1"] == 1.0
    assert set(gold["by_split"]) == {"dev", "test"}

    determinism = rp.determinism_suite(pipe, GOLD)
    assert determinism["identical_rate"] == 1.0 and determinism["llm_cache"] and len(determinism["ranking_fingerprint"]) == 64
    checks = rp.rubric(gold, determinism)
    assert [c["threshold"] for c in checks] == [0.7, 0.8, 1.0, 1.0, 3.0, 3.0]
    assert checks[1]["pass"] and checks[2]["pass"] and checks[4]["pass"]

    items = [(r, q["relevant"], q.get("excluded", [])) for r, q in zip(runs, GOLD)]
    ablation = rp.ablation_suite(pipe, items)
    assert list(ablation) == list(rp.ABLATIONS)
    baselines = rp.baseline_suite(pipe, GOLD)
    assert baselines["popularity"]["queries"] == 2 and baselines["random"]["queries"] == 400
    bias = rp.volume_bias_suite(pipe, gold["queries"])
    assert sum(bias["exposure"].values()) == 4

    adversarial = rp.adversarial_suite(pipe)
    outcome = {c["case"]: c for c in adversarial["cases"]}
    assert outcome["vacía"]["pass"] and outcome["una palabra"]["pass"] and outcome["todo negado"]["pass"]
    assert outcome["solo números"]["pass"] and not outcome["fuera de vocabulario"]["pass"]

    report = {
        "meta": {
            "encoder": "fake", "k": 2, "n_archetypes": 8, "n_towns": 4, "lambda": 0.015, "llm_chain": "groq",
            "temperature": 0.4, "llm_used": "groq × 2", "llm_new_calls": 2, "llm_cached": 0, "commit": "abc",
            "generated_at": "hoy", "duration_s": 1.0, "config": "config.yaml", "team_commit": "3ecbcfd",
            "gold_sha256": "0" * 64, "gold_provenance": {"author": "x", "saw_system_output": True},
            "holdout_per_town": 15, "holdout_seed": 2025,
        },
        "rubric": checks, "gold": gold, "baselines": baselines, "ablation": ablation,
        "determinism": determinism, "volume_bias": bias, "adversarial": adversarial,
        "robustness": rp.robustness_suite(pipe, gold["queries"]),
    }
    text = rp.render_markdown(report)
    for heading in ["## 1. Rúbrica", "## 2. Gold", "## 3. Holdout", "## 4. Ablación", "## 5. Sesgo", "## 6. Robustez",
                    "## 7. Casos adversariales", "## 8. Determinismo", "## 9. Consultas con fallos"]:
        assert heading in text
    assert "temperature 0.4" in text


def test_budget_leaves_queries_pending_and_estimate_counts_cache(index, cfg, tmp_path) -> None:
    pipe, fake = make_pipeline(index, cfg, cache=LLMCache(tmp_path), max_new_calls=1)
    assert pipe.run("ruinas tranquilas y comida") is not None
    assert pipe.run("playa con fiesta y antros") is None and pipe.pending == 1     # presupuesto agotado
    assert pipe.run("ruinas tranquilas y comida").expansion.cached                 # la caché no gasta presupuesto
    assert len(fake.calls) == 1

    plan = rp.plan_queries(GOLD, [{"id": "x", "query": "playa con fiesta y antros"}, {"id": "y", "query": "playa"}],
                           ["texto enmascarado uno"], [], robustness_n=1)
    est = rp.estimate_calls(pipe, plan)
    assert est["sections"]["gold"] == {"queries": 2, "new_calls": 1}               # una ya está en caché
    assert est["sections"]["banco de consultas"] == {"queries": 1, "new_calls": 1}  # "playa" se rechaza antes
    assert est["sections"]["robustez (3 variantes)"]["queries"] == 3
    assert est["new_calls"] == sum(s["new_calls"] for s in est["sections"].values())
    assert est["estimated_tokens"] == est["new_calls"] * rp.TOKENS_PER_CALL and est["fits_tokens_per_day"]

    exported = rp.bank_rankings(pipe, [{"id": "a", "query": "ruinas tranquilas y comida", "split": "dev"},
                                       {"id": "b", "query": "playa"}, {"id": "c", "query": "sierra con cabañas y frío"}])
    assert list(exported["rankings"]) == ["a"] and "b" in exported["rejected"] and exported["pending"] == ["c"]
    entry = exported["rankings"]["a"]
    assert entry["top"] == entry["ranking"][:2] and entry["bottom"] == entry["ranking"][-2:][::-1]


def test_holdout_suite_with_fakes(index, cfg, tmp_path) -> None:
    pipe, _ = make_pipeline(index, cfg, cache=LLMCache(tmp_path))
    sample = pd.DataFrame({
        "town": ["Arqueo", "Arqueo", "Frio", "Playa"],
        "text": ["Las ruinas de Arqueo son enormes, ruinas por todos lados", "Ruinas y más ruinas",
                 "Hace mucho frío en Frio, frío de verdad", "La playa es preciosa, playa limpia"],
    })
    terms = rp.mask_terms([(t, "Estado") for t in index.towns])
    out = rp.holdout_suite(pipe, sample, terms, hyde_n=2)
    # "Arqueo", "Frio" y "Playa" son nombres de pueblo en el índice de prueba: se enmascaran.
    assert out["evaluated_queries"] == 4 and out["changed_by_mask"] == 0.75
    doc = out["variants"]["reseña como documento (sin LLM)"]
    assert doc["n"] == 4 and doc["metrics"]["hit_at_5"] >= 0.5
    leak = out["leak"]["canónico sin enmascarar"]["reseña como documento (sin LLM)"]
    assert leak["hit_at_1"] >= doc["metrics"]["hit_at_1"]
    hyde = out["variants"]["HyDE + Anti-HyDE (pipeline web)"]
    assert hyde["n"] == 2 and hyde["requested"] == 2 and hyde["pending"] == 0
    assert set(out["leak"]) == {"original (sin enmascarar)", "canónico sin enmascarar", "canónico enmascarado"}
    assert list(out["ablation_hyde_subset"]) == list(rp.ABLATIONS)
    assert out["random_expectation"]["hit_at_5"] == 0.5          # k = 2 de 4 pueblos en la prueba
