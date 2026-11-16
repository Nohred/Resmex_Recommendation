"""Reporte de evaluación con el protocolo del equipo (repo `restmex-orbita`).

Las definiciones de métricas (`ranking_metrics`, `aggregate`), la rúbrica, las líneas base,
las variantes de robustez, los casos adversariales y el enmascarado del holdout se portan de
`src/restmex_recommender/evaluation.py` del equipo, para que los números sean comparables.
Lo que cambia es el sistema evaluado: aquí el ranking sale de HyDE / Anti-HyDE + arquetipos.

Ranking completo de una consulta (40 pueblos), tal como lo expone el sistema:
    Top 5 (score ajustado) + pueblos intermedios por score ajustado + Bottom 5 al final,
con el más parecido al Anti-HyDE en la última posición. Así `ranking[:5]` y `ranking[-5:]`
son exactamente el Top y el Bottom de la web y las métricas del equipo se aplican sin cambios.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .query_expansion import QueryExpansion, build_prompt, expand_query, parse_expansion
from .prompts import HYDE_SYSTEM
from .ranking import popularity_penalty
from .recommender import Recommender
from .validation import QueryError, canonical_text, validate_query

RUBRIC = {
    "precision_at_5": 0.70,
    "hit_at_1": 0.80,
    "contract": 1.0,
    "determinism": 1.0,
    "latency_p95_s": 3.0,
}

# Capa gratuita de Groq para openai/gpt-oss-120b (docs de Groq y cabeceras x-ratelimit-*
# de la cuenta, 2026-11-16). Una llamada de HyDE gasta ~1,200 tokens (medido: 1,160).
GROQ_FREE = {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200_000}
TOKENS_PER_CALL = 1200

METRIC_KEYS = [
    "precision_at_5", "capped_recall_at_5", "hit_at_1", "hit_at_5", "mrr", "ndcg_at_5",
    "bottom_error_rate", "exclusion_violation_rate",
]


# ---------------------------------------------------------------- métricas por consulta
def ranking_metrics(
    ranking: Sequence[str],
    relevant: Iterable[str],
    excluded: Iterable[str] = (),
    k: int = 5,
) -> dict:
    """Métricas binarias sobre un ranking completo de destinos (definición del equipo)."""
    relevant, excluded = set(relevant), set(excluded)
    top, bottom = list(ranking[:k]), list(ranking[-k:])
    hits = [1.0 if town in relevant else 0.0 for town in top]
    first = next((i for i, t in enumerate(ranking) if t in relevant), None)
    ideal = min(k, len(relevant))
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return {
        "precision_at_5": sum(hits) / k,
        "recall_at_5": sum(hits) / len(relevant) if relevant else 0.0,
        # Recall acotado: hits / min(5, |R|). Con 1 relevante, acertarlo vale 1.0.
        "capped_recall_at_5": sum(hits) / ideal if ideal else 0.0,
        "relevant_count": float(len(relevant)),
        "hit_at_1": hits[0] if hits else 0.0,
        "hit_at_5": 1.0 if any(hits) else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg_at_5": dcg / idcg if idcg else 0.0,
        "relevant_in_bottom": float(len(relevant.intersection(bottom))),
        "exclusion_violation": 1.0 if excluded.intersection(top) else 0.0,
    }


def _mean(rows: list[dict], key: str) -> float:
    return round(float(np.mean([r[key] for r in rows])), 4) if rows else 0.0


def aggregate(rows: list[dict]) -> dict:
    """Promedios por consulta (definición del equipo)."""
    keys = [
        "precision_at_5", "recall_at_5", "capped_recall_at_5", "hit_at_1", "hit_at_5", "mrr",
        "ndcg_at_5",
    ]
    out = {key: _mean(rows, key) for key in keys}
    # Precision@5 solo es comparable con el umbral cuando hay al menos 5 relevantes posibles.
    wide = [r for r in rows if r.get("relevant_count", 0) >= 5]
    out["precision_at_5_wide"] = _mean(wide, "precision_at_5")
    out["wide_queries"] = len(wide)
    out["bottom_error_rate"] = (
        round(float(np.mean([r["relevant_in_bottom"] > 0 for r in rows])), 4) if rows else 0.0
    )
    with_exclusion = [r for r in rows if r.get("has_exclusion")]
    out["exclusion_violation_rate"] = _mean(with_exclusion, "exclusion_violation")
    out["queries"] = len(rows)
    return out


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    return round(float(np.percentile(values, q)), 4) if len(values) else None


def _latency(values: Sequence[float]) -> dict:
    return {
        "n": len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": round(float(max(values)), 4) if len(values) else None,
    }


def _spearman(a: Sequence[float], b: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None, None
    rho, p = spearmanr(a, b)
    return round(float(rho), 4), round(float(p), 4)


# ---------------------------------------------------------------- ranking completo
def compose_ranking(
    adjusted: np.ndarray, sim_anti: Optional[np.ndarray], top_k: int = 5, bottom_k: int = 5
) -> list[int]:
    """Orden completo de pueblos con la misma regla que `ranking.rank_towns`.

    Con `sim_anti`: Top por score ajustado; Bottom = más parecidos al Anti-HyDE fuera del
    Top, colocados al final (el peor en la última posición); el resto, por score ajustado.
    Sin `sim_anti`: orden por score ajustado (el Bottom es el reverso del ranking).
    """
    order = [int(t) for t in np.argsort(-np.asarray(adjusted), kind="stable")]
    if sim_anti is None:
        return order
    top = order[:top_k]
    in_top = set(top)
    bottom = [int(t) for t in np.argsort(-np.asarray(sim_anti), kind="stable") if int(t) not in in_top][:bottom_k]
    in_bottom = set(bottom)
    middle = [t for t in order[top_k:] if t not in in_bottom]
    return top + middle + bottom[::-1]


@dataclass
class QueryRun:
    """Una consulta ya expandida y codificada."""

    query: str
    expansion: QueryExpansion
    hyde_vec: np.ndarray
    anti_vec: np.ndarray
    raw_vec: Optional[np.ndarray] = None       # consulta cruda con el prefijo de consulta


class Pipeline:
    """El recomendador de la web más lo necesario para evaluarlo: ranking completo,
    variantes de ablación y control de las llamadas nuevas al LLM (presupuesto y pausa)."""

    def __init__(
        self,
        rec: Recommender,
        centroids: Optional[np.ndarray] = None,
        max_new_calls: Optional[int] = None,
        gap: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rec = rec
        self.cfg = rec.cfg
        self.index = rec.index
        self.centroids = centroids
        self.max_new_calls = max_new_calls
        self.gap = gap
        self._sleep = sleep
        self._last_call = 0.0
        self.new_calls = 0                       # consultas que necesitaron al LLM en esta corrida
        self.cached_calls = 0
        self.pending = 0                         # consultas omitidas por presupuesto o sin LLM
        self.providers_used: Counter = Counter()
        self.k = self.cfg.ranking.top_k
        self.bottom_k = self.cfg.ranking.bottom_k

    # -- LLM
    def is_cached(self, query: str) -> bool:
        """¿Algún proveedor de la cadena ya respondió (bien) el prompt de esta consulta?"""
        if self.rec.cache is None:
            return False
        prompt = build_prompt(query)
        for prov in self.rec.providers:
            raw = self.rec.cache.get(prov.name, prov.model, HYDE_SYSTEM, prompt)
            if raw is not None:
                try:
                    parse_expansion(raw)
                    return True
                except ValueError:
                    continue
        return False

    def guard(self, query: str, force: bool = False) -> bool:
        """Antes de una consulta: `False` si necesita al LLM y ya no hay presupuesto.

        Si hará una llamada nueva, espera `gap` segundos desde la anterior (fuera de
        cualquier medición de latencia) para no chocar con el límite de tokens por minuto.
        """
        if self.is_cached(query):
            return True
        if not force and self.max_new_calls is not None and self.new_calls >= self.max_new_calls:
            self.pending += 1
            return False
        wait = self.gap - (time.time() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        return True

    def note(self, expansion: QueryExpansion) -> None:
        """Registra de dónde salió una expansión."""
        if expansion.cached:
            self.cached_calls += 1
        else:
            self.new_calls += 1
            self._last_call = time.time()
        self.providers_used[llm_tag(expansion)] += 1

    def run(self, query: str, force: bool = False) -> Optional[QueryRun]:
        """Expande y codifica igual que `Recommender.recommend`. `None` = pendiente
        (sin presupuesto, o ningún LLM respondió y solo quedaba la plantilla)."""
        if not self.guard(query, force):
            return None
        expansion = expand_query(query, self.rec.providers, self.cfg.llm, self.rec.cache, batch=True)
        self.note(expansion)
        if expansion.fallback:
            self.pending += 1
            return None
        vecs = self.rec.encoder.encode_documents([expansion.hyde, expansion.anti_hyde])
        return QueryRun(expansion.query, expansion, vecs[0], vecs[1])

    def with_raw(self, runs: Sequence[QueryRun]) -> None:
        """Añade a cada consulta su vector crudo (prefijo `search_query`)."""
        if runs:
            for run, vec in zip(runs, self.rec.encoder.encode_queries([r.query for r in runs])):
                run.raw_vec = vec

    # -- ranking
    def sims(self, vecs: np.ndarray, centroid: bool = False) -> np.ndarray:
        """Similitud por pueblo, (Q, T): máximo sobre arquetipos o un centroide por pueblo."""
        q = np.atleast_2d(np.asarray(vecs, dtype=np.float32))
        if centroid:
            if self.centroids is None:
                raise ValueError("Faltan los centroides por pueblo (town_centroids.npy)")
            return q @ self.centroids.T
        return self.index.town_scores(q)[0]

    def adjusted(self, sims: np.ndarray, lam: Optional[float] = None) -> np.ndarray:
        lam = self.cfg.ranking.lam if lam is None else lam
        return np.asarray(sims, dtype=np.float64) - popularity_penalty(self.index.n_reviews, self.index.n_max, lam)

    def ranking(
        self,
        run: QueryRun,
        *,
        lam: Optional[float] = None,
        centroid: bool = False,
        raw: bool = False,
        bottom: str = "anti",
    ) -> list[str]:
        """Ranking completo de una consulta; los argumentos son las palancas de la ablación."""
        vec = run.raw_vec if raw else run.hyde_vec
        adjusted = self.adjusted(self.sims(vec, centroid)[0], lam)
        sim_anti = self.sims(run.anti_vec, centroid)[0] if bottom == "anti" else None
        return [self.index.towns[t] for t in compose_ranking(adjusted, sim_anti, self.k, self.bottom_k)]

    def score_rankings(self, vecs: np.ndarray, lam: Optional[float] = None, centroid: bool = False) -> list[list[str]]:
        """Rankings por score ajustado (sin Anti-HyDE) para un lote de vectores."""
        adjusted = self.adjusted(self.sims(vecs, centroid), lam)
        return [[self.index.towns[t] for t in compose_ranking(row, None)] for row in adjusted]


def llm_tag(expansion: QueryExpansion) -> str:
    return f"{expansion.provider}:{expansion.model}" if expansion.model else expansion.provider


# ---------------------------------------------------------------- gold
def load_gold(path) -> list[dict]:
    return json.loads(open(path, encoding="utf-8").read())["queries"]


def gold_suite(pipe: Pipeline, gold: list[dict], splits: Optional[dict[str, str]] = None) -> tuple[dict, list[QueryRun]]:
    """Pipeline completo de la web sobre el gold: métricas, contrato y latencia.

    La latencia en frío es la de las consultas que no estaban en la caché del LLM al
    empezar (llamada real); la latencia en caliente, una segunda pasada con todo en caché.
    """
    unknown = {t for q in gold for t in q["relevant"] + q.get("excluded", [])} - set(pipe.index.towns)
    if unknown:
        raise ValueError(f"El gold set menciona pueblos inexistentes: {sorted(unknown)}")
    k, rows, runs, cold, warm, contract_ok, consistent = pipe.k, [], [], [], [], [], []
    for item in gold:
        pipe.guard(item["query"], force=True)
        start = time.perf_counter()
        result = pipe.rec.recommend(item["query"], batch=True)
        elapsed = time.perf_counter() - start
        pipe.note(result.expansion)
        if not result.expansion.cached:
            cold.append(elapsed)
        top = [x.town for x in result.top]
        bottom = [x.town for x in result.bottom]
        contract_ok.append(len(top) == k and len(bottom) == pipe.bottom_k and not set(top) & set(bottom))
        run = pipe.run(item["query"], force=True)
        if run is None:                           # ningún LLM respondió: plantilla sin LLM
            exp = result.expansion
            vecs = pipe.rec.encoder.encode_documents([exp.hyde, exp.anti_hyde])
            run = QueryRun(exp.query, exp, vecs[0], vecs[1])
        runs.append(run)
        ranking = pipe.ranking(run)
        consistent.append(ranking[:k] == top and ranking[-pipe.bottom_k:][::-1] == bottom)
        metrics = ranking_metrics(ranking, item["relevant"], item.get("excluded", []), k)
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "split": (splits or {}).get(item["id"]),
                "query": item["query"],
                "top": top,
                "bottom": bottom,
                "ranking": ranking,
                "relevant": item["relevant"],
                "excluded": item.get("excluded", []),
                "has_exclusion": bool(item.get("excluded")),
                "llm": llm_tag(run.expansion),
                "fallback": run.expansion.fallback,
                **{key: round(v, 4) for key, v in metrics.items()},
            }
        )
    for item in gold:                             # segunda pasada: todo sale de la caché
        start = time.perf_counter()
        result = pipe.rec.recommend(item["query"], batch=True)
        elapsed = time.perf_counter() - start
        if result.expansion.cached:
            warm.append(elapsed)
    pipe.with_raw(runs)

    def grouped(key: str) -> dict:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(str(row[key]), []).append(row)
        return {name: aggregate(group) for name, group in sorted(groups.items())}

    result = {
        "aggregate": aggregate(rows),
        "by_category": grouped("category"),
        "by_split": grouped("split") if splits else {},
        "contract_pass_rate": round(float(np.mean(contract_ok)), 4),
        "pipeline_consistency": round(float(np.mean(consistent)), 4),
        "fallback_queries": int(sum(r["fallback"] for r in rows)),
        "latency_s": {"cold": _latency(cold), "warm": _latency(warm)},
        "queries": rows,
    }
    return result, runs


def baseline_suite(pipe: Pipeline, gold: list[dict], seed: int = 2025) -> dict:
    """Referencias mínimas: azar (200 permutaciones por consulta) y popularidad."""
    rng = np.random.default_rng(seed)
    towns = list(pipe.index.towns)
    by_volume = [towns[i] for i in np.argsort(-pipe.index.n_reviews, kind="stable")]
    random_rows, popularity_rows = [], []
    for item in gold:
        rel, exc = item["relevant"], item.get("excluded", [])
        for _ in range(200):
            perm = list(rng.permutation(towns))
            random_rows.append({**ranking_metrics(perm, rel, exc, pipe.k), "has_exclusion": bool(exc)})
        popularity_rows.append({**ranking_metrics(by_volume, rel, exc, pipe.k), "has_exclusion": bool(exc)})
    return {"random": aggregate(random_rows), "popularity": aggregate(popularity_rows)}


ABLATIONS: dict[str, dict[str, Any]] = {
    "completo (arquetipos, λ, HyDE, bottom Anti-HyDE)": {},
    "1 centroide por pueblo": {"centroid": True},
    "λ = 0": {"lam": 0.0},
    "consulta cruda (search_query, sin HyDE)": {"raw": True},
    "bottom = reverso del ranking": {"bottom": "reverso"},
}


def ablation_suite(pipe: Pipeline, items: Sequence[tuple[QueryRun, Sequence[str], Sequence[str]]]) -> dict:
    """Cada fila cambia una sola pieza del sistema completo. `items`: (consulta, relevantes, excluidos)."""
    out = {}
    for name, kwargs in ABLATIONS.items():
        if kwargs.get("centroid") and pipe.centroids is None:
            continue
        rows = [
            {**ranking_metrics(pipe.ranking(run, **kwargs), rel, exc, pipe.k), "has_exclusion": bool(exc)}
            for run, rel, exc in items
        ]
        out[name] = aggregate(rows)
    return out


def volume_bias_suite(pipe: Pipeline, gold_rows: list[dict]) -> dict:
    """¿Los pueblos con más reseñas aparecen más en el top 5? Spearman exposición vs volumen."""
    exposure = {t: 0 for t in pipe.index.towns}
    for row in gold_rows:
        for town in row["top"]:
            exposure[town] += 1
    rho, pvalue = _spearman([exposure[t] for t in pipe.index.towns], list(pipe.index.n_reviews))
    return {
        "spearman_exposure_vs_reviews": rho,
        "p_value": pvalue,
        "max_exposure": list(max(exposure.items(), key=lambda kv: kv[1])),
        "towns_never_in_top5": sorted(t for t, v in exposure.items() if v == 0),
        "exposure": exposure,
    }


# ---------------------------------------------------------------- robustez y adversariales
def query_variants(query: str) -> list[str]:
    """Las tres variantes superficiales del equipo."""
    clauses = [c.strip() for c in re.split(r",| y ", query) if c.strip()]
    return [
        canonical_text(query).upper(),            # sin acentos y en mayúsculas
        "por favor, " + query + ", gracias",      # relleno
        ", ".join(reversed(clauses)),             # orden de cláusulas invertido
    ]


def robustness_suite(pipe: Pipeline, gold_rows: list[dict], n_queries: Optional[int] = None) -> dict:
    """Jaccard del top 5 entre la consulta original y sus variantes."""
    scores, pending = [], 0
    for row in gold_rows[: n_queries if n_queries is not None else len(gold_rows)]:
        base = set(row["top"])
        for variant in query_variants(row["query"]):
            try:
                validate_query(variant)
            except QueryError:
                scores.append(0.0)                # rechazada: top vacío, como en el equipo
                continue
            run = pipe.run(variant)
            if run is None:
                pending += 1
                continue
            other = set(pipe.ranking(run)[: pipe.k])
            scores.append(len(base & other) / len(base | other))
    return {
        "mean_top5_jaccard": round(float(np.mean(scores)), 4) if scores else None,
        "min_top5_jaccard": round(float(np.min(scores)), 4) if scores else None,
        "pairs": len(scores),
        "pending_pairs": pending,
    }


ADVERSARIAL = [
    ("vacía", "", "reject"),
    ("una palabra", "playa", "reject"),
    ("solo stopwords", "de la que el en", "reject_or_low"),
    ("fuera de vocabulario", "xyzzy qwerty plugh", "reject_or_low"),
    ("solo números", "12345 67890", "reject_or_low"),
    ("emoji", "🏖️ 🌊 playa mar", "serve"),
    ("html", "<script>alert(1)</script> playa con snorkel", "serve"),
    ("muy larga", " ".join(["bosque montaña cabañas"] * 300), "serve"),
    ("todo negado", "sin playa, sin bosque, sin comida", "reject"),
    ("pueblo excluido", "cenotes y comida yucateca sin Valladolid", "serve"),
]


def adversarial_suite(pipe: Pipeline) -> dict:
    """Los 10 casos del equipo. Este sistema no tiene nivel de confianza: un caso
    `reject_or_low` solo pasa si la consulta se rechaza."""
    cases = []
    for name, query, expected in ADVERSARIAL:
        try:
            validate_query(query)
            if not pipe.guard(query):
                cases.append({"case": name, "expected": expected, "outcome": "pendiente (sin presupuesto de LLM)", "pass": None})
                continue
            result = pipe.rec.recommend(query, batch=True)
            pipe.note(result.expansion)
            top = [x.town for x in result.top]
            bottom = [x.town for x in result.bottom]
            valid = len(top) == pipe.k and len(bottom) == pipe.bottom_k and not set(top) & set(bottom)
            outcome = "servida (plantilla sin LLM)" if result.expansion.fallback else "servida"
            ok = valid and expected == "serve"
            if name == "pueblo excluido":
                ok = ok and "Valladolid" not in top
                outcome += "; Valladolid en el top 5" if "Valladolid" in top else "; Valladolid fuera del top 5"
        except QueryError as exc:
            outcome, ok = f"rechazada: {exc}", expected != "serve"
        cases.append({"case": name, "expected": expected, "outcome": outcome, "pass": bool(ok)})
    done = [c["pass"] for c in cases if c["pass"] is not None]
    return {"pass_rate": round(float(np.mean(done)), 4) if done else None, "cases": cases}


def determinism_suite(pipe: Pipeline, gold: list[dict]) -> dict:
    """Dos corridas completas con la caché del LLM y huella SHA-256 de los rankings."""
    passes = []
    for _ in range(2):
        runs = [pipe.run(q["query"], force=True) for q in gold]
        passes.append([pipe.ranking(r) if r is not None else [] for r in runs])
        all_cached = all(r is not None and r.expansion.cached for r in runs)
    first, second = passes
    return {
        "identical_rate": round(float(np.mean([a == b for a, b in zip(first, second)])), 4),
        "ranking_fingerprint": hashlib.sha256(json.dumps(first).encode("utf-8")).hexdigest(),
        "llm_cache": all_cached,
        "temperature": pipe.cfg.llm.temperature,
    }


# ---------------------------------------------------------------- holdout enmascarado
NICKNAMES = {"tequis", "cuatrocienegas", "sancris", "valladolid"}


def mask_terms(places: Iterable[tuple[str, str]]) -> list[str]:
    """Términos a borrar: nombres de pueblo y estado (lógica de `_mask_terms` del equipo).

    `places`: pares (pueblo, estado) como vienen en el CSV (`TodosSantos`, `QuintanaRoo`).
    """
    terms = set()
    for town, region in places:
        for raw in (town.replace("_", " "), region.replace("_", " ")):
            spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)     # TodosSantos -> Todos Santos
            for text in (raw, spaced):
                for token in canonical_text(text).split():
                    if len(token) >= 3:
                        terms.add(token)
                terms.add(canonical_text(text).replace(" ", ""))
    terms.update(NICKNAMES)
    return sorted(terms, key=len, reverse=True)


def mask_entities(text: str, terms: Sequence[str]) -> str:
    """Texto canónico (minúsculas, sin acentos ni puntuación) sin los términos dados."""
    canonical = canonical_text(text)
    pattern = r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b"
    return re.sub(r"\s+", " ", re.sub(pattern, " ", canonical)).strip()


def holdout_sample(val: pd.DataFrame, per_town: int = 15, seed: int = 2025) -> pd.DataFrame:
    """Reseñas positivas de longitud media, estratificadas por pueblo (como `holdout_split`).

    `val`: partición de validación con columnas `town`, `polarity`, `review_len` (caracteres
    de la reseña original, sin el título) y `text`.
    """
    eligible = val[(val["polarity"] >= 4) & val["review_len"].between(80, 600)]
    parts = [
        group.sample(min(per_town, len(group)), random_state=seed)
        for _, group in eligible.groupby("town", sort=True)
    ]
    return pd.concat(parts).reset_index(drop=True)


def balanced_subset(towns: Sequence[str], n: int) -> list[int]:
    """Posiciones de una submuestra de tamaño `n` repartida por turnos entre pueblos."""
    seen: Counter = Counter()
    turn = []
    for pos, town in enumerate(towns):
        turn.append((seen[town], pos))
        seen[town] += 1
    return sorted(pos for _, pos in sorted(turn)[: max(0, n)])


def _single_rows(rankings: Sequence[Sequence[str]], towns: Sequence[str], k: int) -> list[dict]:
    return [{**ranking_metrics(r, [t], k=k), "has_exclusion": False} for r, t in zip(rankings, towns)]


def _per_town(pipe: Pipeline, rows: list[dict], towns: Sequence[str]) -> dict:
    """Hit@5 por pueblo y su Spearman con el número de reseñas."""
    by_town: dict[str, list[float]] = {}
    for row, town in zip(rows, towns):
        by_town.setdefault(town, []).append(row["hit_at_5"])
    names = sorted(by_town)
    pos = {t: i for i, t in enumerate(pipe.index.towns)}
    rho, _ = _spearman([float(np.mean(by_town[t])) for t in names], [float(pipe.index.n_reviews[pos[t]]) for t in names])
    return {
        "per_town_hit_at_5": {t: round(float(np.mean(by_town[t])), 3) for t in names},
        "spearman_hit5_vs_reviews": rho,
    }


def _variant(pipe: Pipeline, rows: list[dict], towns: Sequence[str]) -> dict:
    return {"n": len(rows), "metrics": aggregate(rows), **_per_town(pipe, rows, towns)}


def holdout_suite(
    pipe: Pipeline,
    sample: pd.DataFrame,
    terms: Sequence[str],
    hyde_n: int = 200,
    hyde_unmasked_n: int = 0,
    progress: Callable[[str], None] = lambda _msg: None,
) -> dict:
    """Cada reseña reservada es una consulta; el destino correcto es su pueblo.

    Tres variantes sobre el texto enmascarado: reseña como documento (sin LLM), consulta
    cruda con el prefijo de consulta, y HyDE + Anti-HyDE (pipeline de la web) en una
    submuestra de `hyde_n`. También se evalúa sin enmascarar para medir la fuga de nombres.
    """
    k, enc = pipe.k, pipe.rec.encoder
    towns = sample["town"].tolist()
    row_ids = [int(r) for r in sample["row_id"]] if "row_id" in sample else list(range(len(sample)))
    original = sample["text"].tolist()
    canonical = [canonical_text(t) for t in original]
    masked = [mask_entities(t, terms) for t in original]
    keep = [i for i, t in enumerate(masked) if len(t.split()) >= 2]
    if len(keep) < len(masked):
        towns, row_ids, original, canonical, masked = (
            [seq[i] for i in keep] for seq in (towns, row_ids, original, canonical, masked)
        )

    texts = {"original (sin enmascarar)": original, "canónico sin enmascarar": canonical, "canónico enmascarado": masked}
    vecs = {name: {"doc": enc.encode_documents(t), "raw": enc.encode_queries(t)} for name, t in texts.items()}
    progress(f"Holdout: {len(masked)} reseñas codificadas ({len(texts)} versiones × 2 prefijos)")
    doc, raw = vecs["canónico enmascarado"]["doc"], vecs["canónico enmascarado"]["raw"]
    names = {"doc": "reseña como documento (sin LLM)", "raw": "consulta cruda (search_query)"}
    doc_rankings, raw_rankings = pipe.score_rankings(doc), pipe.score_rankings(raw)
    variants = {
        names["doc"]: _variant(pipe, _single_rows(doc_rankings, towns, k), towns),
        names["raw"]: _variant(pipe, _single_rows(raw_rankings, towns, k), towns),
    }

    # HyDE + Anti-HyDE en una submuestra balanceada (cuida la cuota del LLM).
    subset = balanced_subset(towns, hyde_n)
    runs, sub_towns, sub_pos = [], [], []
    for n_done, pos in enumerate(subset, 1):
        run = pipe.run(masked[pos])
        if run is not None:
            run.raw_vec = raw[pos]
            runs.append(run)
            sub_towns.append(towns[pos])
            sub_pos.append(pos)
        if n_done % 25 == 0:
            progress(f"Holdout HyDE {n_done}/{len(subset)} (nuevas llamadas al LLM: {pipe.new_calls})")
    hyde_name = "HyDE + Anti-HyDE (pipeline web)"
    hyde_rankings = [pipe.ranking(r) for r in runs]
    hyde_rows = _single_rows(hyde_rankings, sub_towns, k)
    # Posición (1 = mejor) del pueblo correcto por consulta; `None` = sin HyDE para esa reseña.
    hyde_rank: list = [None] * len(towns)
    for pos, ranking, town in zip(sub_pos, hyde_rankings, sub_towns):
        hyde_rank[pos] = ranking.index(town) + 1
    per_query = {
        "row_id": row_ids,
        "town": towns,
        "n_towns": len(pipe.index.towns),
        "rank": {
            names["doc"]: [r.index(t) + 1 for r, t in zip(doc_rankings, towns)],
            names["raw"]: [r.index(t) + 1 for r, t in zip(raw_rankings, towns)],
            hyde_name: hyde_rank,
        },
    }
    variants[hyde_name] = {
        **_variant(pipe, hyde_rows, sub_towns),
        "requested": len(subset),
        "pending": len(subset) - len(runs),
        "providers": dict(Counter(llm_tag(r.expansion) for r in runs)),
    }
    same_subset = {
        names[key]: aggregate(_single_rows(pipe.score_rankings(v[sub_pos]), sub_towns, k)) if sub_pos else aggregate([])
        for key, v in (("doc", doc), ("raw", raw))
    }
    same_subset[hyde_name] = aggregate(hyde_rows)

    # Ablación: en la submuestra con HyDE (5 filas) y, sin LLM, en todo el holdout.
    ablation_hyde = ablation_suite(pipe, [(r, [t], []) for r, t in zip(runs, sub_towns)]) if runs else {}
    ablation_full = {}
    for key, v in (("doc", doc), ("raw", raw)):
        configs = {"arquetipos, λ elegido": {}, "arquetipos, λ = 0": {"lam": 0.0}}
        if pipe.centroids is not None:
            configs["1 centroide por pueblo, λ elegido"] = {"centroid": True}
        ablation_full[names[key]] = {
            name: aggregate(_single_rows(pipe.score_rankings(v, **kw), towns, k)) for name, kw in configs.items()
        }

    # Fuga de nombres: mismo texto con y sin enmascarar (y el efecto de canonizar aparte).
    leak = {
        name: {
            names[key]: aggregate(_single_rows(pipe.score_rankings(v[key]), towns, k)) for key in ("doc", "raw")
        }
        for name, v in vecs.items()
    }
    leak_hyde = None
    if hyde_unmasked_n > 0:
        u_runs, u_towns = [], []
        for pos in balanced_subset(towns, hyde_unmasked_n):
            run = pipe.run(original[pos])
            if run is not None:
                u_runs.append(run)
                u_towns.append(towns[pos])
        leak_hyde = {
            "requested": hyde_unmasked_n,
            "n": len(u_runs),
            "metrics": aggregate(_single_rows([pipe.ranking(r) for r in u_runs], u_towns, k)),
        }

    by_volume = [pipe.index.towns[i] for i in np.argsort(-pipe.index.n_reviews, kind="stable")]
    n_towns = len(pipe.index.towns)
    return {
        "description": (
            "Cada reseña reservada de validación (no vista al construir los arquetipos) actúa como "
            "consulta con nombres de pueblo y estado enmascarados; el destino correcto es su pueblo."
        ),
        "held_out_reviews": len(sample),
        "evaluated_queries": len(masked),
        "changed_by_mask": round(float(np.mean([m != c for m, c in zip(masked, canonical)])), 4),
        "variants": variants,
        "same_subset": {"n": len(sub_pos), "metrics": same_subset},
        "random_expectation": {
            "hit_at_1": round(1 / n_towns, 4), "hit_at_5": round(k / n_towns, 4),
            "bottom_error_rate": round(pipe.bottom_k / n_towns, 4),
        },
        "popularity_baseline": aggregate(_single_rows([by_volume] * len(towns), towns, k)),
        "ablation_hyde_subset": ablation_hyde,
        "ablation_no_llm": ablation_full,
        "leak": leak,
        "leak_hyde_unmasked": leak_hyde,
        "per_query": per_query,
    }


# ---------------------------------------------------------------- rúbrica y exportación
def rubric(gold: dict, determinism: dict) -> list[dict]:
    """Los cinco criterios del equipo; la latencia se evalúa en caliente y en frío."""
    agg, lat = gold["aggregate"], gold["latency_s"]
    checks = [
        (f"Precision@5 (gold, {agg['wide_queries']} consultas con ≥5 relevantes)",
         agg["precision_at_5_wide"], RUBRIC["precision_at_5"], ">="),
        ("Hit@1 (gold)", agg["hit_at_1"], RUBRIC["hit_at_1"], ">="),
        ("Contrato 5 + 5", gold["contract_pass_rate"], RUBRIC["contract"], ">="),
        ("Determinismo (con caché de LLM)", determinism["identical_rate"], RUBRIC["determinism"], ">="),
        (f"Latencia p95 en caliente, LLM en caché (s, n = {lat['warm']['n']})",
         lat["warm"]["p95"], RUBRIC["latency_p95_s"], "<"),
        (f"Latencia p95 en frío, sin caché de LLM (s, n = {lat['cold']['n']})",
         lat["cold"]["p95"], RUBRIC["latency_p95_s"], "<"),
    ]
    return [
        {
            "check": name,
            "value": value,
            "threshold": threshold,
            "pass": None if value is None else (value >= threshold if op == ">=" else value < threshold),
        }
        for name, value, threshold, op in checks
    ]


def bank_rankings(pipe: Pipeline, bank: list[dict]) -> dict:
    """Ranking completo de cada consulta del banco, para la evaluación humana ciega del equipo."""
    rankings, rejected, pending = {}, {}, []
    for item in bank:
        try:
            validate_query(item["query"])
        except QueryError as exc:
            rejected[item["id"]] = str(exc)
            continue
        run = pipe.run(item["query"])
        if run is None:
            pending.append(item["id"])
            continue
        ranking = pipe.ranking(run)
        rankings[item["id"]] = {
            "query": item["query"],
            "split": item.get("split"),
            "ranking": ranking,
            "top": ranking[: pipe.k],
            "bottom": ranking[-pipe.bottom_k:][::-1],
            "ranking_por_score": pipe.ranking(run, bottom="reverso"),
            "llm": llm_tag(run.expansion),
        }
    return {"rankings": rankings, "rejected": rejected, "pending": pending}


def plan_queries(
    gold: list[dict], bank: list[dict], holdout_masked: Sequence[str], holdout_original: Sequence[str],
    robustness_n: Optional[int] = None,
) -> dict[str, list[str]]:
    """Consultas que cada sección manda al LLM, en el orden en que se ejecutan."""

    def valid(queries: Iterable[str]) -> list[str]:
        out = []
        for q in queries:
            try:
                out.append(validate_query(q))
            except QueryError:
                pass
        return out

    rob = gold[: robustness_n if robustness_n is not None else len(gold)]
    return {
        "gold": valid(q["query"] for q in gold),
        "adversariales": valid(q for _, q, _ in ADVERSARIAL),
        "banco de consultas": valid(q["query"] for q in bank),
        "holdout HyDE (enmascarado)": list(holdout_masked),
        "holdout HyDE (sin enmascarar)": list(holdout_original),
        "robustez (3 variantes)": valid(v for q in rob for v in query_variants(q["query"])),
    }


def estimate_calls(pipe: Pipeline, plan: dict[str, list[str]]) -> dict:
    """Llamadas nuevas al LLM por sección (prompts que no están en la caché), sin llamar a nadie."""
    seen: set[str] = set()
    sections = {}
    for name, queries in plan.items():
        new = 0
        for q in queries:
            prompt = build_prompt(q)
            if prompt in seen:
                continue
            seen.add(prompt)
            new += not pipe.is_cached(q)
        sections[name] = {"queries": len(queries), "new_calls": int(new)}
    total = sum(s["new_calls"] for s in sections.values())
    tokens = total * TOKENS_PER_CALL
    return {
        "sections": sections,
        "new_calls": total,
        "estimated_tokens": tokens,
        "groq_free_tier": GROQ_FREE,
        "fits_requests_per_day": total <= GROQ_FREE["rpd"],
        "fits_tokens_per_day": tokens <= GROQ_FREE["tpd"],
        "days_by_tokens": math.ceil(tokens / GROQ_FREE["tpd"]) if tokens else 0,
        "minutes_at_tpm": round(tokens / GROQ_FREE["tpm"], 1),
    }


# ---------------------------------------------------------------- markdown
def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sí" if value else "no"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _state(value: Optional[bool]) -> str:
    return "—" if value is None else ("✅" if value else "❌")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in row) + " |" for row in rows]
    return lines


_ABL_COLS = ["precision_at_5", "hit_at_1", "hit_at_5", "mrr", "ndcg_at_5", "bottom_error_rate"]
_ABL_HEAD = ["Configuración", "P@5", "Hit@1", "Hit@5", "MRR", "nDCG@5", "Error en bottom"]


def render_markdown(report: dict) -> str:
    """El reporte en Markdown, con las mismas secciones que el del equipo y las adicionales."""
    meta, gold = report["meta"], report["gold"]
    agg, base = gold["aggregate"], report["baselines"]
    lat = gold["latency_s"]
    L: list[str] = [
        "# Reporte de evaluación automática",
        "",
        f"- **Encoder:** {meta['encoder']} · **k:** {meta['k']} arquetipos por pueblo "
        f"({meta['n_archetypes']} en {meta['n_towns']} pueblos) · **λ:** {meta['lambda']}",
        f"- **Cadena de LLM (evaluación):** {meta['llm_chain']} · temperature {meta['temperature']}",
        f"- **LLM usado en esta corrida:** {meta['llm_used']} "
        f"({meta['llm_new_calls']} llamadas nuevas, {meta['llm_cached']} desde la caché)",
        f"- **Commit:** {meta['commit']} · **Fecha:** {meta['generated_at']} · "
        f"**Duración:** {meta['duration_s']} s · **Config:** `{meta['config']}`",
        "",
        "> Protocolo, métricas y umbrales del equipo (`restmex-orbita`, commit "
        f"{meta['team_commit']}). “5 mejores” = mayor afinidad con la consulta; “5 peores” = los "
        "más parecidos al Anti-HyDE (lo opuesto a lo pedido), no los de peor calidad.",
        "",
        "## 1. Rúbrica",
        "",
    ]
    L += _table(
        ["Criterio", "Valor", "Umbral", "Estado"],
        [(r["check"], r["value"], str(r["threshold"]), _state(r["pass"])) for r in report["rubric"]],
    )
    L += [
        "",
        f"Latencia en proceso de `Recommender.recommend` (LLM + embeddings + búsqueda). En frío: "
        f"p50 {_fmt(lat['cold']['p50'])} s, p95 {_fmt(lat['cold']['p95'])} s, máx. {_fmt(lat['cold']['max'])} s "
        f"(n = {lat['cold']['n']}, consultas que no estaban en la caché del LLM; la pausa entre llamadas "
        f"para respetar la cuota no se cuenta). En caliente: p50 {_fmt(lat['warm']['p50'])} s, "
        f"p95 {_fmt(lat['warm']['p95'])} s, máx. {_fmt(lat['warm']['max'])} s (n = {lat['warm']['n']}). "
        "La web usa primero Gemini; aquí la latencia en frío es la de la cadena de evaluación.",
        "",
        "## 2. Gold (seed heurístico del equipo, pendiente de adjudicación humana)",
        "",
        f"`eval/gold_queries.json` copiado sin cambios (SHA-256 `{meta['gold_sha256'][:16]}…`; autor: "
        f"{meta['gold_provenance'].get('author')}; vio salidas del sistema del equipo: "
        f"{_fmt(meta['gold_provenance'].get('saw_system_output'))}). Pipeline completo de la web "
        f"(HyDE + Anti-HyDE + λ = {meta['lambda']}) sobre {agg['queries']} consultas.",
        "",
    ]
    L += _table(
        ["Métrica", "Sistema", "Popularidad", "Azar"],
        [(key, agg[key], base["popularity"][key], base["random"][key]) for key in METRIC_KEYS],
    )
    L += [
        "",
        f"Top/Bottom de `recommend` idénticos al ranking evaluado: {_fmt(gold['pipeline_consistency'])}. "
        f"Consultas resueltas con la plantilla sin LLM: {gold['fallback_queries']}. Las "
        f"{sum(1 for q in gold['queries'] if q['has_exclusion'])} consultas con exclusión miden "
        "`exclusion_violation_rate`; este sistema no interpreta exclusiones de forma explícita.",
    ]
    if gold.get("by_split"):
        L += ["", "Por partición del banco de consultas del equipo (`test` no se usa para ajustar nada):", ""]
        L += _table(
            ["Partición", "Consultas", "P@5", "Hit@1", "Hit@5", "MRR", "nDCG@5"],
            [(name, a["queries"], a["precision_at_5"], a["hit_at_1"], a["hit_at_5"], a["mrr"], a["ndcg_at_5"])
             for name, a in gold["by_split"].items()],
        )

    h = report.get("holdout")
    L += ["", "## 3. Holdout enmascarado (sin consultas escritas a mano)", ""]
    if h:
        pop, rnd = h["popularity_baseline"], h["random_expectation"]
        L += [
            h["description"],
            "",
            f"{h['evaluated_queries']} reseñas de `val` (polaridad 4–5, 80–600 caracteres, "
            f"{meta['holdout_per_town']} por pueblo, seed {meta['holdout_seed']}). El enmascarado "
            f"modificó el {100 * h['changed_by_mask']:.1f} % de las reseñas. El texto enmascarado queda "
            "en forma canónica (minúsculas, sin acentos ni puntuación), como en el equipo.",
            "",
        ]
        rows = [
            (name, v["n"], v["metrics"]["hit_at_1"], v["metrics"]["hit_at_5"], v["metrics"]["mrr"],
             v["metrics"]["bottom_error_rate"], v["spearman_hit5_vs_reviews"])
            if v["n"] else (name, 0, "—", "—", "—", "—", "—")
            for name, v in h["variants"].items()
        ]
        rows.append(("Popularidad", h["evaluated_queries"], pop["hit_at_1"], pop["hit_at_5"], pop["mrr"],
                     pop["bottom_error_rate"], "—"))
        rows.append(("Azar (esperado)", "—", rnd["hit_at_1"], rnd["hit_at_5"], "—", rnd["bottom_error_rate"], "—"))
        L += _table(["Variante", "n", "Hit@1", "Hit@5", "MRR", "Pueblo correcto en bottom 5",
                     "Spearman(Hit@5 por pueblo, nº reseñas)"], rows)
        hyde = h["variants"]["HyDE + Anti-HyDE (pipeline web)"]
        L += [
            "",
            f"HyDE se evalúa en una submuestra balanceada (`--hyde-holdout {hyde['requested']}`; "
            f"pendientes por cuota: {hyde['pending']}; LLM: "
            f"{', '.join(f'{k} × {v}' for k, v in hyde['providers'].items()) or '—'}). "
            "En las variantes sin Anti-HyDE el bottom es el reverso del ranking. "
            f"Mismas {h['same_subset']['n']} reseñas en las tres variantes:",
            "",
        ]
        L += _table(
            ["Variante (misma submuestra)", "Hit@1", "Hit@5", "MRR"],
            [(name, a["hit_at_1"], a["hit_at_5"], a["mrr"]) for name, a in h["same_subset"]["metrics"].items()],
        )
        L += ["", "### Fuga de nombres: el mismo holdout sin enmascarar", ""]
        leak_rows = []
        for text_name, by_variant in h["leak"].items():
            for variant, a in by_variant.items():
                leak_rows.append((text_name, variant, a["hit_at_1"], a["hit_at_5"], a["mrr"]))
        L += _table(["Texto de la consulta", "Variante", "Hit@1", "Hit@5", "MRR"], leak_rows)
        doc_name = "reseña como documento (sin LLM)"
        leak_gain = (h["leak"]["canónico sin enmascarar"][doc_name]["hit_at_5"]
                     - h["leak"]["canónico enmascarado"][doc_name]["hit_at_5"])
        canon_gain = (h["leak"]["original (sin enmascarar)"][doc_name]["hit_at_5"]
                      - h["leak"]["canónico sin enmascarar"][doc_name]["hit_at_5"])
        L += [
            "",
            f"Con la reseña como documento, la fuga de nombres inflaba Hit@5 en {100 * leak_gain:+.1f} puntos "
            f"(canónico sin enmascarar − enmascarado); pasar el texto a forma canónica cuesta además "
            f"{100 * canon_gain:+.1f} puntos (original − canónico), efecto del formato y no de la fuga.",
        ]
        if h.get("leak_hyde_unmasked"):
            u = h["leak_hyde_unmasked"]
            L += ["", f"HyDE sin enmascarar (n = {u['n']}): Hit@1 {_fmt(u['metrics']['hit_at_1'])}, "
                      f"Hit@5 {_fmt(u['metrics']['hit_at_5'])}, MRR {_fmt(u['metrics']['mrr'])}."]
        else:
            L += ["", "HyDE sin enmascarar no se corrió (`--hyde-unmasked 0`): duplicaría las llamadas al "
                      "LLM y el prompt ya prohíbe nombres propios en el HyDE."]
    else:
        L.append("No se corrió (`--no-holdout`).")

    L += ["", "## 4. Ablación", "", "Cada fila cambia una sola pieza del sistema completo. Es descriptiva: "
          "los hiperparámetros se eligen en validación (`scripts/evaluate.py`), no con estas tablas.",
          "", f"### Gold ({agg['queries']} consultas)", ""]
    L += _table(_ABL_HEAD, [(name, *[a[c] for c in _ABL_COLS]) for name, a in report["ablation"].items()])
    if h:
        L += ["", f"### Holdout, submuestra con HyDE (n = {h['same_subset']['n']})", ""]
        L += _table(_ABL_HEAD, [(name, *[a[c] for c in _ABL_COLS]) for name, a in h["ablation_hyde_subset"].items()])
        L += ["", f"### Holdout completo, sin LLM (n = {h['evaluated_queries']})", ""]
        rows = []
        for variant, configs in h["ablation_no_llm"].items():
            for name, a in configs.items():
                rows.append((variant, name, a["hit_at_1"], a["hit_at_5"], a["mrr"], a["bottom_error_rate"]))
        L += _table(["Consulta", "Configuración", "Hit@1", "Hit@5", "MRR", "Error en bottom"], rows)

    bias = report["volume_bias"]
    L += [
        "",
        "## 5. Sesgo por volumen de reseñas",
        "",
        f"Spearman(apariciones en top 5 del gold, nº de reseñas) = {_fmt(bias['spearman_exposure_vs_reviews'])} "
        f"(p = {_fmt(bias['p_value'])}). Pueblo más expuesto: {bias['max_exposure'][0]} "
        f"({bias['max_exposure'][1]} apariciones). Pueblos que nunca aparecen en el top 5 del gold "
        f"({len(bias['towns_never_in_top5'])}): {', '.join(bias['towns_never_in_top5']) or 'ninguno'}.",
    ]

    rob = report.get("robustness")
    L += ["", "## 6. Robustez", ""]
    if rob and rob["pairs"]:
        L.append(
            f"Jaccard medio del top 5 ante variantes (sin acentos en mayúsculas, relleno, cláusulas "
            f"invertidas): {_fmt(rob['mean_top5_jaccard'])} (mínimo {_fmt(rob['min_top5_jaccard'])}; "
            f"{rob['pairs']} pares; pendientes por cuota: {rob['pending_pairs']}). Cada variante es un "
            "prompt nuevo para el LLM: aquí se mide la sensibilidad del HyDE a cambios superficiales."
        )
    else:
        L.append(f"No se corrió o quedó pendiente por cuota ({(rob or {}).get('pending_pairs', 0)} pares pendientes).")

    adv = report["adversarial"]
    L += ["", "## 7. Casos adversariales", "",
          f"Tasa de aprobación: {_fmt(adv['pass_rate'])}. Este sistema no tiene nivel de confianza, así que "
          "un caso `reject_or_low` solo pasa si la consulta se rechaza. La validación (≥ 2 palabras, no solo "
          "exclusiones, al menos una palabra con contenido) vive en `restmex/validation.py`; la API además "
          "limita la consulta a 600 caracteres, por lo que el caso “muy larga” solo se sirve en proceso.", ""]
    L += _table(
        ["Caso", "Esperado", "Resultado", "OK"],
        [(c["case"], c["expected"], c["outcome"][:90], _state(c["pass"])) for c in adv["cases"]],
    )

    det = report["determinism"]
    L += [
        "",
        "## 8. Determinismo",
        "",
        f"Dos corridas de las {agg['queries']} consultas del gold con la caché del LLM: rankings idénticos "
        f"en {_fmt(det['identical_rate'])} de las consultas. Huella SHA-256 de los rankings: "
        f"`{det['ranking_fingerprint']}`.",
        "",
        f"Sin caché el sistema **no** es determinista: el LLM genera HyDE y Anti-HyDE con "
        f"temperature {det['temperature']}, así que dos llamadas con la misma consulta producen textos "
        "distintos y el ranking puede cambiar. El determinismo que se reporta es el de la caché en disco "
        "(`artifacts/llm_cache`): la primera respuesta válida de cada prompt queda fija, y con ella el "
        "encoder y el ranking son reproducibles. Borrar la caché cambia la huella.",
        "",
        "## 9. Consultas con fallos (Hit@5 = 0 o relevante en el bottom 5)",
        "",
    ]
    failures = [q for q in gold["queries"] if q["hit_at_5"] == 0 or q["relevant_in_bottom"] > 0]
    for q in failures:
        why = []
        if q["hit_at_5"] == 0:
            why.append("Hit@5 = 0")
        if q["relevant_in_bottom"] > 0:
            why.append("relevante en bottom: " + ", ".join(t for t in q["bottom"] if t in q["relevant"]))
        L.append(f"- **{q['id']}** ({'; '.join(why)}) “{q['query']}” → top {', '.join(q['top'])}; "
                 f"bottom {', '.join(q['bottom'])}; esperados {', '.join(q['relevant'])}")
    if not failures:
        L.append("- Ninguna.")

    bank = report.get("bank_export")
    if bank:
        L += [
            "",
            "## Exportación para la evaluación humana ciega",
            "",
            f"`{bank['path']}`: ranking completo de {bank['exported']} de {bank['queries']} consultas de "
            f"`eval/query_bank.json` (rechazadas por la validación: {bank['rejected']}; pendientes por "
            f"cuota: {bank['pending']}).",
        ]
    return "\n".join(L) + "\n"
