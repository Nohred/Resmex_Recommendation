"""Perfilado de destinos: K-Means por pueblo → arquetipos (centroides L2)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from .data import display_region, display_town
from .encoders import l2_normalize
from .llm import ChainExhausted, LLMProvider, run_chain
from .prompts import LABEL_PROMPT, LABEL_SYSTEM
from .stopwords import SPANISH_STOPWORDS

logger = logging.getLogger(__name__)

_TOKEN = r"(?u)\b[a-záéíóúüñ]{3,}\b"


def cluster_keywords(texts: list[str], labels: np.ndarray, k: int, n_keywords: int) -> list[list[str]]:
    """Palabras clave por clúster: TF-IDF medio del clúster menos el del resto del pueblo."""
    try:
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=_TOKEN,
            stop_words=list(SPANISH_STOPWORDS),
            ngram_range=(1, 2),
            min_df=min(3, max(1, len(texts) // 50)),
            max_df=0.9,
            sublinear_tf=True,
            max_features=20000,
        )
        X = vec.fit_transform(texts)
    except ValueError:  # vocabulario vacío
        return [[] for _ in range(k)]
    terms = np.asarray(vec.get_feature_names_out())
    total = np.asarray(X.sum(axis=0)).ravel()
    n = X.shape[0]
    out: list[list[str]] = []
    for c in range(k):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            out.append([])
            continue
        in_sum = np.asarray(X[mask].sum(axis=0)).ravel()
        rest_mean = (total - in_sum) / max(n - n_c, 1)
        score = in_sum / n_c - rest_mean
        chosen: list[str] = []
        for j in np.argsort(-score):
            if score[j] <= 0 or len(chosen) >= n_keywords:
                break
            term = str(terms[j])
            # evita redundancia entre unigramas y bigramas ("zona", "zona arqueológica")
            if any(term in c_ or c_ in term for c_ in chosen):
                continue
            chosen.append(term)
        out.append(chosen)
    return out


def fallback_label(keywords: list[str]) -> str:
    """Etiqueta de respaldo: las 3 palabras clave principales."""
    if not keywords:
        return "Experiencia general"
    return ", ".join(keywords[:3]).capitalize()


def _valid_label(label: Any) -> Optional[str]:
    if not isinstance(label, str):
        return None
    label = re.sub(r"\s+", " ", label).strip(" .\"'")
    if not label or len(label) > 40 or len(label.split()) > 5:
        return None
    return label[0].upper() + label[1:]


def _labels_parser(k: int) -> Callable[[str], list[str]]:
    """Valida `{"etiquetas": [k cadenas]}`; lanza `ValueError` si no cumple."""

    def parse(raw: str) -> list[str]:
        try:
            labels = json.loads(raw).get("etiquetas")
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ValueError(f"JSON inválido: {exc}") from exc
        if not isinstance(labels, list) or len(labels) != k:
            raise ValueError(f"Se esperaban {k} etiquetas")
        clean = [_valid_label(x) for x in labels]
        if any(c is None for c in clean):
            raise ValueError("Etiqueta vacía o demasiado larga")
        return clean  # type: ignore[return-value]

    return parse


def label_archetypes(meta: list[dict[str, Any]], providers: list[LLMProvider], cfg: Config) -> int:
    """Etiqueta los arquetipos (in situ) con una llamada al LLM por pueblo.

    Recorre la cadena de proveedores; si ninguno da una salida válida, el pueblo
    conserva la etiqueta de respaldo (3 palabras clave). Devuelve cuántos
    arquetipos quedaron etiquetados por un LLM.
    """
    by_town: dict[str, list[dict[str, Any]]] = {}
    for m in meta:
        by_town.setdefault(m["town"], []).append(m)
    labeled = 0
    for i, (town, items) in enumerate(by_town.items()):
        groups = []
        for j, m in enumerate(items, start=1):
            mix = ", ".join(f"{t} {round(100 * v)}%" for t, v in m.get("type_mix", {}).items())
            review = m["representatives"][0]["text"][:260].replace('"', "'") if m.get("representatives") else ""
            groups.append(
                f"Grupo {j} | tipo: {mix} | palabras clave: {', '.join(m['keywords'])} | reseña: \"{review}\""
            )
        prompt = LABEL_PROMPT.format(k=len(items), town=items[0].get("town_display", town), groups="\n".join(groups))
        try:
            res = run_chain(
                providers, prompt, _labels_parser(len(items)),
                system=LABEL_SYSTEM, temperature=0.2, num_predict=300, retries=1,
            )
        except ChainExhausted as exc:
            logger.warning("Sin etiquetas del LLM para %s: %s", town, exc)
            for m in items:
                m["label"], m["label_source"] = fallback_label(m["keywords"]), "keywords"
            continue
        for m, label in zip(items, res.value):
            m["label"], m["label_source"] = label, f"{res.provider.name}:{res.provider.model}"
        labeled += len(items)
        logger.info("[%2d/%d] etiquetas de %s (%s)", i + 1, len(by_town), town, res.provider.model)
    return labeled


def build_archetypes(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Construye los arquetipos con las reseñas de entrenamiento.

    `df` y `embeddings` deben estar alineados (solo filas de train).
    Las etiquetas quedan con el respaldo de palabras clave; `label_archetypes`
    las reemplaza con el LLM. Devuelve (centroides (A, d) normalizados, metadatos por arquetipo, estadísticas por pueblo).
    """
    acfg = cfg.archetypes
    vectors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    town_stats: dict[str, Any] = {}
    df = df.reset_index(drop=True)
    towns = sorted(df["town"].unique())
    n_max = int(df["town"].value_counts().max())

    for t_i, town in enumerate(towns):
        idx = np.flatnonzero(df["town"].to_numpy() == town)
        X = embeddings[idx]
        k = min(acfg.k, len(idx))
        km = KMeans(n_clusters=k, n_init=acfg.n_init, random_state=acfg.seed).fit(X)
        labels = km.labels_
        centroids = l2_normalize(km.cluster_centers_)
        sub = df.iloc[idx]
        texts = sub["text"].tolist()
        keywords = cluster_keywords(texts, labels, k, acfg.n_keywords)
        polarity = sub["polarity"].to_numpy()
        types = sub["type"].to_numpy()

        for c in range(k):
            members = np.flatnonzero(labels == c)
            if len(members) == 0:
                continue
            sims = X[members] @ centroids[c]
            top = members[np.argsort(-sims)[: acfg.n_representative]]
            reps = [
                {"text": texts[m], "polarity": float(polarity[m]), "type": str(types[m])}
                for m in top
            ]
            type_mix = pd.Series(types[members]).value_counts(normalize=True).round(3).to_dict()
            meta.append(
                {
                    "id": len(meta),
                    "town": town,
                    "town_display": display_town(town),
                    "region": display_region(str(sub["region"].iloc[0])),
                    "cluster": c,
                    "size": int(len(members)),
                    "share": round(len(members) / len(idx), 4),
                    "mean_polarity": round(float(polarity[members].mean()), 3),
                    "type_mix": type_mix,
                    "keywords": keywords[c],
                    "label": fallback_label(keywords[c]),
                    "label_source": "keywords",
                    "representatives": reps,
                }
            )
            vectors.append(centroids[c])

        town_stats[town] = {
            "town_display": display_town(town),
            "region": display_region(str(sub["region"].iloc[0])),
            "n_reviews": int(len(idx)),
            "mean_polarity": round(float(polarity.mean()), 3),
            "n_archetypes": int(k),
            "centroid_id": t_i,
        }
        logger.info("[%2d/%d] %-28s n=%6d k=%d", t_i + 1, len(towns), town, len(idx), k)

    stats = {"n_max": n_max, "towns": town_stats}
    return np.vstack(vectors).astype(np.float32), meta, stats


def town_centroids(df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    """Línea base: un solo centroide (L2) por pueblo, en orden alfabético."""
    towns = sorted(df["town"].unique())
    arr = df["town"].to_numpy()
    return l2_normalize(np.vstack([embeddings[arr == t].mean(axis=0) for t in towns]))
