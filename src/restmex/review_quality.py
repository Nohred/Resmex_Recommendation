"""Calidad de las reseñas que forman los arquetipos: rasgos, filtros y pesos.

Todo está apagado por defecto (`review_quality` en el YAML). Los filtros y pesos se aplican
solo a las reseñas de `train` que entran al K-Means; validación, holdout y gold no cambian.

Rasgos por reseña (`review_features`):
- `n_content`    palabras de contenido: tokens alfabéticos de 3+ letras que no son stopwords
                 (`SPANISH_STOPWORDS` ya incluye los elogios genéricos: excelente, bonito…).
- `mean_idf`     IDF medio de sus palabras de contenido (0 si no tiene ninguna).
- `cos_global`   coseno de su embedding con el centroide global del conjunto.
- `genericity`   media de los percentiles de (−n_content, −mean_idf, cos_global): 1 = lo más
                 genérico. `informativeness` = 1 − genericity.
- `language`     idioma detectado con lingua (offline); "?" si hay menos de 3 palabras o no
                 hay certeza.
- `near_dup`     tiene una reseña anterior (en el orden del DataFrame) del mismo pueblo con
                 coseno > umbral; quitarlas deja una copia de cada grupo.
- `mentions_place`  nombra su propio pueblo o su estado.
- `service_only`    Hotel o Restaurant que habla del servicio (habitación, mesero,
                    recepción…) y de nada del lugar (calles, playa, centro…) ni lo nombra.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .stopwords import SPANISH_STOPWORDS

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-záéíóúüñ]{3,}")

# Apodos del enmascarado del equipo (report.NICKNAMES) y formas sin espacios comunes.
NICKNAMES: dict[str, list[str]] = {
    "San_Cristobal_de_las_Casas": ["sancris", "san cris"],
    "Tequisquiapan": ["tequis"],
    "Cuatro_Cienegas": ["cuatrocienegas"],
    "Estado_de_Mexico": ["edomex"],
}
# Tokens de los nombres que también son palabras comunes: no se buscan sueltos (sí dentro
# del nombre completo, p. ej. "valle de bravo", "isla mujeres").
GENERIC_NAME_TOKENS = frozenset(
    "las san sal sur isla valle real estado mexico méxico todos casas cuatro luis baja santos mujeres "
    "bravo dolores santa del los ciudad pueblo".split()
)

# Léxicos del diagnóstico "solo servicio" (forma plegada: minúsculas y sin acentos).
SERVICE_TERMS = frozenset(
    """habitacion habitaciones cuarto cuartos cama camas almohada almohadas recepcion recepcionista
    recepcionistas mesero meseros mesera meseras personal staff atencion atentos atento servicio
    servicios regadera bano banos toalla toallas limpieza limpio limpia limpias limpios sucio sucia
    propina cuenta orden ordenamos pedimos platillo platillos desayuno desayunos buffet alberca wifi
    internet estacionamiento aire acondicionado precio precios caro cara caros barato economico
    amable amables trato gerente anfitrion check reservacion reserva suite cocina chef bebidas
    bebida cafe cerveza postre postres menu carta porciones sazon""".split()
)
PLACE_TERMS = frozenset(
    """pueblo pueblito pueblos centro calle calles callejones plaza zocalo iglesia iglesias catedral
    templo convento parroquia playa playas mar laguna lago rio cascada cascadas cenote cenotes
    montana montanas cerro cerros bosque ruinas piramide piramides zona arqueologica mercado
    malecon vista vistas paisaje paisajes ubicacion ubicado ubicada cerca caminando caminar ciudad
    colonial historia historico historica tradicional tradiciones cultura artesanias artesania
    museo parque jardin naturaleza atardecer clima isla bahia desierto pozas grutas cueva cuevas
    mirador arquitectura fachadas magico magicos tour recorrido excursion sendero muelle""".split()
)


# ---------------------------------------------------------------- texto
def fold(text: str) -> str:
    """Minúsculas y sin acentos, carácter por carácter (misma longitud que `text`)."""
    out = []
    for ch in text:
        base = unicodedata.normalize("NFKD", ch)[0] if ch not in "ñÑ" else ch
        out.append(base.lower() if len(base.lower()) == 1 else ch)
    return "".join(out)


def content_words(text: str) -> list[str]:
    """Palabras de contenido: tokens de 3+ letras que no son stopwords."""
    return [w for w in _TOKEN.findall(text.lower()) if w not in SPANISH_STOPWORDS]


def _split_camel(raw: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw.replace("_", " "))


def place_names(town: str, region: str) -> list[str]:
    """Formas plegadas de un pueblo y su estado: nombre completo, sin espacios, apodos y
    tokens distintivos (sin los que también son palabras comunes)."""
    from .data import display_region, display_town

    names: set[str] = set()
    for raw, shown in ((town, display_town(town)), (region, display_region(region))):
        for form in {fold(shown), fold(_split_camel(raw))}:
            form = " ".join(form.split())
            names.add(form)
            names.add(form.replace(" ", ""))
            names.update(t for t in form.split() if len(t) >= 4 and t not in GENERIC_NAME_TOKENS
                         and t not in SPANISH_STOPWORDS)
        names.update(fold(n) for n in NICKNAMES.get(raw, []))
    return sorted((n for n in names if len(n) >= 3), key=len, reverse=True)


def names_pattern(names: Iterable[str]) -> re.Pattern:
    """Regex sobre texto plegado; los espacios del nombre aceptan cualquier separador."""
    alts = sorted(set(names), key=len, reverse=True)
    body = "|".join(r"[\s\-]+".join(re.escape(p) for p in n.split()) for n in alts)
    return re.compile(r"(?<![a-zñ0-9])(?:" + body + r")(?![a-zñ0-9])")


def mask_names(text: str, pattern: re.Pattern) -> str:
    """Borra los nombres de `pattern` del texto original (conserva acentos y mayúsculas)."""
    folded = fold(text)
    if len(folded) != len(text):        # no debería pasar; por seguridad no se toca el texto
        return text
    pieces, last = [], 0
    for m in pattern.finditer(folded):
        pieces.append(text[last:m.start()])
        last = m.end()
    if not pieces:
        return text
    pieces.append(text[last:])
    out = re.sub(r"[(\[]\s*[)\]]", " ", "".join(pieces))                # paréntesis vacíos
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", out)                  # "de,," → "de,"
    return re.sub(r"\s+", " ", out).strip()


def all_places_pattern(places: Iterable[tuple[str, str]]) -> re.Pattern:
    """Patrón con los nombres de todos los pueblos y estados (para enmascarar)."""
    names: set[str] = set()
    for town, region in places:
        names.update(place_names(town, region))
    return names_pattern(names)


def mask_place_names(df: pd.DataFrame, rows: Optional[np.ndarray] = None) -> pd.Series:
    """Textos con los nombres de pueblos y estados borrados (solo en `rows`, booleano)."""
    pattern = all_places_pattern(df[["town", "region"]].drop_duplicates().itertuples(index=False, name=None))
    out = df["text"].copy()
    rows = np.ones(len(df), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    out[rows] = [mask_names(t, pattern) for t in df["text"].to_numpy()[rows]]
    return out


# ---------------------------------------------------------------- rasgos
def _percentile(x: np.ndarray) -> np.ndarray:
    """Percentil (0–1] de cada valor; empates con el rango medio."""
    return pd.Series(x).rank(method="average", pct=True).to_numpy()


def mean_idf(tokens: Sequence[list[str]]) -> np.ndarray:
    """IDF medio por documento con idf = ln((1 + N) / (1 + df)) + 1 sobre el propio conjunto."""
    df_count: dict[str, int] = {}
    for toks in tokens:
        for w in set(toks):
            df_count[w] = df_count.get(w, 0) + 1
    n = len(tokens)
    idf = {w: np.log((1 + n) / (1 + c)) + 1 for w, c in df_count.items()}
    return np.array([np.mean([idf[w] for w in toks]) if toks else 0.0 for toks in tokens])


def detect_languages(texts: Sequence[str], min_words: int = 3) -> list[str]:
    """Código ISO 639-1 por texto (lingua, offline); "?" si es muy corto o no hay certeza.

    Sin portugués ni italiano: en el corpus casi todo lo que lingua marcaba así era español
    sin acentos o con faltas ("energia piramidal", "acseso sencillo").
    """
    from lingua import Language, LanguageDetectorBuilder

    langs = [Language.SPANISH, Language.ENGLISH, Language.FRENCH, Language.GERMAN]
    detector = (LanguageDetectorBuilder.from_languages(*langs)
                .with_minimum_relative_distance(0.1).with_preloaded_language_models().build())
    out = ["?"] * len(texts)
    idx = [i for i, t in enumerate(texts) if len(re.findall(r"[^\W\d_]{2,}", t)) >= min_words]
    found = detector.detect_languages_in_parallel_of([texts[i] for i in idx])
    for i, lang in zip(idx, found):
        out[i] = lang.iso_code_639_1.name.lower() if lang is not None else "?"
    return out


def near_duplicates(emb: np.ndarray, groups: np.ndarray, threshold: float = 0.95,
                    block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(`near_dup`, `has_twin`): reseñas con una anterior del mismo grupo a coseno > umbral,
    y reseñas con cualquier otra del grupo a coseno > umbral. `emb` debe estar en L2."""
    near_dup = np.zeros(len(emb), dtype=bool)
    has_twin = np.zeros(len(emb), dtype=bool)
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        X = np.asarray(emb[idx], dtype=np.float32)
        for start in range(0, len(idx), block):
            S = X[start:start + block] @ X.T
            rows = np.arange(S.shape[0])
            S[rows, start + rows] = -1.0                       # sin la propia reseña
            hit = S > threshold
            has_twin[idx[start:start + block]] = hit.any(axis=1)
            earlier = hit & (np.arange(len(idx))[None, :] < (start + rows)[:, None])
            near_dup[idx[start:start + block]] = earlier.any(axis=1)
    return near_dup, has_twin


def review_features(
    df: pd.DataFrame,
    emb: np.ndarray,
    near_dup_threshold: float = 0.95,
    languages: Optional[Sequence[str]] = None,
    detect_language: bool = True,
) -> pd.DataFrame:
    """Rasgos de calidad de cada reseña de `df` (alineado con `emb`, en L2).

    `languages` permite pasar el idioma ya detectado (la detección es lo más lento); con
    `detect_language=False` y sin `languages` el idioma queda en "?".
    """
    texts = df["text"].tolist()
    tokens = [content_words(t) for t in texts]
    n_content = np.array([len(t) for t in tokens])
    idf = mean_idf(tokens)
    centroid = np.asarray(emb, dtype=np.float64).mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    cos_global = np.asarray(emb, dtype=np.float32) @ centroid.astype(np.float32)
    genericity = (_percentile(-n_content) + _percentile(-idf) + _percentile(cos_global)) / 3
    towns, regions = df["town"].to_numpy(), df["region"].to_numpy()
    near_dup, has_twin = near_duplicates(emb, towns, near_dup_threshold)

    patterns = {(t, r): names_pattern(place_names(t, r)) for t, r in set(zip(towns, regions))}
    folded = [fold(t) for t in texts]
    mentions = np.array([bool(patterns[(t, r)].search(f)) for f, t, r in zip(folded, towns, regions)])
    words = [set(re.findall(r"[a-zñ]+", f)) for f in folded]
    n_service = np.array([len(w & SERVICE_TERMS) for w in words])
    n_place = np.array([len(w & PLACE_TERMS) for w in words])
    is_business = df["type"].isin(["Hotel", "Restaurant"]).to_numpy()
    service_only = is_business & (n_service > 0) & (n_place == 0) & ~mentions

    if languages is not None:
        langs = list(languages)
    else:
        langs = detect_languages(texts) if detect_language else ["?"] * len(texts)
    return pd.DataFrame({
        "row_id": df["row_id"].to_numpy(),
        "n_content": n_content,
        "mean_idf": idf.round(4),
        "cos_global": cos_global.round(4),
        "genericity": genericity.round(4),
        "informativeness": (1 - genericity).round(4),
        "language": langs,
        "near_dup": near_dup,
        "has_twin": has_twin,
        "mentions_place": mentions,
        "n_service": n_service,
        "n_place": n_place,
        "service_only": service_only,
    }, index=df.index)


# ---------------------------------------------------------------- selección y pesos
@dataclass
class Selection:
    keep: np.ndarray            # booleano, alineado con el DataFrame
    weights: np.ndarray         # peso de cada reseña que queda (alineado con `keep`; 0 si sale)
    restored: dict[str, int]    # pueblo → reseñas devueltas para llegar al mínimo
    reasons: pd.Series          # motivo del descarte ("" si se queda)


def is_active(rq: Any) -> bool:
    """¿La configuración `review_quality` descarta o pesa reseñas? (el enmascarado de nombres
    es aparte: cambia el texto antes de codificar)."""
    return bool(
        rq.min_content_words or rq.drop_generic_frac or rq.drop_other_languages or rq.drop_near_duplicates
        or rq.weight_informativeness or rq.weight_polarity or rq.type_weights
    )


def select_reviews(df: pd.DataFrame, feats: pd.DataFrame, rq: Any) -> Selection:
    """Reseñas que entran al K-Means y su peso, según `rq` (ReviewQualityConfig).

    Los descartes se acumulan; si un pueblo queda con menos de `rq.min_reviews_per_town`, se
    devuelven sus descartadas más informativas hasta llegar al mínimo (o a todas las que tenga).
    """
    n = len(df)
    reasons = np.array([""] * n, dtype=object)

    def drop(mask: np.ndarray, why: str) -> None:
        mask = np.asarray(mask, dtype=bool) & (reasons == "")
        reasons[mask] = why

    if rq.min_content_words:
        drop(feats["n_content"].to_numpy() < rq.min_content_words, f"menos de {rq.min_content_words} palabras de contenido")
    if rq.drop_other_languages:
        drop(~feats["language"].isin(["es", "?"]).to_numpy(), "otro idioma")
    if rq.drop_near_duplicates:
        drop(feats["near_dup"].to_numpy(), "casi duplicada")
    if rq.drop_generic_frac:
        # las round(x · n) más genéricas de cada pueblo (n = reseñas del pueblo antes de filtrar)
        town_arr = df["town"].to_numpy()
        rank = feats.groupby(town_arr)["genericity"].rank(method="first", ascending=False).to_numpy()
        size = pd.Series(town_arr).map(pd.Series(town_arr).value_counts()).to_numpy()
        drop(rank <= np.round(rq.drop_generic_frac * size), f"{round(100 * rq.drop_generic_frac)} % más genérico del pueblo")

    keep = reasons == ""
    restored: dict[str, int] = {}
    towns = df["town"].to_numpy()
    info = feats["informativeness"].to_numpy()
    for town in pd.unique(towns):
        idx = np.flatnonzero(towns == town)
        missing = min(rq.min_reviews_per_town, len(idx)) - int(keep[idx].sum())
        if missing > 0:
            out = idx[~keep[idx]]
            back = out[np.argsort(-info[out], kind="stable")[:missing]]
            keep[back] = True
            reasons[back] = ""
            restored[str(town)] = int(len(back))

    w = np.ones(n)
    if rq.weight_informativeness:
        w *= rq.min_weight + (1 - rq.min_weight) * info
    if rq.weight_polarity:
        w *= df["polarity"].to_numpy(dtype=float) / 5.0
    if rq.type_weights:
        w *= df["type"].map(lambda t: float(rq.type_weights.get(t, 1.0))).to_numpy()
    w[~keep] = 0.0
    if restored:
        logger.warning("Pueblos que quedaban con menos de %d reseñas (se devolvieron las más informativas): %s",
                       rq.min_reviews_per_town, restored)
    return Selection(keep=keep, weights=w, restored=restored, reasons=pd.Series(reasons, index=df.index))
