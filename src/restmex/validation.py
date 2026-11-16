"""Validación mínima de la consulta antes de gastar una llamada al LLM.

Las reglas y la detección de exclusiones ("sin playa", "no quiero ...") siguen las del
recomendador del equipo (restmex-orbita, `text.py` y `engine._score_query`), para que los
casos adversariales se evalúen con el mismo criterio.
"""
from __future__ import annotations

import re
import unicodedata

from .stopwords import SPANISH_STOPWORDS


class QueryError(ValueError):
    """La consulta no describe un viaje (vacía, de una palabra o solo con exclusiones)."""


# Locuciones con "sin" que no expresan exclusión ("sin duda la mejor playa" pide playa).
NON_EXCLUDING_SIN = (
    "duda", "dudas", "dudarlo", "dudado", "embargo", "palabras", "problema", "problemas",
    "igual", "falta", "pensarlo", "prisa", "pretensiones", "culpa", "medida", "querer",
    "exagerar", "importar", "parar", "fin", "comparacion", "comparación",
)

NEGATIVE_PATTERN = re.compile(
    r"(?:\b(?:sin(?!\s+(?:" + "|".join(NON_EXCLUDING_SIN) + r")\b)|evitar|evito|excepto)\b"
    r"|\bno\s+(?:quiero|deseo|busco|me\s+gusta)\b)"
    # La exclusión termina en puntuación o en "pero": "sin playa, quiero bosque" no niega el bosque.
    r"\s+([^,.;:!?]+?)(?=\s+pero\b|[,.;:!?]|$)",
    flags=re.IGNORECASE,
)


def normalize_text(value: object) -> str:
    """Minúsculas, NFKC y sin puntuación (conserva acentos y guiones)."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("_", " ")
    text = re.sub(r"[^\wáéíóúüñ\s-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def canonical_text(value: object) -> str:
    """Forma sin acentos, para comparar entidades y palabras."""
    text = normalize_text(value)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


_STOPWORDS = frozenset(canonical_text(w) for w in SPANISH_STOPWORDS)


def positive_part(query: str) -> str:
    """La consulta sin sus exclusiones ("bosque, sin playa" → "bosque")."""
    raw = unicodedata.normalize("NFKC", str(query or "")).lower()
    return normalize_text(NEGATIVE_PATTERN.sub(" , ", raw))


def validate_query(query: str) -> str:
    """Devuelve la consulta sin espacios sobrantes o lanza `QueryError`.

    Rechaza: menos de dos palabras, solo exclusiones, y texto sin ninguna palabra con
    contenido (solo números, símbolos o palabras vacías).
    """
    query = " ".join(str(query or "").split())
    if len(normalize_text(query).split()) < 2:
        raise QueryError("Describe al menos dos palabras sobre el viaje que buscas.")
    positive = positive_part(query)
    if not positive:
        raise QueryError("La consulta solo contiene exclusiones; di también qué buscas.")
    content = [
        w for w in canonical_text(positive).split()
        if w not in _STOPWORDS and re.search(r"[^\W\d_]", w)
    ]
    if not content:
        raise QueryError("La consulta no describe ningún rasgo del destino; agrega más detalles.")
    return query
