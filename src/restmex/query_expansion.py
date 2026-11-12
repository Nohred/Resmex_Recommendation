"""Expansión de la consulta: HyDE y Anti-HyDE con el LLM local, con fallback."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from .config import LLMConfig
from .ollama_client import OllamaClient, OllamaError
from .prompts import ANTONYM_DEFAULT, ANTONYM_RULES, ANTONYM_TEMPLATE, HYDE_PROMPT, HYDE_SYSTEM

logger = logging.getLogger(__name__)

MIN_WORDS = 30  # menos que esto se considera una salida mal formada


@dataclass
class QueryExpansion:
    """Textos generados para una consulta."""

    query: str
    hyde: str
    anti_hyde: str
    fallback: bool = False
    fallback_reason: Optional[str] = None
    attempts: int = 0
    seconds: float = 0.0
    model: str = ""


def antonym_anti_hyde(query: str) -> str:
    """Anti-HyDE sin LLM: plantilla con los opuestos de lo que menciona la consulta."""
    opposites = [phrase for pattern, phrase in ANTONYM_RULES if pattern.search(query)]
    return ANTONYM_TEMPLATE.format(opuestos="; ".join(opposites) if opposites else ANTONYM_DEFAULT)


def parse_expansion(raw: str) -> tuple[str, str]:
    """Valida el JSON del LLM. Lanza `ValueError` si viene mal formado."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("La salida no es un objeto JSON")
    hyde, anti = data.get("hyde"), data.get("anti_hyde")
    if not isinstance(hyde, str) or not isinstance(anti, str):
        raise ValueError("Faltan las claves 'hyde' / 'anti_hyde'")
    hyde, anti = " ".join(hyde.split()), " ".join(anti.split())
    if len(hyde.split()) < MIN_WORDS or len(anti.split()) < MIN_WORDS:
        raise ValueError("Textos demasiado cortos")
    if hyde == anti:
        raise ValueError("HyDE y Anti-HyDE son idénticos")
    return hyde, anti


def expand_query(query: str, client: OllamaClient, cfg: LLMConfig) -> QueryExpansion:
    """Genera HyDE y Anti-HyDE. Nunca lanza: si el LLM falla, usa el fallback.

    Hace 1 + `cfg.retries` intentos. El fallback usa la consulta original como
    HyDE y una plantilla de antónimos como Anti-HyDE.
    """
    query = " ".join(query.split())
    start = time.time()
    reason = "sin intentos"
    attempts = 0
    prompt = HYDE_PROMPT.format(query=query.replace('"', "'"))
    for attempt in range(1 + max(0, cfg.retries)):
        attempts = attempt + 1
        try:
            raw = client.generate(
                cfg.model,
                prompt,
                system=HYDE_SYSTEM,
                json_format=True,
                temperature=min(cfg.temperature + 0.1 * attempt, 1.0),
                num_predict=cfg.num_predict,
                timeout=cfg.timeout,
            )
            hyde, anti = parse_expansion(raw)
            return QueryExpansion(
                query, hyde, anti, attempts=attempts, seconds=time.time() - start, model=cfg.model
            )
        except OllamaError as exc:
            reason = f"Ollama no respondió: {exc}"
            logger.warning("HyDE intento %d: %s", attempts, reason)
        except ValueError as exc:
            reason = f"Salida mal formada: {exc}"
            logger.warning("HyDE intento %d: %s", attempts, reason)
    return QueryExpansion(
        query,
        hyde=query,
        anti_hyde=antonym_anti_hyde(query),
        fallback=True,
        fallback_reason=reason,
        attempts=attempts,
        seconds=time.time() - start,
        model=cfg.model,
    )
