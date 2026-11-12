"""Expansión de la consulta: HyDE y Anti-HyDE con la cadena de LLM y fallback.

Cadena: Gemini → modelo local en Ollama → plantilla de antónimos (sin LLM).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import LLMConfig
from .llm import ChainExhausted, LLMProvider, run_chain
from .prompts import ANTONYM_DEFAULT, ANTONYM_RULES, ANTONYM_TEMPLATE, HYDE_PROMPT, HYDE_SYSTEM

logger = logging.getLogger(__name__)

MIN_WORDS = 30  # menos que esto se considera una salida mal formada
TEMPLATE_PROVIDER = "plantilla"


@dataclass
class QueryExpansion:
    """Textos generados para una consulta y quién los generó."""

    query: str
    hyde: str
    anti_hyde: str
    provider: str = TEMPLATE_PROVIDER      # "gemini" | "ollama" | "plantilla"
    model: str = ""
    fallback: bool = False                 # True si se usó la plantilla (ningún LLM respondió)
    degraded: bool = False                 # True si no respondió el primer proveedor de la cadena
    fallback_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    attempts: int = 0
    seconds: float = 0.0


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


def expand_query(query: str, providers: list[LLMProvider], cfg: LLMConfig) -> QueryExpansion:
    """Genera HyDE y Anti-HyDE. Nunca lanza: si ningún LLM responde, usa la plantilla.

    Cada proveedor tiene 1 + `cfg.retries` intentos ante una salida mal formada.
    El fallback usa la consulta original como HyDE y una plantilla de antónimos
    como Anti-HyDE.
    """
    query = " ".join(query.split())
    start = time.time()
    prompt = HYDE_PROMPT.format(query=query.replace('"', "'"))
    try:
        res = run_chain(
            providers,
            prompt,
            parse_expansion,
            system=HYDE_SYSTEM,
            temperature=cfg.temperature,
            num_predict=cfg.num_predict,
            retries=cfg.retries,
        )
    except ChainExhausted as exc:
        return QueryExpansion(
            query,
            hyde=query,
            anti_hyde=antonym_anti_hyde(query),
            fallback=True,
            degraded=True,
            fallback_reason=str(exc),
            errors=exc.errors,
            attempts=exc.attempts,
            seconds=time.time() - start,
        )
    hyde, anti = res.value
    degraded = bool(providers) and res.provider is not providers[0]
    return QueryExpansion(
        query,
        hyde,
        anti,
        provider=res.provider.name,
        model=res.provider.model,
        degraded=degraded,
        fallback_reason="; ".join(res.errors) if degraded else None,
        errors=res.errors,
        attempts=res.attempts,
        seconds=time.time() - start,
    )
