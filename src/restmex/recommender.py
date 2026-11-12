"""Orquestación del pipeline en línea: consulta → HyDE/Anti-HyDE → embeddings → ranking."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Config
from .encoders import Encoder, build_encoder
from .llm import LLMCache, LLMProvider, build_cache, build_llm_chain
from .ollama_client import OllamaClient
from .query_expansion import QueryExpansion, expand_query
from .ranking import TownResult, rank_towns
from .retrieval import ArchetypeIndex

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Respuesta completa a una consulta."""

    expansion: QueryExpansion
    top: list[TownResult]
    bottom: list[TownResult]
    timings_ms: dict[str, float] = field(default_factory=dict)
    lam: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.expansion.query,
            "hyde": self.expansion.hyde,
            "anti_hyde": self.expansion.anti_hyde,
            "fallback": self.expansion.fallback,
            "fallback_reason": self.expansion.fallback_reason,
            "degraded": self.expansion.degraded,
            "llm_provider": self.expansion.provider,
            "llm_model": self.expansion.model,
            "llm_errors": self.expansion.errors,
            "llm_attempts": self.expansion.attempts,
            "cached": self.expansion.cached,
            "lambda": self.lam,
            "top": [t.to_dict() for t in self.top],
            "bottom": [t.to_dict() for t in self.bottom],
            "timings_ms": self.timings_ms,
        }


class Recommender:
    """Mantiene el índice en memoria y responde consultas."""

    def __init__(
        self,
        cfg: Config,
        index: ArchetypeIndex,
        encoder: Optional[Encoder] = None,
        client: Optional[OllamaClient] = None,
        providers: Optional[list[LLMProvider]] = None,
        cache: Optional[LLMCache] = None,
    ) -> None:
        self.cfg = cfg
        self.index = index
        self.client = client or OllamaClient(cfg.ollama.host, cfg.ollama.keep_alive)
        self.encoder = encoder or build_encoder(cfg, self.client)
        self.providers = providers if providers is not None else build_llm_chain(cfg, self.client)
        self.cache = cache if cache is not None else build_cache(cfg)

    def recommend(self, query: str, lam: Optional[float] = None) -> Recommendation:
        """Ejecuta el pipeline completo y mide el tiempo de cada etapa."""
        lam = self.cfg.ranking.lam if lam is None else lam
        t0 = time.perf_counter()
        expansion = expand_query(query, self.providers, self.cfg.llm, self.cache)
        t1 = time.perf_counter()
        # Ambos textos se codifican como documentos (ver encoders.py).
        vecs = self.encoder.encode_documents([expansion.hyde, expansion.anti_hyde])
        t2 = time.perf_counter()
        top, bottom = rank_towns(
            self.index, vecs[0], vecs[1], lam, self.cfg.ranking.top_k, self.cfg.ranking.bottom_k
        )
        t3 = time.perf_counter()
        timings = {
            "llm": round((t1 - t0) * 1000, 1),
            "embedding": round((t2 - t1) * 1000, 1),
            "busqueda": round((t3 - t2) * 1000, 3),
            "total": round((t3 - t0) * 1000, 1),
        }
        return Recommendation(expansion, top, bottom, timings, lam)
