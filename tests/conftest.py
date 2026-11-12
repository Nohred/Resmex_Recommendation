"""Fixtures compartidas: índice sintético y Ollama simulado."""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pytest

from restmex.config import Config
from restmex.encoders import Encoder
from restmex.ollama_client import OllamaError
from restmex.retrieval import ArchetypeIndex

GOOD_HYDE = " ".join(["Un pueblo tranquilo con ruinas y comida tradicional deliciosa."] * 8)
GOOD_ANTI = " ".join(["Un destino masivo con antros, ruido y fiesta toda la noche."] * 8)


class FakeOllama:
    """Cliente simulado: devuelve respuestas en orden; una excepción se lanza."""

    def __init__(self, responses: Optional[list[Any]] = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def generate(self, model: str, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        if not self.responses:
            raise OllamaError("sin conexión")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def list_models(self, timeout: float = 3.0) -> list[str]:
        return ["qwen2.5:3b", "nomic-embed-text-v2-moe:latest"]


class FakeEncoder(Encoder):
    """Codificador determinista de 4 dimensiones basado en palabras clave."""

    name = "fake"
    AXES = ["ruinas", "playa", "fiesta", "frío"]

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.full((len(texts), 4), 0.01, dtype=np.float32)
        for i, t in enumerate(texts):
            for j, w in enumerate(self.AXES):
                out[i, j] += t.lower().count(w)
        return out


def good_json() -> str:
    return json.dumps({"hyde": GOOD_HYDE, "anti_hyde": GOOD_ANTI}, ensure_ascii=False)


@pytest.fixture
def index() -> ArchetypeIndex:
    """4 pueblos × 2 arquetipos en 4 dimensiones. `Grande` es el más reseñado."""
    towns = {"Arqueo": 100, "Frio": 200, "Grande": 10000, "Playa": 500}
    e = np.eye(4, dtype=np.float32)
    mix = lambda a, b, w: (w * e[a] + (1 - w) * e[b]) / np.linalg.norm(w * e[a] + (1 - w) * e[b])  # noqa: E731
    vectors = np.vstack([
        e[0], mix(0, 3, 0.7),          # Arqueo
        e[3], mix(3, 0, 0.8),          # Frio
        mix(0, 1, 0.75), mix(2, 1, 0.75),  # Grande: de todo un poco
        e[1], mix(1, 2, 0.6),          # Playa
    ])
    meta = []
    for t in towns:
        for c in range(2):
            meta.append({
                "id": len(meta), "town": t, "town_display": t, "region": "Estado", "cluster": c,
                "label": f"{t}-{c}", "keywords": ["a", "b"], "mean_polarity": 4.5,
                "representatives": [{"text": f"reseña de {t} {c}", "polarity": 5.0, "type": "Attractive"}],
            })
    stats = {"n_max": 10000, "towns": {t: {"n_reviews": n} for t, n in towns.items()}}
    return ArchetypeIndex.from_parts(vectors, meta, stats)


@pytest.fixture
def cfg() -> Config:
    config = Config()
    config.llm.cache = False                     # las pruebas no escriben en artifacts/
    return config
