"""Cliente HTTP mínimo para Ollama (embeddings y generación)."""
from __future__ import annotations

import logging
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Ollama no respondió o devolvió un error."""


class OllamaClient:
    """Envoltura delgada sobre la API REST de Ollama."""

    def __init__(self, host: str = "http://localhost:11434", keep_alive: str = "30m") -> None:
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self._session = requests.Session()

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            resp = self._session.post(f"{self.host}{path}", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Fallo en {path}: {exc}") from exc

    def embed(self, model: str, texts: list[str], timeout: float = 300.0) -> list[list[float]]:
        """`POST /api/embed` con una lista de textos."""
        data = self._post(
            "/api/embed",
            {"model": model, "input": texts, "truncate": True, "keep_alive": self.keep_alive},
            timeout,
        )
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Respuesta de /api/embed con un número inesperado de vectores")
        return embeddings

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        json_format: bool = False,
        temperature: float = 0.4,
        num_predict: int = 400,
        timeout: float = 90.0,
    ) -> str:
        """`POST /api/generate` sin streaming; devuelve el texto generado."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if system:
            payload["system"] = system
        if json_format:
            payload["format"] = "json"
        return str(self._post("/api/generate", payload, timeout).get("response", ""))

    def list_models(self, timeout: float = 3.0) -> list[str]:
        """Modelos disponibles (`GET /api/tags`)."""
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=timeout)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"Ollama no disponible: {exc}") from exc
