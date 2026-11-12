"""Proveedores de LLM (Gemini y Ollama) detrás de la misma interfaz `generate(...)`.

Cadena de respaldo: Gemini → modelo local en Ollama → (plantilla, en
`query_expansion`). La clave de Gemini se lee de `.env` y nunca se registra.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .config import ROOT, Config
from .errors import LLMError
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class GeminiError(LLMError):
    """Gemini no respondió, no hay clave o devolvió una respuesta vacía."""


class LLMClient(Protocol):
    """Interfaz común de los clientes de LLM."""

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
    ) -> str: ...


def load_api_key(env_var: str = "GEMINI_API") -> Optional[str]:
    """Lee la clave desde el entorno o desde `.env` en la raíz del proyecto."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # pragma: no cover
        pass
    key = os.environ.get(env_var, "").strip()
    return key or None


class GeminiClient:
    """Cliente de Gemini (SDK oficial `google-genai`) con salida JSON estructurada."""

    def __init__(self, api_key: Optional[str], thinking_level: Optional[str] = "low") -> None:
        if not api_key:
            raise GeminiError("Falta la clave de Gemini (variable GEMINI_API en .env)")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise GeminiError("Falta el paquete google-genai") from exc
        self._genai = genai
        self._types = types
        self._api_key = api_key
        self._thinking_level = thinking_level
        self._clients: dict[int, Any] = {}

    def __repr__(self) -> str:  # nunca exponer la clave
        return "GeminiClient(api_key=***)"

    def _client(self, timeout: float) -> Any:
        ms = int(timeout * 1000)
        if ms not in self._clients:
            self._clients[ms] = self._genai.Client(
                api_key=self._api_key, http_options=self._types.HttpOptions(timeout=ms)
            )
        return self._clients[ms]

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
        """Genera texto; con `json_format` pide `response_mime_type="application/json"`."""
        types = self._types
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            # el razonamiento interno también consume tokens de salida
            "max_output_tokens": num_predict + 1024,
        }
        if system:
            kwargs["system_instruction"] = system
        if json_format:
            kwargs["response_mime_type"] = "application/json"
        if self._thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self._thinking_level)
        text: Optional[str] = None
        for attempt in range(2):
            try:
                resp = self._client(timeout).models.generate_content(
                    model=model, contents=prompt, config=types.GenerateContentConfig(**kwargs)
                )
                text = resp.text
                break
            except Exception as exc:  # errores de red, cuota, 5xx, etc.
                msg = f"{type(exc).__name__}: {str(exc)[:160]}"
                # 503 "high demand" suele ser momentáneo: un reintento rápido.
                if attempt == 0 and "503" in msg:
                    time.sleep(1.5)
                    continue
                raise GeminiError(msg) from exc
        if not text:
            raise GeminiError("Respuesta vacía de Gemini")
        return str(text)

    def list_models(self) -> list[str]:
        """Modelos que aceptan `generateContent` con esta clave."""
        try:
            return [
                m.name.removeprefix("models/")
                for m in self._client(15).models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
            ]
        except Exception as exc:
            raise GeminiError(f"{type(exc).__name__}: {str(exc)[:160]}") from exc


@dataclass
class LLMProvider:
    """Un eslabón de la cadena: cliente + modelo + nombre para mostrar."""

    name: str                      # "gemini" | "ollama"
    model: str
    client: LLMClient
    timeout: float = 90.0
    cooldown: float = 60.0         # segundos que se omite tras fallar (para no frenar la demo)
    _down_until: float = field(default=0.0, repr=False)

    @property
    def available(self) -> bool:
        return time.time() >= self._down_until

    def mark_down(self) -> None:
        self._down_until = time.time() + self.cooldown

    def mark_up(self) -> None:
        self._down_until = 0.0


def build_llm_chain(cfg: Config, ollama: Optional[OllamaClient] = None) -> list[LLMProvider]:
    """Cadena de proveedores según `llm.provider`.

    - `gemini`: modelos de Gemini en orden y, al final, el modelo local de Ollama.
    - `ollama`: solo el modelo local.
    """
    lc = cfg.llm
    chain: list[LLMProvider] = []
    if lc.provider.lower() == "gemini":
        try:
            gemini = GeminiClient(load_api_key(lc.api_key_env), lc.gemini_thinking_level)
            for model in dict.fromkeys([lc.gemini_model, *lc.gemini_fallback_models]):
                chain.append(LLMProvider("gemini", model, gemini, lc.gemini_timeout, lc.cooldown))
        except GeminiError as exc:
            logger.warning("Gemini no disponible, se usará el respaldo: %s", exc)
    elif lc.provider.lower() != "ollama":
        raise ValueError(f"llm.provider desconocido: {lc.provider}")
    ollama = ollama or OllamaClient(cfg.ollama.host, cfg.ollama.keep_alive)
    chain.append(LLMProvider("ollama", lc.ollama_model, ollama, lc.timeout, lc.cooldown))
    return chain


@dataclass
class ChainResult:
    """Resultado de recorrer la cadena de proveedores."""

    value: Any
    provider: LLMProvider
    attempts: int
    errors: list[str]              # fallos previos, p. ej. "gemini/gemini-3.8-flash: 503 ..."


class ChainExhausted(LLMError):
    """Ningún proveedor produjo una salida válida."""

    def __init__(self, errors: list[str], attempts: int) -> None:
        super().__init__("; ".join(errors) or "sin proveedores de LLM")
        self.errors = errors
        self.attempts = attempts


def run_chain(
    providers: list[LLMProvider],
    prompt: str,
    parse: Any,
    *,
    system: Optional[str] = None,
    temperature: float = 0.4,
    num_predict: int = 400,
    retries: int = 1,
) -> ChainResult:
    """Prueba cada proveedor en orden hasta obtener una salida que `parse` acepte.

    - Error de conexión/servicio (`LLMError`): se marca el proveedor como caído
      durante su `cooldown` y se pasa al siguiente.
    - Salida mal formada (`parse` lanza `ValueError`): se reintenta `retries`
      veces con el mismo proveedor y después se pasa al siguiente.
    """
    errors: list[str] = []
    attempts = 0
    for prov in providers:
        tag = f"{prov.name}/{prov.model}"
        if not prov.available:
            errors.append(f"{tag}: omitido (falló hace poco)")
            continue
        for attempt in range(1 + max(0, retries)):
            attempts += 1
            try:
                raw = prov.client.generate(
                    prov.model,
                    prompt,
                    system=system,
                    json_format=True,
                    temperature=min(temperature + 0.1 * attempt, 1.0),
                    num_predict=num_predict,
                    timeout=prov.timeout,
                )
                value = parse(raw)
                prov.mark_up()
                return ChainResult(value, prov, attempts, errors)
            except LLMError as exc:
                errors.append(f"{tag}: no respondió ({exc})")
                logger.warning("%s no respondió: %s", tag, exc)
                prov.mark_down()
                break
            except ValueError as exc:
                errors.append(f"{tag}: salida mal formada ({exc})")
                logger.warning("%s salida mal formada (intento %d): %s", tag, attempt + 1, exc)
    raise ChainExhausted(errors, attempts)
