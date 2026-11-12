"""Proveedores de LLM (Gemini, Groq y Ollama) detrás de la misma interfaz `generate(...)`.

Incluye la cadena de respaldo con control de cuotas (límite de tasa por
proveedor, backoff ante 429/503, cuota diaria agotada) y una caché en disco de
respuestas. Las claves se leen de `.env` y nunca se registran.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import requests

from .config import ROOT, Config
from .errors import LLMError
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_RETRY_IN = re.compile(r"(?:retry|try again) in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", re.I)
_STATUS = re.compile(r"\b([45]\d\d)\b")
_DAILY = re.compile(r"per day|PerDay|\(RPD\)|\(TPD\)|daily", re.I)


class GeminiError(LLMError):
    """Gemini no respondió, no hay clave o devolvió una respuesta vacía."""


class GroqError(LLMError):
    """Groq no respondió, no hay clave o devolvió una respuesta vacía."""


def parse_retry_after(message: str) -> Optional[float]:
    """Extrae "Please retry in 5h31m22.6s" / "try again in 7m12s" (en segundos)."""
    m = _RETRY_IN.search(message)
    if not m or not any(m.groups()):
        return None
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3) or 0)


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


def load_api_key(env_var: str) -> Optional[str]:
    """Lee una clave desde el entorno o desde `.env` en la raíz del proyecto."""
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
        try:
            resp = self._client(timeout).models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**kwargs)
            )
            text = resp.text
        except Exception as exc:  # errores de red, cuota, 5xx, etc.
            full = str(exc)
            err = GeminiError(f"{type(exc).__name__}: {full[:160]}")
            code = _STATUS.search(full[:40])
            err.status = int(code.group(1)) if code else None
            if err.status == 429:
                err.retry_after = parse_retry_after(full)
                err.daily = bool(_DAILY.search(full))
            raise err from exc
        if not text:
            raise GeminiError("Respuesta vacía de Gemini")
        return str(text)


class GroqClient:
    """Cliente de Groq por su API compatible con OpenAI (`/chat/completions`)."""

    def __init__(self, api_key: Optional[str], base_url: str = "https://api.groq.com/openai/v1",
                 reasoning_effort: Optional[str] = "low") -> None:
        if not api_key:
            raise GroqError("Falta la clave de Groq (variable GROQ_API en .env)")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._reasoning_effort = reasoning_effort
        self._session = requests.Session()

    def __repr__(self) -> str:  # nunca exponer la clave
        return "GroqClient(api_key=***)"

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
        """Genera texto; con `json_format` pide `response_format={"type": "json_object"}`."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": num_predict + 1024,   # incluye el razonamiento del modelo
        }
        if json_format:
            payload["response_format"] = {"type": "json_object"}
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        try:
            resp = self._session.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except requests.RequestException as exc:
            raise GroqError(f"{type(exc).__name__}: sin conexión con Groq") from exc
        if resp.status_code != 200:
            try:
                message = str(resp.json().get("error", {}).get("message", ""))
            except ValueError:
                message = resp.text
            err = GroqError(f"{resp.status_code}: {message[:160]}")
            err.status = resp.status_code
            if resp.status_code == 429:
                header = resp.headers.get("retry-after")
                err.retry_after = float(header) if header and header.replace(".", "", 1).isdigit() else parse_retry_after(message)
                err.daily = bool(_DAILY.search(message))
            raise err
        try:
            text = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise GroqError("Respuesta de Groq con formato inesperado") from exc
        if not text:
            raise GroqError("Respuesta vacía de Groq")
        return str(text)

    def list_models(self, timeout: float = 15.0) -> list[str]:
        """Modelos disponibles con esta clave."""
        try:
            resp = self._session.get(f"{self.base_url}/models", timeout=timeout,
                                     headers={"Authorization": f"Bearer {self._api_key}"})
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise GroqError(f"{type(exc).__name__}: no se pudieron listar los modelos") from exc


class LLMCache:
    """Caché en disco de respuestas del LLM: un archivo por hash(proveedor + modelo + prompt)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(provider: str, model: str, system: Optional[str], prompt: str) -> str:
        return hashlib.sha256("\x1f".join([provider, model, system or "", prompt]).encode("utf-8")).hexdigest()

    def get(self, provider: str, model: str, system: Optional[str], prompt: str) -> Optional[str]:
        path = self.directory / f"{self.key(provider, model, system, prompt)}.json"
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))["response"])
        except (OSError, ValueError, KeyError):
            return None

    def put(self, provider: str, model: str, system: Optional[str], prompt: str, response: str) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{self.key(provider, model, system, prompt)}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"provider": provider, "model": model, "response": response},
                                      ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:  # la caché nunca debe romper el flujo
            logger.warning("No se pudo escribir la caché del LLM: %s", exc)


@dataclass
class LLMProvider:
    """Un eslabón de la cadena: cliente + modelo + estado de cuota."""

    name: str                      # "gemini" | "groq" | "ollama"
    model: str
    client: LLMClient
    timeout: float = 90.0
    cooldown: float = 300.0        # s que se omite tras un error (para no frenar la demo)
    rpm: int = 0                   # solicitudes por minuto permitidas (0 = sin límite)
    exhausted: bool = False        # cuota diaria agotada: no se usa en el resto de la sesión
    _down_until: float = field(default=0.0, repr=False)
    _calls: deque = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def tag(self) -> str:
        return f"{self.name}/{self.model}"

    @property
    def available(self) -> bool:
        return not self.exhausted and time.time() >= self._down_until

    def mark_down(self, seconds: Optional[float] = None) -> None:
        """Omite el proveedor durante `seconds` (o su `cooldown` por defecto)."""
        self._down_until = time.time() + (self.cooldown if seconds is None else seconds)

    def mark_up(self) -> None:
        self._down_until = 0.0

    def acquire(self, wait: bool) -> bool:
        """Limitador de tasa (ventana deslizante de 60 s).

        Con `wait=True` (procesos por lotes) duerme hasta que haya cupo; con
        `wait=False` (web) devuelve `False` para que la cadena pase al siguiente.
        """
        if self.rpm <= 0:
            return True
        while True:
            with self._lock:
                now = time.time()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.rpm:
                    self._calls.append(now)
                    return True
                pause = 60.0 - (now - self._calls[0]) + 0.05
            if not wait:
                return False
            time.sleep(pause)


def build_llm_chain(
    cfg: Config, ollama: Optional[OllamaClient] = None, names: Optional[Sequence[str]] = None
) -> list[LLMProvider]:
    """Crea la cadena de proveedores en el orden de `names` (por defecto `llm.chain`).

    Un proveedor sin clave configurada se omite con una advertencia.
    """
    lc = cfg.llm
    chain: list[LLMProvider] = []
    for name in (names if names is not None else lc.chain):
        key = name.lower()
        try:
            if key == "gemini":
                client: LLMClient = GeminiClient(load_api_key(lc.gemini_api_key_env), lc.gemini_thinking_level)
                chain.append(LLMProvider("gemini", lc.gemini_model, client, lc.gemini_timeout, lc.cooldown, lc.gemini_rpm))
            elif key == "groq":
                client = GroqClient(load_api_key(lc.groq_api_key_env), lc.groq_base_url, lc.groq_reasoning_effort)
                chain.append(LLMProvider("groq", lc.groq_model, client, lc.groq_timeout, lc.cooldown, lc.groq_rpm))
            elif key == "ollama":
                client = ollama or OllamaClient(cfg.ollama.host, cfg.ollama.keep_alive)
                chain.append(LLMProvider("ollama", lc.ollama_model, client, lc.timeout, lc.cooldown))
            else:
                raise ValueError(f"Proveedor de LLM desconocido en la cadena: {name}")
        except LLMError as exc:
            logger.warning("Proveedor %s no disponible, se omite: %s", name, exc)
    return chain


def build_cache(cfg: Config) -> Optional[LLMCache]:
    """Caché configurada en `llm.cache` / `llm.cache_dir` (o `None` si está apagada)."""
    return LLMCache(cfg.root / cfg.llm.cache_dir) if cfg.llm.cache else None


@dataclass
class ChainResult:
    """Resultado de recorrer la cadena de proveedores."""

    value: Any
    provider: LLMProvider
    attempts: int                  # llamadas reales al LLM (0 si salió de la caché)
    errors: list[str]              # fallos previos, p. ej. "gemini/gemini-3.5-flash: 503 ..."
    cached: bool = False


class ChainExhausted(LLMError):
    """Ningún proveedor produjo una salida válida."""

    def __init__(self, errors: list[str], attempts: int) -> None:
        super().__init__("; ".join(errors) or "sin proveedores de LLM")
        self.errors = errors
        self.attempts = attempts


def run_chain(
    providers: list[LLMProvider],
    prompt: str,
    parse: Callable[[str], Any],
    *,
    system: Optional[str] = None,
    temperature: float = 0.4,
    num_predict: int = 400,
    retries: int = 1,
    cache: Optional[LLMCache] = None,
    batch: bool = False,
) -> ChainResult:
    """Prueba cada proveedor en orden hasta obtener una salida que `parse` acepte.

    0. Caché: si algún proveedor de la cadena ya respondió este prompt, se
       reutiliza su respuesta sin gastar cuota.
    1. Límite de tasa: en modo `batch` se espera el cupo; en la web se pasa al siguiente.
    2. 429/503 pasajero: backoff y reintento (3 veces en `batch`, 1 breve en la web).
    3. Cuota diaria agotada: el proveedor queda fuera el resto de la sesión.
    4. Otro error: el proveedor se omite durante su `cooldown`.
    5. Salida mal formada (`parse` lanza `ValueError`): `retries` reintentos y luego el siguiente.
    """
    if cache is not None:
        for prov in providers:
            raw = cache.get(prov.name, prov.model, system, prompt)
            if raw is not None:
                try:
                    return ChainResult(parse(raw), prov, 0, [], cached=True)
                except ValueError:
                    continue

    errors: list[str] = []
    attempts = 0
    max_transient, max_wait = (3, 65.0) if batch else (1, 3.0)
    for prov in providers:
        if not prov.available:
            errors.append(f"{prov.tag}: omitido ({'cuota diaria agotada' if prov.exhausted else 'falló hace poco'})")
            continue
        malformed = transient = 0
        while True:
            if not prov.acquire(wait=batch):
                errors.append(f"{prov.tag}: límite de solicitudes por minuto")
                break
            attempts += 1
            try:
                raw = prov.client.generate(
                    prov.model,
                    prompt,
                    system=system,
                    json_format=True,
                    temperature=min(temperature + 0.1 * malformed, 1.0),
                    num_predict=num_predict,
                    timeout=prov.timeout,
                )
                value = parse(raw)
            except LLMError as exc:
                if exc.daily:
                    prov.exhausted = True
                    errors.append(f"{prov.tag}: cuota diaria agotada")
                    logger.warning("%s: cuota diaria agotada; no se usará más en esta sesión", prov.tag)
                    break
                if exc.status in (429, 503) and transient < max_transient:
                    pause = exc.retry_after if exc.retry_after else (2.0 * 2 ** transient if batch else 1.5)
                    if pause <= max_wait:
                        transient += 1
                        logger.info("%s: %s; reintento en %.1f s", prov.tag, exc.status, pause)
                        time.sleep(pause)
                        continue
                errors.append(f"{prov.tag}: no respondió ({exc})")
                logger.warning("%s no respondió: %s", prov.tag, exc)
                prov.mark_down(exc.retry_after + 1 if exc.retry_after else None)
                break
            except ValueError as exc:
                malformed += 1
                errors.append(f"{prov.tag}: salida mal formada ({exc})")
                logger.warning("%s salida mal formada (intento %d): %s", prov.tag, malformed, exc)
                if malformed > max(0, retries):
                    break
                continue
            prov.mark_up()
            if cache is not None:
                cache.put(prov.name, prov.model, system, prompt, raw)
            return ChainResult(value, prov, attempts, errors)
    raise ChainExhausted(errors, attempts)
