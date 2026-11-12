"""Excepciones compartidas."""
from __future__ import annotations


from typing import Optional


class LLMError(RuntimeError):
    """Un proveedor de LLM (Gemini u Ollama) no respondió o devolvió un error.

    `retry_after`: segundos que el servicio pide esperar (cuota agotada), si lo indica.
    """

    retry_after: Optional[float] = None
