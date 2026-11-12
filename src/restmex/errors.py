"""Excepciones compartidas."""
from __future__ import annotations


class LLMError(RuntimeError):
    """Un proveedor de LLM (Gemini u Ollama) no respondió o devolvió un error."""
