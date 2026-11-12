"""Codificadores de texto detrás de una interfaz común y caché reanudable.

Decisión sobre prefijos (nomic-embed-text-v2-moe): las reseñas y los textos
HyDE / Anti-HyDE son todos "documentos" (comparamos reseña hipotética contra
reseñas reales), por lo que se codifican con `search_document: `. El prefijo
`search_query: ` solo se usa para la consulta cruda del usuario (línea base
sin HyDE en la evaluación).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from .config import Config
from .ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normaliza filas a norma L2 = 1."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


class Encoder(ABC):
    """Interfaz de codificador: textos → vectores normalizados (L2)."""

    name: str = "encoder"

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray:
        """Codifica textos ya prefijados; devuelve (n, d) sin normalizar."""

    doc_prefix: str = ""
    query_prefix: str = ""

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Reseñas, HyDE y Anti-HyDE (modo documento)."""
        return l2_normalize(self._encode([self.doc_prefix + t for t in texts]))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Consulta cruda del usuario (modo consulta)."""
        return l2_normalize(self._encode([self.query_prefix + t for t in texts]))


class OllamaEncoder(Encoder):
    """Embeddings vía `POST /api/embed` de Ollama (p. ej. nomic-embed-text-v2-moe)."""

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        doc_prefix: str = "search_document: ",
        query_prefix: str = "search_query: ",
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.client = client
        self.model = model
        self.name = f"ollama:{model}"
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.timeout = timeout
        self.max_retries = max_retries

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                arr = np.asarray(self.client.embed(self.model, texts, self.timeout), dtype=np.float32)
                if not np.isfinite(arr).all():
                    raise OllamaError("El modelo devolvió valores no finitos")
                return arr
            except OllamaError as exc:
                last = exc
                logger.warning("Lote de %d textos falló (intento %d): %s", len(texts), attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        if len(texts) > 1:  # aísla el texto problemático partiendo el lote
            mid = len(texts) // 2
            return np.vstack([self._encode(texts[:mid]), self._encode(texts[mid:])])
        raise OllamaError(f"No se pudo codificar el texto: {last}")


class SentenceTransformerEncoder(Encoder):
    """Backend alternativo con `sentence-transformers` (sin prefijos de tarea)."""

    def __init__(self, model: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise ImportError(
                "Instala el backend opcional: pip install -r requirements-optional.txt"
            ) from exc
        self.name = f"st:{model}"
        self.batch_size = batch_size
        self._model = SentenceTransformer(model)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False),
            dtype=np.float32,
        )


def build_encoder(cfg: Config, client: Optional[OllamaClient] = None) -> Encoder:
    """Crea el codificador configurado en `config.yaml`."""
    backend = cfg.encoder.backend.lower()
    if backend == "ollama":
        client = client or OllamaClient(cfg.ollama.host, cfg.ollama.keep_alive)
        return OllamaEncoder(
            client,
            cfg.encoder.model,
            cfg.encoder.doc_prefix,
            cfg.encoder.query_prefix,
            cfg.ollama.embed_timeout,
        )
    if backend in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEncoder(cfg.encoder.st_model, cfg.encoder.batch_size)
    raise ValueError(f"Backend de encoder desconocido: {cfg.encoder.backend}")


def corpus_fingerprint(encoder_name: str, doc_prefix: str, texts: Sequence[str]) -> str:
    """Huella del corpus + encoder, para validar que la caché corresponde."""
    h = hashlib.sha1()
    h.update(f"{encoder_name}|{doc_prefix}|{len(texts)}".encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def embed_corpus(
    encoder: Encoder,
    texts: Sequence[str],
    cache_dir: Path,
    batch_size: int = 128,
    save_every: int = 20,
    name: str = "embeddings",
) -> np.ndarray:
    """Codifica el corpus con caché en disco, reanudable por bloques.

    Escribe `<name>.npy` (memmap) y `<name>_state.json` con el avance. Si se
    interrumpe, la siguiente llamada continúa donde se quedó; si el corpus o
    el encoder cambiaron (huella distinta), se recalcula desde cero.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{name}.npy"
    state_path = cache_dir / f"{name}_state.json"
    n = len(texts)
    fp = corpus_fingerprint(encoder.name, encoder.doc_prefix, texts)

    done = 0
    mm: Optional[np.memmap] = None
    if npy_path.exists() and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") == fp and state.get("n") == n:
            done = int(state["done"])
            mm = np.load(npy_path, mmap_mode="r+")
            if done >= n:
                logger.info("Embeddings en caché: %s (%d × %d)", npy_path.name, *mm.shape)
                return np.asarray(mm)
            logger.info("Reanudando embeddings en %d / %d", done, n)
        else:
            logger.info("La caché no corresponde al corpus/encoder actual: se recalcula")

    def save_state(d: int, dim: int) -> None:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": fp, "n": n, "done": d, "dim": dim, "encoder": encoder.name}))
        tmp.replace(state_path)

    start = time.time()
    start_done = done
    batches = 0
    for i in range(done, n, batch_size):
        vecs = encoder.encode_documents(list(texts[i : i + batch_size]))
        if mm is None:
            mm = open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(n, vecs.shape[1]))
        mm[i : i + len(vecs)] = vecs
        done = i + len(vecs)
        batches += 1
        if batches % save_every == 0 or done >= n:
            mm.flush()
            save_state(done, mm.shape[1])
            elapsed = time.time() - start
            rate = (done - start_done) / max(elapsed, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            logger.info(
                "Embeddings %d/%d (%.1f%%) · %.0f textos/s · restante ≈ %.1f min",
                done, n, 100 * done / n, rate, eta / 60,
            )
    assert mm is not None
    return np.asarray(mm)
