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
        num_ctx: Optional[int] = None,
    ) -> None:
        self.client = client
        self.model = model
        # El contexto cambia dónde se truncan los textos largos: forma parte de la identidad
        # del encoder (y de la huella de la caché) solo cuando se fija de forma explícita.
        self.name = f"ollama:{model}" + (f"@ctx{num_ctx}" if num_ctx else "")
        self.num_ctx = num_ctx
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
                extra = {"options": {"num_ctx": self.num_ctx}} if self.num_ctx else {}
                arr = np.asarray(self.client.embed(self.model, texts, self.timeout, **extra), dtype=np.float32)
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
    """Backend alternativo con `sentence-transformers`.

    Los prefijos se anteponen igual que en Ollama (p. ej. `passage: ` / `query: `
    en la familia E5); para modelos sin prefijos se dejan vacíos en el YAML.
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 64,
        doc_prefix: str = "",
        query_prefix: str = "",
        max_seq_length: Optional[int] = None,
        fp16: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise ImportError(
                "Instala el backend opcional: pip install -r requirements-optional.txt"
            ) from exc
        self.name = f"st:{model}" + (f"@len{max_seq_length}" if max_seq_length else "") + ("@fp16" if fp16 else "")
        self.batch_size = batch_size
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(model)
        if max_seq_length:
            self._model.max_seq_length = max_seq_length
        if fp16 and str(self._model.device).startswith("cuda"):
            self._model.half()

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True),
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
            num_ctx=cfg.encoder.num_ctx,
        )
    if backend in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEncoder(
            cfg.encoder.st_model,
            cfg.encoder.batch_size,
            cfg.encoder.doc_prefix,
            cfg.encoder.query_prefix,
            cfg.encoder.max_seq_length,
            cfg.encoder.fp16,
        )
    raise ValueError(f"Backend de encoder desconocido: {cfg.encoder.backend}")


def corpus_fingerprint(encoder_name: str, doc_prefix: str, texts: Sequence[str]) -> str:
    """Huella del corpus + encoder, para validar que la caché corresponde."""
    h = hashlib.sha1()
    h.update(f"{encoder_name}|{doc_prefix}|{len(texts)}".encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


EMBEDDINGS_REF = "embeddings_ref.json"


def _cached_fingerprint(cache_dir: Path, name: str = "embeddings") -> Optional[str]:
    """Huella de la caché guardada en `cache_dir` (o `None` si no hay)."""
    state_path = cache_dir / f"{name}_state.json"
    if not (state_path.exists() and (cache_dir / f"{name}.npy").exists()):
        return None
    try:
        return json.loads(state_path.read_text()).get("fingerprint")
    except (OSError, ValueError):
        return None


def embeddings_cache_dir(encoder: Encoder, texts: Sequence[str], out_dir: Path, shared_dir: Path) -> Path:
    """Directorio donde vive la caché de embeddings de este corpus.

    La caché depende solo del encoder y del preprocesamiento (la huella cubre
    nombre del encoder, prefijo y los textos ya limpios); k, λ y el resto del
    experimento no intervienen, así que varios experimentos la comparten.

    - `out_dir` es la raíz compartida (configuración base): se usa tal cual,
      con la disposición de siempre (`artifacts/embeddings.npy`).
    - La caché base de `shared_dir` corresponde a este corpus: se reutiliza.
    - En otro caso: `shared_dir/embeddings_cache/<huella>/`, sin tocar la base.
    """
    if out_dir.resolve() == shared_dir.resolve():
        return out_dir
    fp = corpus_fingerprint(encoder.name, encoder.doc_prefix, texts)
    if _cached_fingerprint(shared_dir) == fp:
        return shared_dir
    return shared_dir / "embeddings_cache" / fp[:16]


def write_embeddings_ref(out_dir: Path, cache_dir: Path, root: Path) -> None:
    """Deja en `out_dir` un puntero a la caché compartida (si no vive ahí mismo)."""
    if cache_dir.resolve() == out_dir.resolve():
        return
    try:
        rel = str(cache_dir.resolve().relative_to(root.resolve()))
    except ValueError:
        rel = str(cache_dir.resolve())
    (out_dir / EMBEDDINGS_REF).write_text(json.dumps({"dir": rel}), encoding="utf-8")


def locate_embeddings(artifacts_dir: Path, root: Path) -> Path:
    """Ruta de `embeddings.npy` para un índice: el puntero si existe, o el propio directorio."""
    ref = artifacts_dir / EMBEDDINGS_REF
    if ref.exists():
        target = Path(json.loads(ref.read_text(encoding="utf-8"))["dir"])
        return (target if target.is_absolute() else root / target) / "embeddings.npy"
    return artifacts_dir / "embeddings.npy"


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
