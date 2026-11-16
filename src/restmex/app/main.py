"""Backend FastAPI: `POST /recommend`, `GET /health` y el frontend estático."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import load_config
from ..examples import EXAMPLES
from ..geo import town_coords
from ..llm import LLMError
from ..recommender import Recommender
from ..retrieval import ArchetypeIndex
from ..validation import QueryError, validate_query

logger = logging.getLogger("restmex.app")
APP_DIR = Path(__file__).resolve().parent

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=600, description="Lo que busca el viajero")
    lam: Optional[float] = Field(None, ge=0.0, le=1.0, alias="lambda", description="Penalización λ (opcional)")

    model_config = {"populate_by_name": True}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carga la configuración y el índice de arquetipos una sola vez."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    # RESTMEX_ARTIFACTS permite apuntar a artifacts/sample para pruebas.
    artifacts = Path(os.environ.get("RESTMEX_ARTIFACTS", cfg.artifacts_dir))
    app.state.cfg = cfg
    app.state.recommender = None
    app.state.index_error = None
    try:
        index = ArchetypeIndex.load(artifacts)
        app.state.recommender = Recommender(cfg, index)
        logger.info("Índice cargado: %d arquetipos de %d pueblos", len(index.meta), index.n_towns)
    except (OSError, ValueError, KeyError) as exc:
        app.state.index_error = f"No se pudo cargar el índice en {artifacts.name}/: {exc}"
        logger.error("%s. Ejecuta `make index`.", app.state.index_error)
    yield


app = FastAPI(title="Recomendador de Pueblos Mágicos", version="0.1.0", lifespan=lifespan)


class RevalidatedStatic(StaticFiles):
    """Estáticos con `Cache-Control: no-cache`: el navegador revalida con el ETag y no
    se queda con un CSS o JS viejo después de actualizar la web."""

    async def get_response(self, path: str, scope: Any):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatedStatic(directory=APP_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(APP_DIR / "templates" / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    """Estado de Ollama, de la cadena de LLM y del índice de arquetipos."""
    cfg = app.state.cfg
    rec: Optional[Recommender] = app.state.recommender
    ollama: dict[str, Any] = {"ok": False, "embedding_model": False, "llm_model": False}
    llm_chain: list[dict[str, Any]] = []
    if rec is not None:
        try:
            models = rec.client.list_models()
            has = lambda name: any(m == name or m.split(":")[0] == name for m in models)  # noqa: E731
            ollama = {"ok": True, "embedding_model": has(cfg.encoder.model), "llm_model": has(cfg.llm.ollama_model)}
        except LLMError as exc:
            ollama["error"] = str(exc)[:200]
        llm_chain = [
            {"provider": p.name, "model": p.model, "available": p.available, "exhausted": p.exhausted}
            for p in rec.providers
        ]
    index = (
        {"loaded": True, "archetypes": len(rec.index.meta), "towns": rec.index.n_towns,
         "encoder": rec.index.stats.get("encoder"), "n_train": rec.index.stats.get("n_train")}
        if rec is not None
        else {"loaded": False, "error": app.state.index_error}
    )
    ok = bool(rec is not None and ollama["ok"] and ollama["embedding_model"])
    return {
        "status": "ok" if ok else "degraded",
        "ollama": ollama,
        "llm": {"chain": llm_chain, "cache": cfg.llm.cache},
        "index": index,
        "lambda": cfg.ranking.lam,
        "examples": EXAMPLES,
    }


EMBEDDINGS_DOWN = "El servicio de embeddings (Ollama) no respondió. Verifica que `ollama serve` esté activo."


@app.get("/towns")
def towns() -> dict[str, Any]:
    """Los pueblos del índice con sus coordenadas y sus arquetipos (para el mapa de la web)."""
    rec: Optional[Recommender] = app.state.recommender
    if rec is None:
        raise HTTPException(503, detail=app.state.index_error or "Índice no disponible")
    index = rec.index
    by_town: dict[str, list[dict[str, Any]]] = {t: [] for t in index.towns}
    for meta in index.meta:
        by_town[meta["town"]].append(meta)
    out = []
    for t, town in enumerate(index.towns):
        metas = sorted(by_town[town], key=lambda m: -float(m.get("share", 0.0)))
        first = metas[0] if metas else {}
        coords = town_coords(town)
        out.append({
            "town": town,
            "town_display": first.get("town_display", town),
            "region": first.get("region", ""),
            "n_reviews": int(index.n_reviews[t]),
            "lat": coords[0] if coords else None,
            "lon": coords[1] if coords else None,
            "archetypes": [m.get("label", "") for m in metas if m.get("label")],
        })
    return {"towns": out, "n_max": index.n_max}


def _checked_query(req: RecommendRequest) -> tuple[Recommender, str]:
    """Validaciones comunes de `/recommend` y `/recommend/stream` (503 sin índice, 422 si la consulta no sirve)."""
    rec: Optional[Recommender] = app.state.recommender
    if rec is None:
        raise HTTPException(503, detail=app.state.index_error or "Índice no disponible")
    query = req.query.strip()
    if len(query) < 3:
        raise HTTPException(422, detail="Escribe una descripción un poco más larga.")
    return rec, query


@app.post("/recommend/stream")
def recommend_stream(req: RecommendRequest) -> StreamingResponse:
    """Igual que `/recommend`, pero por etapas (NDJSON, un objeto por línea).

    Eventos: `expansion` (las dos reseñas, en cuanto el LLM responde), después `result`
    (la misma respuesta de `/recommend`) o `error` (`status` y `detail`). Los errores de
    validación y de índice se devuelven antes de abrir el flujo, con su código HTTP.
    """
    rec, query = _checked_query(req)
    try:
        query = validate_query(query)
    except QueryError as exc:
        raise HTTPException(422, detail=str(exc)) from exc

    def events() -> Iterator[str]:
        line = lambda obj: json.dumps(obj, ensure_ascii=False) + "\n"  # noqa: E731
        start = time.perf_counter()
        expansion = rec.expand(query)
        yield line({
            "event": "expansion", "query": expansion.query, "hyde": expansion.hyde,
            "anti_hyde": expansion.anti_hyde, "llm_provider": expansion.provider,
            "llm_model": expansion.model, "fallback": expansion.fallback,
            "degraded": expansion.degraded, "cached": expansion.cached,
        })
        try:
            result = rec.rank(expansion, req.lam, llm_seconds=time.perf_counter() - start)
        except LLMError as exc:  # el encoder (Ollama) no respondió
            logger.error("Fallo de embeddings: %s", exc)
            yield line({"event": "error", "status": 503, "detail": EMBEDDINGS_DOWN})
            return
        yield line({"event": "result", "data": result.to_dict()})

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    """Consulta → HyDE / Anti-HyDE → Top 5 y Bottom 5 con explicaciones y tiempos."""
    rec, query = _checked_query(req)
    try:
        return rec.recommend(query, req.lam).to_dict()
    except QueryError as exc:  # vacía, una sola palabra o solo exclusiones
        raise HTTPException(422, detail=str(exc)) from exc
    except LLMError as exc:  # el encoder (Ollama) no respondió
        logger.error("Fallo de embeddings: %s", exc)
        raise HTTPException(503, detail=EMBEDDINGS_DOWN) from exc
