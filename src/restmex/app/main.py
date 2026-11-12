"""Backend FastAPI: `POST /recommend`, `GET /health` y el frontend estático."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import load_config
from ..llm import LLMError
from ..recommender import Recommender
from ..retrieval import ArchetypeIndex

logger = logging.getLogger("restmex.app")
APP_DIR = Path(__file__).resolve().parent

EXAMPLES = [
    "Quiero un lugar tranquilo con zonas arqueológicas y buena comida tradicional",
    "Busco fiesta, playa y vida nocturna",
    "Me interesan pueblos coloniales con artesanías y clima frío",
    "Naturaleza, cascadas y aventura lejos de las multitudes",
]


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
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


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
        llm_chain = [{"provider": p.name, "model": p.model, "available": p.available} for p in rec.providers]
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
        "llm": {"provider": cfg.llm.provider, "gemini_configured": any(p["provider"] == "gemini" for p in llm_chain),
                "chain": llm_chain},
        "index": index,
        "lambda": cfg.ranking.lam,
        "examples": EXAMPLES,
    }


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    """Consulta → HyDE / Anti-HyDE → Top 5 y Bottom 5 con explicaciones y tiempos."""
    rec: Optional[Recommender] = app.state.recommender
    if rec is None:
        raise HTTPException(503, detail=app.state.index_error or "Índice no disponible")
    query = req.query.strip()
    if len(query) < 3:
        raise HTTPException(422, detail="Escribe una descripción un poco más larga.")
    try:
        return rec.recommend(query, req.lam).to_dict()
    except LLMError as exc:  # el encoder (Ollama) no respondió
        logger.error("Fallo de embeddings: %s", exc)
        raise HTTPException(
            503, detail="El servicio de embeddings (Ollama) no respondió. Verifica que `ollama serve` esté activo."
        ) from exc
