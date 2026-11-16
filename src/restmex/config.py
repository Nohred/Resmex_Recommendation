"""Carga de configuración (`config.yaml`) en dataclasses tipadas.

Un YAML de experimento (`configs/*.yaml`) puede heredar de otro con la clave
`extends`; las secciones se combinan con merge profundo (ver `load_raw`).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Path = ROOT / "config.yaml"
logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    raw_csv: str = "data/raw/Rest-Mex_2025_train.csv"
    artifacts_dir: str = "artifacts"         # arquetipos e índice; por experimento: artifacts/exp/<nombre>
    reports_dir: str = "reports"             # métricas y figuras; por experimento: reports/exp/<nombre>
    # Raíz compartida de la caché de embeddings (no depende de k ni de λ). Los
    # experimentos la heredan y reutilizan los embeddings de la configuración base.
    embeddings_dir: str = "artifacts"


@dataclass
class DataConfig:
    val_fraction: float = 0.1
    seed: int = 42
    max_chars: int = 1500
    max_reviews_per_town: Optional[int] = None


@dataclass
class EncoderConfig:
    backend: str = "ollama"
    model: str = "nomic-embed-text-v2-moe"
    batch_size: int = 128
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    st_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    embed_timeout: float = 300.0
    keep_alive: str = "30m"


@dataclass
class ArchetypesConfig:
    k: int = 10
    n_init: int = 10
    seed: int = 42
    n_keywords: int = 8
    n_representative: int = 3
    label_with_llm: bool = True


@dataclass
class LLMConfig:
    chain: list[str] = field(default_factory=lambda: ["gemini", "groq", "ollama"])   # web y etiquetas
    eval_chain: list[str] = field(default_factory=lambda: ["groq", "ollama"])       # evaluación con HyDE
    gemini_model: str = "gemini-3.5-flash"
    gemini_thinking_level: Optional[str] = "low"
    gemini_timeout: float = 20.0
    gemini_rpm: int = 10
    gemini_api_key_env: str = "GEMINI_API"
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_reasoning_effort: Optional[str] = "low"
    groq_timeout: float = 30.0
    groq_rpm: int = 30
    groq_api_key_env: str = "GROQ_API"
    ollama_model: str = "qwen2.5:3b"
    temperature: float = 0.4
    num_predict: int = 800
    timeout: float = 90.0                        # timeout del modelo local
    retries: int = 1                             # reintentos por proveedor tras salida mal formada
    cooldown: float = 300.0                      # s que se omite un proveedor tras un error
    cache: bool = True                           # caché en disco de respuestas del LLM
    cache_dir: str = "artifacts/llm_cache"


@dataclass
class RankingConfig:
    lam: float = 0.015
    top_k: int = 5
    bottom_k: int = 5


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    archetypes: ArchetypesConfig = field(default_factory=ArchetypesConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    root: Path = ROOT
    source: Optional[Path] = None            # YAML del que se cargó (None = valores por defecto)

    @property
    def raw_csv(self) -> Path:
        return self.root / self.paths.raw_csv

    @property
    def artifacts_dir(self) -> Path:
        return self.root / self.paths.artifacts_dir

    @property
    def reports_dir(self) -> Path:
        return self.root / self.paths.reports_dir

    @property
    def embeddings_dir(self) -> Path:
        return self.root / self.paths.embeddings_dir


SECTIONS: dict[str, type] = {
    "paths": PathsConfig,
    "data": DataConfig,
    "encoder": EncoderConfig,
    "ollama": OllamaConfig,
    "archetypes": ArchetypesConfig,
    "llm": LLMConfig,
    "ranking": RankingConfig,
}


def _build(cls: type, raw: dict[str, Any], section: str = "") -> Any:
    """Instancia `cls` ignorando (con aviso) las claves desconocidas (`lambda` → `lam`)."""
    raw = dict(raw or {})
    if "lambda" in raw:
        raw["lam"] = raw.pop("lambda")
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - names)
    if unknown:  # un error de dedo en un experimento correría la configuración base sin avisar
        logger.warning("Claves desconocidas en `%s` (se ignoran): %s", section or cls.__name__, unknown)
    return cls(**{k: v for k, v in raw.items() if k in names})


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Combina dos diccionarios: los sub-diccionarios se mezclan clave por clave;
    escalares y listas de `override` reemplazan a los de `base`."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve(path: Path | str, relative_to: Optional[Path] = None) -> Path:
    """Ruta de un YAML: absoluta, relativa a `relative_to`, al directorio actual o a la raíz."""
    p = Path(path)
    if p.is_absolute():
        return p
    bases = ([relative_to] if relative_to else []) + [Path.cwd(), ROOT]
    for base in bases:
        if (base / p).exists():
            return (base / p).resolve()
    return (ROOT / p).resolve()


def load_raw(path: Path | str, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Lee un YAML resolviendo la herencia (`extends: config.yaml`).

    La ruta de `extends` se busca junto al propio archivo y después en la raíz
    del repositorio. El hijo solo declara lo que cambia; el resto se hereda.
    """
    cfg_path = _resolve(path)
    if cfg_path in _seen:
        raise ValueError(f"Herencia circular de configuración en {cfg_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} debe contener un mapeo YAML")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = load_raw(_resolve(parent, cfg_path.parent), _seen + (cfg_path,))
    return deep_merge(base, raw)


def load_config(path: Optional[Path | str] = None) -> Config:
    """Lee el YAML de configuración. `RESTMEX_CONFIG` permite cambiar la ruta.

    Sin `config.yaml` se usan los valores por defecto; una ruta pedida de forma
    explícita (argumento o variable de entorno) que no exista es un error.
    """
    explicit = path or os.environ.get("RESTMEX_CONFIG")
    raw: dict[str, Any] = {}
    source: Optional[Path] = None
    if explicit or DEFAULT_CONFIG.exists():
        source = _resolve(explicit or DEFAULT_CONFIG)
        raw = load_raw(source)
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        logger.warning("Secciones desconocidas en la configuración (se ignoran): %s", unknown)
    sections = {name: _build(cls, raw.get(name, {}), name) for name, cls in SECTIONS.items()}
    return Config(**sections, source=source)
