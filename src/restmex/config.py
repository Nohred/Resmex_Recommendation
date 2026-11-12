"""Carga de configuración (`config.yaml`) en dataclasses tipadas."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Path = ROOT / "config.yaml"


@dataclass
class PathsConfig:
    raw_csv: str = "data/raw/Rest-Mex_2025_train.csv"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"


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
    provider: str = "gemini"                     # gemini | ollama
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_models: list[str] = field(default_factory=list)
    gemini_thinking_level: Optional[str] = "low"
    gemini_timeout: float = 15.0
    api_key_env: str = "GEMINI_API"
    ollama_model: str = "qwen2.5:3b"
    temperature: float = 0.4
    num_predict: int = 800
    timeout: float = 90.0                        # timeout del modelo local
    retries: int = 1                             # reintentos por proveedor tras salida mal formada
    cooldown: float = 300.0                      # s que se omite un proveedor tras fallar


@dataclass
class RankingConfig:
    lam: float = 0.05
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

    @property
    def raw_csv(self) -> Path:
        return self.root / self.paths.raw_csv

    @property
    def artifacts_dir(self) -> Path:
        return self.root / self.paths.artifacts_dir

    @property
    def reports_dir(self) -> Path:
        return self.root / self.paths.reports_dir


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Instancia `cls` ignorando claves desconocidas (`lambda` → `lam`)."""
    raw = dict(raw or {})
    if "lambda" in raw:
        raw["lam"] = raw.pop("lambda")
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def load_config(path: Optional[Path | str] = None) -> Config:
    """Lee el YAML de configuración. `RESTMEX_CONFIG` permite cambiar la ruta."""
    cfg_path = Path(path or os.environ.get("RESTMEX_CONFIG", DEFAULT_CONFIG))
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return Config(
        paths=_build(PathsConfig, raw.get("paths", {})),
        data=_build(DataConfig, raw.get("data", {})),
        encoder=_build(EncoderConfig, raw.get("encoder", {})),
        ollama=_build(OllamaConfig, raw.get("ollama", {})),
        archetypes=_build(ArchetypesConfig, raw.get("archetypes", {})),
        llm=_build(LLMConfig, raw.get("llm", {})),
        ranking=_build(RankingConfig, raw.get("ranking", {})),
    )
