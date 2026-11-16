"""Configuración por experimento: herencia (`extends`), rutas propias y caché compartida de embeddings."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from restmex import config as config_mod
from restmex.config import ROOT, SECTIONS, deep_merge, load_config, load_raw
from restmex.encoders import embed_corpus, embeddings_cache_dir, locate_embeddings, write_embeddings_ref

from conftest import FakeEncoder

BASE_FILES = {
    "archetypes.npy", "archetypes_meta.json", "embeddings.npy", "embeddings_state.json",
    "embeddings_index.csv", "town_centroids.npy", "town_stats.json",
}


def write_yaml(path: Path, content: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, allow_unicode=True), encoding="utf-8")
    return path


# --- config.yaml base: igual que antes -------------------------------------------------

def test_base_config_unchanged(monkeypatch) -> None:
    """Con `config.yaml` la carga da lo mismo que el cargador anterior (sin herencia)."""
    monkeypatch.delenv("RESTMEX_CONFIG", raising=False)
    cfg = load_config()
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert "extends" not in raw
    for name, cls in SECTIONS.items():
        section = dict(raw.get(name, {}))
        if "lambda" in section:
            section["lam"] = section.pop("lambda")
        assert asdict(getattr(cfg, name)) == asdict(cls(**section)), name
    assert cfg.artifacts_dir == ROOT / "artifacts"
    assert cfg.reports_dir == ROOT / "reports"
    assert cfg.embeddings_dir == cfg.artifacts_dir      # los embeddings siguen en artifacts/
    assert (cfg.archetypes.k, cfg.ranking.lam) == (10, 0.015)
    assert cfg.encoder.model == "nomic-embed-text-v2-moe"
    assert cfg.source == ROOT / "config.yaml"


def test_missing_explicit_config_is_an_error(tmp_path, monkeypatch) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no_existe.yaml")
    monkeypatch.setenv("RESTMEX_CONFIG", str(tmp_path / "tampoco.yaml"))
    with pytest.raises(FileNotFoundError):
        load_config()


# --- herencia ---------------------------------------------------------------------------

def test_deep_merge() -> None:
    base = {"a": {"x": 1, "y": [1, 2], "z": {"p": 1, "q": 2}}, "b": 1}
    out = deep_merge(base, {"a": {"y": [3], "z": {"q": 5}}, "c": 2})
    assert out == {"a": {"x": 1, "y": [3], "z": {"p": 1, "q": 5}}, "b": 1, "c": 2}
    assert base["a"]["z"] == {"p": 1, "q": 2}           # no modifica la base


def test_extends_inherits_base_config(tmp_path, monkeypatch) -> None:
    """`extends: config.yaml` se resuelve contra la raíz; solo cambia lo declarado."""
    monkeypatch.delenv("RESTMEX_CONFIG", raising=False)
    base = load_config()
    exp = load_config(write_yaml(tmp_path / "configs" / "2026-01-01_k20.yaml", {
        "extends": "config.yaml",
        "paths": {"artifacts_dir": "artifacts/exp/k20", "reports_dir": "reports/exp/k20"},
        "archetypes": {"k": 20},
        "ranking": {"lambda": 0.03},
        "llm": {"eval_chain": ["ollama"]},
    }))
    assert exp.artifacts_dir == ROOT / "artifacts" / "exp" / "k20"
    assert exp.reports_dir == ROOT / "reports" / "exp" / "k20"
    assert exp.embeddings_dir == base.embeddings_dir    # caché compartida con la base
    assert exp.paths.raw_csv == base.paths.raw_csv
    assert (exp.archetypes.k, exp.archetypes.n_init) == (20, base.archetypes.n_init)
    assert exp.ranking.lam == 0.03 and exp.ranking.top_k == base.ranking.top_k
    assert exp.llm.eval_chain == ["ollama"]             # las listas se reemplazan
    assert exp.llm.chain == base.llm.chain and exp.llm.groq_model == base.llm.groq_model
    assert asdict(exp.encoder) == asdict(base.encoder) and asdict(exp.data) == asdict(base.data)

    monkeypatch.setenv("RESTMEX_CONFIG", str(tmp_path / "configs" / "2026-01-01_k20.yaml"))
    assert load_config().archetypes.k == 20             # también por variable de entorno


def test_extends_chain_and_cycle(tmp_path) -> None:
    write_yaml(tmp_path / "a.yaml", {"archetypes": {"k": 4, "n_init": 2}, "ranking": {"lambda": 0.1}})
    write_yaml(tmp_path / "b.yaml", {"extends": "a.yaml", "archetypes": {"k": 6}})
    c = write_yaml(tmp_path / "sub" / "c.yaml", {"extends": "../b.yaml", "ranking": {"top_k": 3}})
    raw = load_raw(c)
    assert raw == {"archetypes": {"k": 6, "n_init": 2}, "ranking": {"lambda": 0.1, "top_k": 3}}
    cfg = load_config(c)
    assert (cfg.archetypes.k, cfg.archetypes.n_init, cfg.ranking.lam, cfg.ranking.top_k) == (6, 2, 0.1, 3)

    write_yaml(tmp_path / "x.yaml", {"extends": "y.yaml"})
    write_yaml(tmp_path / "y.yaml", {"extends": "x.yaml"})
    with pytest.raises(ValueError, match="circular"):
        load_raw(tmp_path / "x.yaml")
    write_yaml(tmp_path / "huerfano.yaml", {"extends": "no_existe_en_ningun_lado.yaml"})
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "huerfano.yaml")


def test_unknown_keys_warn(tmp_path, caplog) -> None:
    path = write_yaml(tmp_path / "typo.yaml", {"ranking": {"lamda": 0.5}, "rankin": {}})
    with caplog.at_level("WARNING", logger=config_mod.logger.name):
        cfg = load_config(path)
    assert cfg.ranking.lam == 0.015                      # el valor por defecto, pero con aviso
    assert "lamda" in caplog.text and "rankin" in caplog.text


# --- caché de embeddings compartida -----------------------------------------------------

class Counting(FakeEncoder):
    """FakeEncoder que cuenta los textos codificados."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.n = 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        self.n += len(texts)
        return super()._encode(texts)


def test_embeddings_cache_shared_between_experiments(tmp_path) -> None:
    texts = [f"ruinas {i} playa" for i in range(10)]
    shared = tmp_path / "artifacts"
    exp_a, exp_b = shared / "exp" / "a", shared / "exp" / "b"

    # Configuración base: la caché vive en artifacts/ con los nombres de siempre.
    assert embeddings_cache_dir(Counting(), texts, shared, shared) == shared
    base = embed_corpus(Counting(), texts, shared, batch_size=4)
    before = (shared / "embeddings.npy").read_bytes()

    # Mismo encoder y mismos textos (cambien k o λ): se reutiliza sin recalcular.
    enc = Counting()
    cache = embeddings_cache_dir(enc, texts, exp_a, shared)
    assert cache == shared
    np.testing.assert_array_equal(embed_corpus(enc, texts, cache, batch_size=4), base)
    assert enc.n == 0

    # Otro encoder u otro preprocesamiento: caché propia, la base no se toca.
    other = Counting("otro-encoder")
    cache_enc = embeddings_cache_dir(other, texts, exp_a, shared)
    cache_pre = embeddings_cache_dir(Counting(), [t.lower() + "!" for t in texts], exp_a, shared)
    assert cache_enc.parent == cache_pre.parent == shared / "embeddings_cache"
    assert len({cache_enc, cache_pre, shared}) == 3
    embed_corpus(other, texts, cache_enc, batch_size=4)
    assert other.n == len(texts)
    assert (shared / "embeddings.npy").read_bytes() == before
    assert json.loads((shared / "embeddings_state.json").read_text())["encoder"] == "fake"

    # Un segundo experimento con ese encoder comparte la misma caché (no depende del experimento).
    again = Counting("otro-encoder")
    assert embeddings_cache_dir(again, texts, exp_b, shared) == cache_enc
    embed_corpus(again, texts, cache_enc, batch_size=4)
    assert again.n == 0


def test_embeddings_ref_roundtrip(tmp_path) -> None:
    art, cache = tmp_path / "artifacts" / "exp" / "a", tmp_path / "artifacts" / "embeddings_cache" / "abc"
    art.mkdir(parents=True)
    assert locate_embeddings(art, tmp_path) == art / "embeddings.npy"      # sin puntero: como antes
    write_embeddings_ref(art, art, tmp_path)
    assert not (art / "embeddings_ref.json").exists()
    write_embeddings_ref(art, cache, tmp_path)
    assert json.loads((art / "embeddings_ref.json").read_text()) == {"dir": "artifacts/embeddings_cache/abc"}
    assert locate_embeddings(art, tmp_path) == cache / "embeddings.npy"


# --- de punta a punta con los scripts ---------------------------------------------------

def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def test_scripts_base_layout_and_experiment_isolation(tmp_path, monkeypatch) -> None:
    """Base: mismos archivos que hoy en artifacts/ y reports/. Experimento: sus propios
    directorios, sin recalcular embeddings y sin tocar los resultados de la base."""
    words = ["ruinas", "playa", "fiesta", "frío"]
    rows = [
        {"Title": f"Visita {i} a {town}", "Review": f"{words[t % 4]} {words[i % 4]} {words[(i // 4) % 4]} reseña {i}",
         "Polarity": 4 + i % 2, "Town": town, "Region": "Estado", "Type": "Attractive"}
        for t, town in enumerate(["Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis"]) for i in range(40)
    ]
    (tmp_path / "data").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "reviews.csv", index=False)
    base_yaml = write_yaml(tmp_path / "config.yaml", {
        "paths": {"raw_csv": "data/reviews.csv", "artifacts_dir": "artifacts", "reports_dir": "reports"},
        "archetypes": {"k": 2, "n_init": 1, "label_with_llm": False},
    })
    exp_yaml = write_yaml(tmp_path / "configs" / "2026-01-01_k3.yaml", {
        "extends": "../config.yaml",
        "paths": {"artifacts_dir": "artifacts/exp/k3", "reports_dir": "reports/exp/k3"},
        "archetypes": {"k": 3},
        "ranking": {"lambda": 0.05},
    })

    encoder = Counting()
    build, evaluate = load_script("build_index"), load_script("evaluate")

    def rooted(path=None):
        cfg = load_config(path)
        cfg.root = tmp_path
        return cfg

    for module in (build, evaluate):
        monkeypatch.setattr(module, "load_config", rooted)
    monkeypatch.setattr(build, "build_encoder", lambda cfg, client=None: encoder)

    def run(module, *args: str) -> None:
        monkeypatch.setattr(sys, "argv", [module.__name__, *args])
        module.main()

    art, rep = tmp_path / "artifacts", tmp_path / "reports"
    run(build, "--config", str(base_yaml))
    run(evaluate, "--config", str(base_yaml), "--hyde-sample", "0")
    assert {p.name for p in art.iterdir()} == BASE_FILES            # disposición de siempre
    assert (rep / "metrics.json").exists() and (rep / "figures").is_dir()
    assert encoder.n == 240
    base_art, base_rep = snapshot(art), snapshot(rep)

    run(build, "--config", str(exp_yaml))
    run(evaluate, "--config", str(exp_yaml), "--hyde-sample", "0")
    assert encoder.n == 240                                          # embeddings reutilizados
    exp_files = {p.name for p in (art / "exp" / "k3").iterdir()}
    assert exp_files == (BASE_FILES - {"embeddings.npy", "embeddings_state.json"}) | {"embeddings_ref.json"}
    assert json.loads((rep / "exp" / "k3" / "metrics.json").read_text())["n_archetypes"] == 18
    assert json.loads((rep / "metrics.json").read_text())["n_archetypes"] == 12
    assert snapshot(art) == base_art and snapshot(rep) == base_rep   # la base queda intacta
