"""Ingesta del CSV de Rest-Mex: limpieza ligera, deduplicado y partición train/val."""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd
from ftfy import fix_text
from ftfy.badness import badness
from sklearn.model_selection import train_test_split

from .config import Config

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_TRUNC_MAS = re.compile(r"(\.\.\.|…)\s*Más\s*$")

# Nombres para mostrar (el CSV no trae acentos ni espacios en Town/Region).
TOWN_NAMES: dict[str, str] = {
    "San_Cristobal_de_las_Casas": "San Cristóbal de las Casas",
    "Teotihuacan": "Teotihuacán",
    "TodosSantos": "Todos Santos",
    "Patzcuaro": "Pátzcuaro",
    "Tepoztlan": "Tepoztlán",
    "Zacatlan": "Zacatlán",
    "Tepotzotlan": "Tepotzotlán",
    "Cuatro_Cienegas": "Cuatro Ciénegas",
}
REGION_NAMES: dict[str, str] = {
    "QuintanaRoo": "Quintana Roo",
    "Estado_de_Mexico": "Estado de México",
    "Yucatan": "Yucatán",
    "Baja_CaliforniaSur": "Baja California Sur",
    "Queretaro": "Querétaro",
    "Michoacan": "Michoacán",
    "San_Luis_Potosi": "San Luis Potosí",
}


def display_town(town: str) -> str:
    """Nombre legible de un pueblo (`Valle_de_Bravo` → `Valle de Bravo`)."""
    return TOWN_NAMES.get(town, town.replace("_", " "))


def display_region(region: str) -> str:
    """Nombre legible de un estado."""
    return REGION_NAMES.get(region, region.replace("_", " "))


def clean_text(text: str) -> str:
    """Limpieza ligera: arregla mojibake, NFKC y espacios. Conserva acentos."""
    if badness(text) > 0:
        text = fix_text(text)
    text = unicodedata.normalize("NFKC", text).replace("�", "")
    text = _TRUNC_MAS.sub("", text)
    return _WS.sub(" ", text).strip()


def truncate_text(text: str, max_chars: int) -> str:
    """Trunca de forma explícita en un límite de palabra (contexto corto del encoder)."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return cut[:space] if space > max_chars * 0.6 else cut


def build_text(title: object, review: object) -> str:
    """Texto de la reseña = título + reseña."""
    t = "" if pd.isna(title) else str(title).strip()
    r = "" if pd.isna(review) else str(review).strip()
    if t and r:
        sep = " " if t[-1] in ".!?…" else ". "
        return clean_text(t + sep + r)
    return clean_text(t or r)


def load_reviews(cfg: Config, sample: Optional[int] = None) -> pd.DataFrame:
    """Carga el CSV y devuelve un DataFrame limpio con la columna `split`.

    Columnas: row_id (fila original del CSV), text, town, region, type,
    polarity, split ("train" | "val").

    `sample` toma una muestra estratificada por pueblo (para pruebas rápidas);
    `data.max_reviews_per_town` aplica un tope por pueblo.
    """
    try:
        raw = pd.read_csv(cfg.raw_csv, encoding="utf-8")
    except UnicodeDecodeError:
        raw = pd.read_csv(cfg.raw_csv, encoding="latin-1")
    required = {"Title", "Review", "Polarity", "Town", "Region", "Type"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")

    seed = cfg.data.seed
    raw = raw.reset_index(names="row_id")
    if sample is not None and sample < len(raw):
        frac = sample / len(raw)
        raw = (
            raw.groupby("Town", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), max(40, int(round(len(g) * frac)))), random_state=seed))
            .sort_values("row_id")
        )
    cap = cfg.data.max_reviews_per_town
    if cap:
        raw = (
            raw.groupby("Town", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), cap), random_state=seed))
            .sort_values("row_id")
        )

    df = pd.DataFrame(
        {
            "row_id": raw["row_id"].to_numpy(),
            "text": [
                truncate_text(build_text(t, r), cfg.data.max_chars)
                for t, r in zip(raw["Title"], raw["Review"])
            ],
            "town": raw["Town"].astype(str).to_numpy(),
            "region": raw["Region"].astype(str).to_numpy(),
            "type": raw["Type"].astype(str).to_numpy(),
            "polarity": raw["Polarity"].astype(float).to_numpy(),
        }
    )
    before = len(df)
    df = df[df["text"].str.len() > 0].drop_duplicates("text", keep="first").reset_index(drop=True)
    logger.info("Reseñas: %d (se descartaron %d vacías/duplicadas)", len(df), before - len(df))

    train_idx, val_idx = train_test_split(
        df.index.to_numpy(),
        test_size=cfg.data.val_fraction,
        stratify=df["town"],
        random_state=seed,
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    logger.info("Partición: %d train / %d val, %d pueblos", len(train_idx), len(val_idx), df["town"].nunique())
    return df
