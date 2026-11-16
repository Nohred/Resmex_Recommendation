"""Filtros y pesos de calidad de reseñas (`review_quality`): apagados por defecto."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from restmex import review_quality as rq
from restmex.archetypes import build_archetypes
from restmex.config import Config, ReviewQualityConfig, load_config
from restmex.encoders import l2_normalize


def _frame(n_per_town: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    texts = [
        "Excelente lugar, muy bonito y recomendado",
        "Las cascadas de agua turquesa y el cenote del pueblo son espectaculares",
        "La habitación limpia y el personal de recepción muy amable",
        "Caminar por las calles empedradas del centro y la iglesia colonial",
        "Great place to visit with the whole family, we loved the beach",
        "Las cascadas de agua turquesa y el cenote del pueblo son espectaculares!",
    ]
    rows = []
    for town, region in (("Bacalar", "QuintanaRoo"), ("Valle_de_Bravo", "Estado_de_Mexico")):
        for i in range(n_per_town):
            rows.append({"row_id": len(rows), "text": texts[i % len(texts)], "town": town, "region": region,
                         "type": ["Attractive", "Attractive", "Hotel", "Attractive", "Restaurant", "Attractive"][i % 6],
                         "polarity": 5.0 - (i % 3), "split": "train"})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(df), 8))
    emb[5] = emb[1] + 1e-3                    # casi duplicado de la fila 1 (mismo pueblo)
    emb[11] = emb[7] + 1e-3
    return df, l2_normalize(emb).astype(np.float32)


def test_defaults_are_off() -> None:
    cfg = Config()
    assert not rq.is_active(cfg.review_quality)
    assert cfg.review_quality.mask_place_names is False
    assert load_config().review_quality == ReviewQualityConfig()     # config.yaml no los enciende


def test_fold_keeps_length_and_strips_accents() -> None:
    text = "Ñandú en Pátzcuaro, MÉXICO"
    assert rq.fold(text) == "ñandu en patzcuaro, mexico"
    assert len(rq.fold(text)) == len(text)


def test_content_words_drop_generic_praise() -> None:
    assert rq.content_words("Excelente lugar, muy bonito y recomendado") == []
    assert rq.content_words("Las cascadas de agua turquesa") == ["cascadas", "agua", "turquesa"]


def test_mask_names_keeps_text_and_generic_words() -> None:
    pattern = rq.all_places_pattern([("Isla_Mujeres", "QuintanaRoo"), ("Valle_de_Bravo", "Estado_de_Mexico")])
    out = rq.mask_names("Fuimos a Isla Mujeres (Quintana Roo) y a un valle precioso en Valle de Bravo.", pattern)
    assert "Isla Mujeres" not in out and "Quintana" not in out and "Valle de Bravo" not in out
    assert "valle precioso" in out and "Fuimos" in out and "()" not in out


def test_place_mention_and_service_only() -> None:
    df, emb = _frame()
    df.loc[1, "text"] = "Bacalar y su laguna de siete colores"
    feats = rq.review_features(df, emb, detect_language=False)
    assert bool(feats.loc[1, "mentions_place"]) and not bool(feats.loc[0, "mentions_place"])
    assert bool(feats.loc[2, "service_only"])           # Hotel: habitación, recepción, sin lugar
    assert not bool(feats.loc[3, "service_only"])       # Attractive
    assert set(feats["language"]) == {"?"}


def test_near_duplicates_keep_first_copy() -> None:
    df, emb = _frame()
    dup, twin = rq.near_duplicates(emb, df["town"].to_numpy(), 0.95)
    assert dup[5] and not dup[1] and twin[1] and twin[5]
    assert dup[11] and not dup[7]
    assert dup.sum() == 2


def test_select_reviews_filters_floor_and_weights() -> None:
    df, emb = _frame()
    feats = rq.review_features(df, emb, languages=["es", "es", "es", "es", "en", "es"] * 2)
    cfg = ReviewQualityConfig(min_content_words=3, drop_other_languages=True, drop_near_duplicates=True,
                              min_reviews_per_town=0)
    sel = rq.select_reviews(df, feats, cfg)
    assert not sel.keep[0] and not sel.keep[4] and not sel.keep[5] and sel.keep[1]
    assert sel.reasons[4] == "otro idioma"
    assert (sel.weights[~sel.keep] == 0).all()
    # piso por pueblo: se devuelven las más informativas hasta llegar al mínimo
    floor = rq.select_reviews(df, feats, ReviewQualityConfig(min_content_words=50, min_reviews_per_town=2))
    assert floor.keep.sum() == 4 and floor.restored == {"Bacalar": 2, "Valle_de_Bravo": 2}
    # pesos: informatividad × polaridad × tipo
    w = rq.select_reviews(df, feats, ReviewQualityConfig(weight_informativeness=True, weight_polarity=True,
                                                         type_weights={"Hotel": 0.5}))
    expected = (0.1 + 0.9 * feats["informativeness"].to_numpy()) * df["polarity"].to_numpy() / 5
    expected[df["type"].to_numpy() == "Hotel"] *= 0.5
    np.testing.assert_allclose(w.weights, expected)


def test_generic_fraction_is_per_town() -> None:
    df, emb = _frame()
    feats = rq.review_features(df, emb, detect_language=False)
    sel = rq.select_reviews(df, feats, ReviewQualityConfig(drop_generic_frac=0.34, min_reviews_per_town=0))
    dropped = df.loc[~sel.keep, "town"].value_counts()
    assert dropped.to_dict() == {"Bacalar": 2, "Valle_de_Bravo": 2}


def test_build_archetypes_with_filter_keeps_full_popularity() -> None:
    df, emb = _frame(n_per_town=12)
    cfg = Config()
    cfg.archetypes.k = 2
    cfg.archetypes.n_init = 1
    cfg.review_quality = ReviewQualityConfig(min_content_words=3, min_reviews_per_town=0, weight_informativeness=True)
    vectors, meta, stats = build_archetypes(df, emb, cfg)
    town = stats["towns"]["Bacalar"]
    assert town["n_reviews"] == 12 and town["n_reviews_used"] < 12
    assert vectors.shape == (4, 8)


def test_unknown_review_quality_key_warns(tmp_path, caplog) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("review_quality:\n  min_content_word: 3\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        cfg = load_config(path)
    assert cfg.review_quality.min_content_words is None
    assert "min_content_word" in caplog.text


@pytest.mark.parametrize("text", ["Great place to visit with the family, loved it"])
def test_detect_languages_offline(text: str) -> None:
    pytest.importorskip("lingua")
    assert rq.detect_languages([text, "Muy bonito lugar para visitar con la familia", "Excelente"]) == ["en", "es", "?"]
