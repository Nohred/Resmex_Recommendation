"""Limpieza de texto, normalización y caché reanudable de embeddings."""
from __future__ import annotations

import numpy as np

from restmex.archetypes import cluster_keywords, fallback_label
from restmex.data import build_text, clean_text, display_town, truncate_text
from restmex.encoders import OllamaEncoder, embed_corpus, l2_normalize

from conftest import FakeEncoder


def test_clean_text_fixes_mojibake_and_keeps_accents() -> None:
    assert clean_text("un  buen panorama cerca de CancÃºn\n") == "un buen panorama cerca de Cancún"
    assert clean_text("Música y  café") == "Música y café"


def test_build_text_and_truncate() -> None:
    assert build_text("Muy bien", "La comida rica") == "Muy bien. La comida rica"
    assert build_text(float("nan"), "Solo reseña") == "Solo reseña"
    cut = truncate_text("palabra " * 100, 50)
    assert len(cut) <= 50 and not cut.endswith(" ")


def test_display_names() -> None:
    assert display_town("Valle_de_Bravo") == "Valle de Bravo"
    assert display_town("Patzcuaro") == "Pátzcuaro"


def test_l2_normalize() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out[0], [0.6, 0.8])
    assert np.isfinite(out).all()


def test_ollama_encoder_prefixes() -> None:
    class Client:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def embed(self, model, texts, timeout):
            self.seen += texts
            return [[1.0, 1.0]] * len(texts)

    client = Client()
    enc = OllamaEncoder(client, "m")
    vecs = enc.encode_documents(["a"])
    enc.encode_queries(["b"])
    assert client.seen == ["search_document: a", "search_query: b"]
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_ollama_encoder_num_ctx() -> None:
    """`num_ctx` viaja como opción a Ollama y distingue al encoder en la caché."""
    seen: list[dict] = []

    class Client:
        def embed(self, model, texts, timeout, **kwargs):
            seen.append(kwargs)
            return [[1.0, 0.0]] * len(texts)

    plain, wide = OllamaEncoder(Client(), "m"), OllamaEncoder(Client(), "m", num_ctx=4096)
    plain.encode_documents(["a"])
    wide.encode_documents(["a"])
    assert seen == [{}, {"options": {"num_ctx": 4096}}]
    assert (plain.name, wide.name) == ("ollama:m", "ollama:m@ctx4096")


def test_sentence_transformer_encoder_prefixes(monkeypatch) -> None:
    """El backend alternativo aplica los prefijos de documento y consulta del YAML."""
    import sys
    import types

    from restmex.config import Config
    from restmex.encoders import build_encoder

    class FakeST:
        def __init__(self, model: str) -> None:
            self.model, self.seen, self.max_seq_length, self.device = model, [], 128, "cpu"

        def encode(self, texts, batch_size, **kwargs):
            self.seen += list(texts)
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    cfg = Config()
    cfg.encoder.backend, cfg.encoder.st_model = "sentence_transformers", "intfloat/multilingual-e5-large"
    cfg.encoder.doc_prefix, cfg.encoder.query_prefix = "passage: ", "query: "
    cfg.encoder.max_seq_length, cfg.encoder.fp16 = 512, True
    enc = build_encoder(cfg)
    vecs = enc.encode_documents(["reseña"])
    enc.encode_queries(["consulta"])
    assert enc._model.seen == ["passage: reseña", "query: consulta"]
    assert enc._model.max_seq_length == 512
    assert enc.name == "st:intfloat/multilingual-e5-large@len512@fp16"
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)


def test_embed_corpus_resumes(tmp_path) -> None:
    texts = [f"ruinas {i} playa" for i in range(10)]

    class Flaky(FakeEncoder):
        calls = 0

        def _encode(self, t):
            Flaky.calls += 1
            if Flaky.calls == 3:
                raise RuntimeError("interrumpido")
            return super()._encode(t)

    try:
        embed_corpus(Flaky(), texts, tmp_path, batch_size=2, save_every=1)
    except RuntimeError:
        pass
    calls: list[int] = []

    class Counting(FakeEncoder):
        def _encode(self, t):
            calls.append(len(t))
            return super()._encode(t)

    out = embed_corpus(Counting(), texts, tmp_path, batch_size=2, save_every=1)
    assert sum(calls) == 6                       # solo lo que faltaba (4 ya estaban)
    np.testing.assert_allclose(out, FakeEncoder().encode_documents(texts), rtol=1e-6)
    calls.clear()
    embed_corpus(Counting(), texts, tmp_path, batch_size=2)
    assert calls == []                           # caché completa


def test_cluster_keywords_and_fallback_label() -> None:
    texts = ["la pirámide y las ruinas mayas"] * 5 + ["tacos de pescado y ceviche fresco"] * 5
    kws = cluster_keywords(texts, np.array([0] * 5 + [1] * 5), 2, 3)
    assert any("ruinas" in k or "pirámide" in k for k in kws[0])
    assert any("tacos" in k or "ceviche" in k for k in kws[1])
    assert fallback_label(["tacos", "pescado", "ceviche", "x"]) == "Tacos, pescado, ceviche"
    assert fallback_label([]) == "Experiencia general"
