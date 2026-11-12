"""Offline: datos → embeddings (caché reanudable) → arquetipos por pueblo.

Uso:
    python scripts/build_index.py                # dataset completo
    python scripts/build_index.py --sample 2000  # prueba rápida (artifacts/sample/)
    python scripts/build_index.py --benchmark 500
    python scripts/build_index.py --relabel      # solo regenera las etiquetas con el LLM
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restmex.archetypes import build_archetypes, label_archetypes, town_centroids  # noqa: E402
from restmex.config import load_config  # noqa: E402
from restmex.data import load_reviews  # noqa: E402
from restmex.encoders import build_encoder, embed_corpus  # noqa: E402
from restmex.llm import build_llm_chain  # noqa: E402
from restmex.ollama_client import OllamaClient  # noqa: E402

logger = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Ruta a config.yaml")
    parser.add_argument("--sample", type=int, default=None, help="Muestra estratificada de N reseñas")
    parser.add_argument("--benchmark", type=int, default=None, help="Solo mide el rendimiento con N reseñas")
    parser.add_argument("--relabel", action="store_true", help="Solo re-etiquetar arquetipos existentes")
    parser.add_argument("--no-labels", action="store_true", help="No etiquetar arquetipos con el LLM")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    if args.no_labels:
        cfg.archetypes.label_with_llm = False
    client = OllamaClient(cfg.ollama.host, cfg.ollama.keep_alive)
    encoder = build_encoder(cfg, client)

    out_dir = cfg.artifacts_dir / "sample" if args.sample else cfg.artifacts_dir
    meta_path = out_dir / "archetypes_meta.json"

    def save_meta(meta: list) -> None:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.relabel:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_llm = label_archetypes(meta, build_llm_chain(cfg, client), cfg)
        save_meta(meta)
        logger.info("Re-etiquetado: %d de %d arquetipos con LLM", n_llm, len(meta))
        return

    df = load_reviews(cfg, sample=args.sample or args.benchmark)

    if args.benchmark:
        texts = df["text"].tolist()[: args.benchmark]
        encoder.encode_documents(texts[:8])  # carga del modelo
        t0 = time.time()
        for i in range(0, len(texts), cfg.encoder.batch_size):
            encoder.encode_documents(texts[i : i + cfg.encoder.batch_size])
        rate = len(texts) / (time.time() - t0)
        total = sum(1 for _ in open(cfg.raw_csv, encoding="utf-8", errors="ignore"))
        logger.info("%.1f textos/s → dataset completo (~%d líneas) ≈ %.0f min como máximo", rate, total, total / rate / 60)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    emb = embed_corpus(encoder, df["text"].tolist(), out_dir, cfg.encoder.batch_size)
    df.drop(columns=["text"]).to_csv(out_dir / "embeddings_index.csv", index=False)

    train_mask = (df["split"] == "train").to_numpy()
    df_train = df[train_mask].reset_index(drop=True)
    emb_train = emb[train_mask]

    t0 = time.time()
    vectors, meta, stats = build_archetypes(df_train, emb_train, cfg)
    logger.info("Arquetipos: %d en %.1f s", len(meta), time.time() - t0)

    np.save(out_dir / "archetypes.npy", vectors)
    np.save(out_dir / "town_centroids.npy", town_centroids(df_train, emb_train))
    save_meta(meta)  # con etiquetas de respaldo, por si el LLM falla después
    stats["encoder"] = encoder.name
    stats["n_train"] = int(train_mask.sum())
    stats["n_val"] = int((~train_mask).sum())
    (out_dir / "town_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    n_llm = 0
    if cfg.archetypes.label_with_llm:
        n_llm = label_archetypes(meta, build_llm_chain(cfg, client), cfg)
        save_meta(meta)
    logger.info("Guardado en %s (%d etiquetas del LLM, %d de respaldo)", out_dir, n_llm, len(meta) - n_llm)


if __name__ == "__main__":
    main()
