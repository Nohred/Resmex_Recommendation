"""Evaluación en validación → reports/metrics.json y reports/figures/.

Consultas = reseñas de validación con polaridad 4–5; acierto = su pueblo real
aparece en el Top-k. Compara:
  (a) un centroide por pueblo vs. 10 arquetipos,
  (b) con y sin penalización por popularidad (barrido de λ),
  (c) con y sin HyDE en una muestra pequeña (usa el LLM).

Uso:
    python scripts/evaluate.py                    # todo
    python scripts/evaluate.py --hyde-sample 0    # sin la parte (c)
    python scripts/evaluate.py --sample 2000      # sobre artifacts/sample/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restmex.config import load_config  # noqa: E402
from restmex.data import load_reviews  # noqa: E402
from restmex.encoders import build_encoder  # noqa: E402
from restmex.evaluate import (  # noqa: E402
    balanced_sample, choose_lambda, lambda_sweep, popularity_bias, retrieval_metrics, true_ranks,
)
from restmex.llm import build_llm_chain  # noqa: E402
from restmex.ollama_client import OllamaClient  # noqa: E402
from restmex.query_expansion import expand_query  # noqa: E402
from restmex.ranking import adjusted_scores  # noqa: E402
from restmex.retrieval import ArchetypeIndex  # noqa: E402

logger = logging.getLogger("evaluate")
LAMBDAS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
KEYS = ["hit@1", "hit@5", "mrr", "macro_hit@1", "macro_hit@5", "macro_mrr"]


def hyde_comparison(cfg, index, art_dir: Path, val: pd.DataFrame, val_emb: np.ndarray,
                    true_town: np.ndarray, n: int, lam: float, sample: int | None) -> dict:
    """(c) Muestra pequeña: reseña como consulta, con y sin HyDE."""
    rng = np.random.default_rng(cfg.data.seed)
    pick = np.sort(rng.choice(len(val), size=min(n, len(val)), replace=False))
    texts_df = load_reviews(cfg, sample=sample).set_index("row_id")
    texts = [texts_df.loc[rid, "text"] for rid in val["row_id"].to_numpy()[pick]]

    client = OllamaClient(cfg.ollama.host, cfg.ollama.keep_alive)
    encoder = build_encoder(cfg, client)
    providers = build_llm_chain(cfg, client)
    cache_path = art_dir / "hyde_eval_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    hydes, used = [], []
    for i, (rid, text) in enumerate(zip(val["row_id"].to_numpy()[pick], texts)):
        key = str(int(rid))
        if key not in cache or cache[key]["provider"] == "plantilla":
            exp = expand_query(text[:600], providers, cfg.llm)
            cache[key] = {"hyde": exp.hyde, "provider": exp.provider, "model": exp.model}
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            logger.info("HyDE %d/%d (%s)", i + 1, len(pick), exp.provider)
        hydes.append(cache[key]["hyde"])
        used.append(cache[key]["provider"])

    variants = {
        "reseña como documento (sin LLM)": val_emb[pick],
        "consulta cruda, prefijo search_query (sin HyDE)": encoder.encode_queries(texts),
        "HyDE generado por el LLM": encoder.encode_documents(hydes),
    }
    out: dict = {"n": int(len(pick)), "lambda": lam, "providers": pd.Series(used).value_counts().to_dict(), "variants": {}}
    for name, vecs in variants.items():
        sims, _ = index.town_scores(vecs)
        adj = adjusted_scores(sims, index.n_reviews, index.n_max, lam)
        m = retrieval_metrics(true_ranks(adj, true_town[pick]), true_town[pick], index.n_towns)
        out["variants"][name] = {k: round(m[k], 4) for k in ("hit@1", "hit@5", "mrr")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample", type=int, default=None, help="Evalúa artifacts/sample/ (misma muestra)")
    parser.add_argument("--hyde-sample", type=int, default=100, help="Consultas para la comparación con HyDE (0 = omitir)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    art_dir = cfg.artifacts_dir / "sample" if args.sample else cfg.artifacts_dir
    rep_dir = cfg.reports_dir / "sample" if args.sample else cfg.reports_dir
    (rep_dir / "figures").mkdir(parents=True, exist_ok=True)

    index = ArchetypeIndex.load(art_dir)
    rows = pd.read_csv(art_dir / "embeddings_index.csv")
    emb = np.load(art_dir / "embeddings.npy", mmap_mode="r")
    if len(rows) != len(emb):
        raise SystemExit("embeddings.npy y embeddings_index.csv no coinciden; ejecuta `make index`.")
    centroids = np.load(art_dir / "town_centroids.npy")

    mask = ((rows["split"] == "val") & (rows["polarity"] >= 4)).to_numpy()
    val = rows[mask].reset_index(drop=True)
    val_emb = np.asarray(emb[np.flatnonzero(mask)])
    pos = {t: i for i, t in enumerate(index.towns)}
    true_town = val["town"].map(pos).to_numpy()
    logger.info("Consultas de validación (polaridad 4–5): %d", len(val))

    n_rev, n_max, T = index.n_reviews, index.n_max, index.n_towns
    sims_arch, _ = index.town_scores(val_emb)
    sims_cent = val_emb @ centroids.T
    bal = balanced_sample(true_town, T, cfg.data.seed)

    sweep = lambda_sweep(sims_arch, true_town, n_rev, n_max, LAMBDAS, bal)
    lam = choose_lambda(sweep)
    row0 = next(r for r in sweep if r["lambda"] == 0.0)
    row_star = next(r for r in sweep if r["lambda"] == lam)

    def variant(sims: np.ndarray, lam_: float) -> dict:
        adj = adjusted_scores(sims, n_rev, n_max, lam_)
        m = retrieval_metrics(true_ranks(adj, true_town), true_town, T)
        m["spearman_balanced"] = popularity_bias(adj[bal], n_rev)["spearman"]
        return {k: round(v, 4) for k, v in m.items()}

    metrics: dict = {
        "encoder": index.stats.get("encoder"),
        "n_train": index.stats.get("n_train"),
        "n_queries": int(len(val)),
        "n_queries_balanced": int(len(bal)),
        "n_towns": T,
        "n_archetypes": len(index.meta),
        "lambda_chosen": lam,
        "lambda_criterion": "menor |Spearman| (consultas balanceadas) entre los λ con MRR macro ≥ 99 % del mejor",
        "variants": {
            "1 centroide por pueblo, sin penalización": variant(sims_cent, 0.0),
            "10 arquetipos, sin penalización": variant(sims_arch, 0.0),
            f"10 arquetipos, λ={lam}": variant(sims_arch, lam),
            f"1 centroide por pueblo, λ={lam}": variant(sims_cent, lam),
        },
        "popularity_bias": {
            "spearman_before": round(row0["spearman_balanced"], 4),
            "spearman_after": round(row_star["spearman_balanced"], 4),
            "spearman_before_all_queries": round(row0["spearman_all"], 4),
            "spearman_after_all_queries": round(row_star["spearman_all"], 4),
        },
        "lambda_sweep": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "top5_frequency_balanced"}
            for r in sweep
        ],
    }
    if args.hyde_sample > 0:
        metrics["hyde_comparison"] = hyde_comparison(
            cfg, index, art_dir, val, val_emb, true_town, args.hyde_sample, lam, args.sample
        )

    (rep_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    from restmex.plots import make_figures  # import tardío: matplotlib solo aquí

    make_figures(metrics, sweep, index, lam, rep_dir / "figures")

    logger.info("λ elegido: %s", lam)
    for name, m in metrics["variants"].items():
        logger.info("%-42s %s", name, {k: m[k] for k in KEYS + ["spearman_balanced"]})
    if "hyde_comparison" in metrics:
        logger.info("HyDE: %s", json.dumps(metrics["hyde_comparison"], ensure_ascii=False))
    logger.info("Resultados en %s", rep_dir)


if __name__ == "__main__":
    main()
