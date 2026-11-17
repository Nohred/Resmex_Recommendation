"""Calidad de las reseñas que forman los arquetipos: diagnóstico y barrido de filtros/pesos.

Encoder, k y agregación salen del YAML (fijos en todo el barrido); la rejilla de variantes vive
en la sección `barrido_calidad:`. Las consultas de evaluación no cambian entre variantes:

- Holdout enmascarado con los `row_id` fijos de `eval/holdout_ids.json`, partido en calibración
  y prueba por `eval/holdout_calib_test.json` (los dos archivos ya existen; no se regeneran).
  Sus vectores (reseña enmascarada como documento y HyDE / Anti-HyDE) se reutilizan del barrido
  de arquetipos (`vectores_holdout`), así que no hay llamadas al LLM.
- Gold del equipo: HyDE / Anti-HyDE desde la caché del LLM (sin llamadas nuevas). Para elegir
  solo cuentan las consultas `dev` y `calibration` del banco; las `test` se evalúan en la etapa
  de prueba.
- λ se recalibra por variante con `choose_lambda` sobre val (polaridad 4–5) sin las reseñas de
  la mitad de prueba, como en el barrido de arquetipos.

Etapas:
    python scripts/sweep_quality.py --config configs/<yaml> --etapa diagnostico   # reports/diagnostico_resenas.md
    python scripts/sweep_quality.py --config configs/<yaml> --etapa barrido       # una fila por variante
    python scripts/sweep_quality.py --config configs/<yaml> --etapa semillas      # ruido del K-Means (semillas_extra)
    python scripts/sweep_quality.py --config configs/<yaml> --etapa combinar      # las 2 mejores ideas juntas
    python scripts/sweep_quality.py --config configs/<yaml> --etapa prueba        # una sola vez
    python scripts/sweep_quality.py --config configs/<yaml> --etapa descartadas   # 20 descartadas para revisar a mano
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import LAMBDAS  # noqa: E402
from evaluate_report import git_commit, load_holdout  # noqa: E402
from sweep_archetypes import BOOT_SEED, N_BOOT, positions, split_holdout, summary, with_ci  # noqa: E402
from restmex import report as rp  # noqa: E402
from restmex import review_quality as rq_mod  # noqa: E402
from restmex.aggregation import aggregate_per_town  # noqa: E402
from restmex.bootstrap import paired_bootstrap  # noqa: E402
from restmex.config import ReviewQualityConfig, load_config, load_raw  # noqa: E402
from restmex.data import load_reviews  # noqa: E402
from restmex.encoders import build_encoder, l2_normalize, locate_embeddings  # noqa: E402
from restmex.evaluate import balanced_sample, choose_lambda, lambda_sweep  # noqa: E402
from restmex.ranking import popularity_penalty  # noqa: E402

logger = logging.getLogger("sweep_quality")
GOLD = ROOT / "eval/gold_queries.json"
BANK = ROOT / "eval/query_bank.json"
DIAG = ROOT / "reports/diagnostico_resenas.md"
REF, REF_SAMPLE = "sin_filtro", "muestra_1500"
N_EXAMPLES = 10


# ---------------------------------------------------------------- contexto
class Context:
    def __init__(self, cfg, jobs: int) -> None:
        self.cfg, self.jobs = cfg, jobs
        self.grid = load_raw(cfg.source)["barrido_calidad"]
        self.out = cfg.artifacts_dir
        self.out.mkdir(parents=True, exist_ok=True)
        base = cfg.embeddings_dir
        state = json.loads((base / "embeddings_state.json").read_text())
        self.encoder = build_encoder(cfg)
        rows = pd.read_csv(base / "embeddings_index.csv")
        if state["encoder"] != self.encoder.name or state["done"] != state["n"] or state["n"] != len(rows):
            raise SystemExit(f"La caché de {base} no es del encoder {self.encoder.name} o está incompleta.")
        self.emb = np.load(locate_embeddings(base, cfg.root), mmap_mode="r")
        df = load_reviews(cfg)
        if df["row_id"].tolist() != rows["row_id"].tolist():
            raise SystemExit("El corpus no coincide con embeddings_index.csv")
        self.df = df
        self.train = np.flatnonzero((df["split"] == "train").to_numpy())
        self.towns = sorted(df["town"].unique())
        self.T = len(self.towns)
        self.pos = {t: i for i, t in enumerate(self.towns)}
        tr_town = df["town"].to_numpy()[self.train]
        self.n_reviews = np.array([(tr_town == t).sum() for t in self.towns], dtype=np.float64)
        self.n_max = float(self.n_reviews.max())
        self._lang: Optional[pd.Series] = None
        self._feats: Optional[pd.DataFrame] = None

    # -- rasgos
    def languages(self) -> pd.Series:
        """Idioma por row_id (todo el corpus), detectado una vez y guardado en disco."""
        if self._lang is None:
            path = self.out / "idioma.parquet"
            if path.exists():
                lang = pd.read_parquet(path)
            else:
                t0 = time.time()
                lang = pd.DataFrame({"row_id": self.df["row_id"], "language": rq_mod.detect_languages(self.df["text"].tolist())})
                lang.to_parquet(path, index=False)
                logger.info("Idioma detectado en %.1f min", (time.time() - t0) / 60)
            self._lang = lang.set_index("row_id")["language"]
        return self._lang

    def features(self, rows: np.ndarray, name: str) -> pd.DataFrame:
        """Rasgos de las filas `rows` (posiciones en df); IDF y centroide salen de esas filas."""
        path = self.out / f"rasgos_{name}.parquet"
        if path.exists():
            feats = pd.read_parquet(path)
            if feats["row_id"].tolist() == self.df["row_id"].to_numpy()[rows].tolist():
                feats.index = self.df.index[rows]
                return feats
        t0 = time.time()
        sub = self.df.iloc[rows]
        langs = self.languages().loc[sub["row_id"]].tolist()
        feats = rq_mod.review_features(sub, np.asarray(self.emb[rows]), self.grid.get("umbral_duplicado", 0.95), languages=langs)
        feats.to_parquet(path, index=False)
        logger.info("Rasgos de %s (%d reseñas) en %.1f min", name, len(rows), (time.time() - t0) / 60)
        return feats

    def train_features(self) -> pd.DataFrame:
        if self._feats is None:
            self._feats = self.features(self.train, "train")
        return self._feats

    # -- consultas
    def holdout(self) -> None:
        sample, _ = load_holdout(self.cfg)
        self.split = split_holdout(sample)
        src = ROOT / self.grid["vectores_holdout"]
        self.hold_town = dict(zip(sample["row_id"].astype(int), sample["town"]))
        self.doc, self.hyde = {}, {}
        for part in ("calibracion", "prueba"):
            data = np.load(src / f"holdout_doc_{part}.npz")
            if not set(data["row_id"].tolist()) <= set(self.split[part]):
                raise SystemExit(f"{src}/holdout_doc_{part}.npz no corresponde a la partición del holdout")
            self.doc[part] = (data["row_id"].tolist(), data["vecs"])
            h = np.load(src / f"holdout_hyde_{part}.npz")
            self.hyde[part] = (h["row_id"].tolist(), h["hyde"], h["anti"], json.loads(str(h["providers"])))

    def validation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self.df
        mask = ((df["split"] == "val") & (df["polarity"] >= 4) & ~df["row_id"].isin(self.split["prueba"])).to_numpy()
        true = df.loc[mask, "town"].map(self.pos).to_numpy()
        return np.asarray(self.emb[np.flatnonzero(mask)]), true, balanced_sample(true, self.T, self.cfg.data.seed)

    def gold(self) -> dict[str, Any]:
        """HyDE / Anti-HyDE del gold desde la caché del LLM (nunca llama a un proveedor)."""
        path = self.out / "gold_vectores.npz"
        gold = rp.load_gold(GOLD)
        splits = {q["id"]: q.get("split") for q in json.loads(BANK.read_text(encoding="utf-8"))["queries"]}
        if not path.exists():
            from restmex.llm import build_cache, build_llm_chain
            from restmex.prompts import HYDE_SYSTEM
            from restmex.query_expansion import build_prompt, parse_expansion

            providers = build_llm_chain(self.cfg, None, names=self.cfg.llm.eval_chain)
            cache = build_cache(self.cfg)
            hydes, antis, used = [], [], []
            for q in gold:
                prompt = build_prompt(" ".join(q["query"].split()))
                for p in providers:
                    raw = cache.get(p.name, p.model, HYDE_SYSTEM, prompt)
                    try:
                        h, a = parse_expansion(raw) if raw is not None else (None, None)
                    except ValueError:
                        continue
                    if h:
                        hydes.append(h)
                        antis.append(a)
                        used.append(f"{p.name}:{p.model}")
                        break
                else:
                    raise SystemExit(f"La consulta del gold `{q['id']}` no está en la caché del LLM.")
            vecs = self.encoder.encode_documents(hydes + antis)
            np.savez(path, hyde=vecs[: len(gold)], anti=vecs[len(gold):], llm=np.asarray(used))
        data = np.load(path)
        return {"queries": gold, "split": [splits.get(q["id"]) for q in gold], "hyde": data["hyde"],
                "anti": data["anti"], "llm": pd.Series(data["llm"]).value_counts().to_dict()}

    # -- muestra de 1,500 por pueblo (idea 6)
    def sample_rows(self) -> np.ndarray:
        n = int(self.grid["muestra_por_pueblo"])
        rng = np.random.default_rng(int(self.grid["semilla_muestra"]))
        towns = self.df["town"].to_numpy()[self.train]
        picks = []
        for t in self.towns:
            idx = self.train[towns == t]
            picks.append(np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)))
        return np.sort(np.concatenate(picks))

    def masked_embeddings(self, rows: np.ndarray) -> np.ndarray:
        """Embeddings de las reseñas `rows` con los nombres de pueblos y estados borrados."""
        key = hashlib.sha1(rows.tobytes()).hexdigest()[:12]
        path = self.out / f"emb_sin_nombres_{key}.npy"
        if path.exists():
            return np.load(path)
        sub = self.df.iloc[rows]
        texts = rq_mod.mask_place_names(sub).tolist()
        changed = sum(a != b for a, b in zip(texts, sub["text"]))
        logger.info("Codificando %d reseñas enmascaradas (%d cambiaron)", len(texts), changed)
        t0 = time.time()
        bs = self.cfg.encoder.batch_size
        vecs = np.vstack([self.encoder.encode_documents(texts[i:i + bs]) for i in range(0, len(texts), bs)])
        np.save(path, vecs.astype(np.float32))
        (self.out / f"emb_sin_nombres_{key}.json").write_text(json.dumps(
            {"n": len(rows), "cambiaron": changed, "minutos": round((time.time() - t0) / 60, 1),
             "encoder": self.encoder.name}), encoding="utf-8")
        return vecs


# ---------------------------------------------------------------- K-Means por variante
def _fit(path: str, X: np.ndarray, w: np.ndarray, k: int, n_init: int, seed: int, key: str) -> None:
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X, sample_weight=w)
    tmp = path + ".tmp.npz"
    np.savez(tmp, centroids=l2_normalize(km.cluster_centers_), labels=km.labels_.astype(np.int16), key=key)
    Path(tmp).replace(path)


def build_index(ctx: Context, name: str, rows: np.ndarray, weights: np.ndarray, emb: Optional[np.ndarray] = None,
                seed: Optional[int] = None) -> dict:
    """K-Means (k del YAML) por pueblo con las reseñas `rows` y sus pesos; en disco, reanudable.
    `seed` distinta de `archetypes.seed` guarda en `kmeans/<nombre>__s<semilla>/`."""
    from joblib import Parallel, delayed

    ac = ctx.cfg.archetypes
    seed = ac.seed if seed is None else int(seed)
    d = ctx.out / "kmeans" / (name if seed == ac.seed else f"{name}__s{seed}")
    d.mkdir(parents=True, exist_ok=True)
    towns = ctx.df["town"].to_numpy()[rows]
    jobs = []
    for t in ctx.towns:
        sel = np.flatnonzero(towns == t)
        key = hashlib.sha1(rows[sel].tobytes() + np.round(weights[sel], 6).tobytes() + (b"m" if emb is not None else b"")).hexdigest()
        path = d / f"{t}.npz"
        if path.exists() and str(np.load(path)["key"]) == key:
            continue
        X = np.asarray(emb[sel]) if emb is not None else np.asarray(ctx.emb[rows[sel]])
        jobs.append((len(sel), str(path), X, weights[sel], min(ac.k, len(sel)), key))
    if jobs:
        t0 = time.time()
        jobs.sort(key=lambda j: -j[0])
        Parallel(n_jobs=ctx.jobs)(delayed(_fit)(p, X, w, k, ac.n_init, seed, key) for _, p, X, w, k, key in jobs)
        logger.info("%s (semilla %d): %d K-Means en %.1f min", name, seed, len(jobs), (time.time() - t0) / 60)
    vecs, town_of = [], []
    for t_i, t in enumerate(ctx.towns):
        data = np.load(d / f"{t}.npz")
        vecs.append(data["centroids"])
        town_of += [t_i] * len(data["centroids"])
    return {"vectors": np.vstack(vecs).astype(np.float32), "town_of": np.asarray(town_of)}


def scores(ctx: Context, q: np.ndarray, ix: dict) -> np.ndarray:
    r = ctx.cfg.ranking
    return aggregate_per_town(np.asarray(q, dtype=np.float32) @ ix["vectors"].T, ix["town_of"], ctx.T,
                              r.aggregation, tau=r.lse_tau)


def gold_metrics(ctx: Context, ix: dict, lam: float, gold: dict, keep_split) -> list[dict]:
    pen = popularity_penalty(ctx.n_reviews, ctx.n_max, lam)
    adj = scores(ctx, gold["hyde"], ix) - pen
    anti = scores(ctx, gold["anti"], ix)
    rows = []
    for i, q in enumerate(gold["queries"]):
        if not keep_split(gold["split"][i]):
            continue
        ranking = [ctx.towns[t] for t in rp.compose_ranking(adj[i], anti[i])]
        rows.append({"id": q["id"], **rp.ranking_metrics(ranking, q["relevant"], q.get("excluded", []))})
    return rows


def not_test(split: Optional[str]) -> bool:
    return split != "test"


# ---------------------------------------------------------------- una variante
def rq_config(overrides: Optional[dict]) -> ReviewQualityConfig:
    return ReviewQualityConfig(**(overrides or {}))


def variant_index(ctx: Context, v: dict, seed: Optional[int] = None):
    """Selección de reseñas y K-Means de una variante: (índice, filas, pesos, selección)."""
    rq = rq_config(v.get("review_quality"))
    base_rows = ctx.sample_rows() if v.get("muestra") else ctx.train
    feats = ctx.train_features().loc[ctx.df.index[base_rows]]
    sel = rq_mod.select_reviews(ctx.df.iloc[base_rows], feats, rq)
    rows, weights = base_rows[sel.keep], sel.weights[sel.keep]
    emb = ctx.masked_embeddings(base_rows)[sel.keep] if rq.mask_place_names else None
    return build_index(ctx, v["nombre"], rows, weights, emb, seed), base_rows, rows, weights, sel


def run_variant(ctx: Context, v: dict, val, gold: dict, seed: Optional[int] = None) -> dict:
    ix, base_rows, rows, weights, sel = variant_index(ctx, v, seed)

    val_emb, val_true, bal = val
    s_val = scores(ctx, val_emb, ix)
    sweep = lambda_sweep(s_val, val_true, ctx.n_reviews, ctx.n_max, LAMBDAS, bal)
    lam = choose_lambda(sweep)
    row = next(r for r in sweep if r["lambda"] == lam)
    row0 = next(r for r in sweep if r["lambda"] == 0.0)
    cal_ids, cal_vecs = ctx.doc["calibracion"]
    cal_true = np.array([ctx.pos[ctx.hold_town[r]] for r in cal_ids])
    ranks = positions(scores(ctx, cal_vecs, ix) - popularity_penalty(ctx.n_reviews, ctx.n_max, lam), cal_true)
    g = gold_metrics(ctx, ix, lam, gold, not_test)
    per_town = pd.Series(ctx.df["town"].to_numpy()[rows]).value_counts().reindex(ctx.towns, fill_value=0)
    out = {
        "nombre": v["nombre"], "idea": v.get("idea", ""), "muestra": bool(v.get("muestra")),
        "review_quality": v.get("review_quality") or {},
        "resenas": {"base": int(len(base_rows)), "quedan": int(len(rows)),
                    "por_pueblo_min": int(per_town.min()), "por_pueblo_mediana": float(per_town.median()),
                    "pueblo_min": str(per_town.idxmin()), "pueblos_menos_200": int((per_town < 200).sum()),
                    "devueltas_por_piso": sel.restored, "por_pueblo": per_town.astype(int).to_dict(),
                    "motivos": sel.reasons[sel.reasons != ""].value_counts().to_dict()},
        "peso_medio": round(float(weights.mean()), 4),
        "lambda": lam,
        "calibracion": summary(ranks, cal_true, ctx.T),
        "gold": {"n": len(g), "ndcg_at_5": round(float(np.mean([r["ndcg_at_5"] for r in g])), 4),
                 "mrr": round(float(np.mean([r["mrr"] for r in g])), 4),
                 "hit_at_5": round(float(np.mean([r["hit_at_5"] for r in g])), 4)},
        "val": {"mrr_macro": round(row["macro_mrr"], 4), "spearman": round(row["spearman_balanced"], 4),
                "spearman_lambda0": round(row0["spearman_balanced"], 4)},
        "_ranks": ranks.tolist(), "_cal_true": cal_true.tolist(),
        "_gold_ndcg": [r["ndcg_at_5"] for r in g],
    }
    logger.info("%-28s quedan %6d (mín/pueblo %4d) λ=%-5s calib MRR macro %.4f · gold nDCG@5 %.4f · Spearman %+.3f",
                v["nombre"], len(rows), per_town.min(), lam, out["calibracion"]["mrr_macro"],
                out["gold"]["ndcg_at_5"], out["val"]["spearman"])
    return out


def compare(res: dict, ref: dict) -> dict:
    a, b = 1.0 / np.asarray(res["_ranks"]), 1.0 / np.asarray(ref["_ranks"])
    return {"mrr_macro": paired_bootstrap(a, b, np.asarray(res["_cal_true"]), N_BOOT, BOOT_SEED),
            "gold_ndcg_at_5": paired_bootstrap(res["_gold_ndcg"], ref["_gold_ndcg"], None, N_BOOT, BOOT_SEED)}


# ---------------------------------------------------------------- etapas
def run_sweep(ctx: Context, out_dir: Path) -> None:
    path = out_dir / "barrido_calidad.json"
    if path.exists():
        raise SystemExit(f"{path} ya existe: los resultados no se sobrescriben.")
    ctx.holdout()
    val, gold = ctx.validation(), ctx.gold()
    started = time.time()
    results = [run_variant(ctx, v, val, gold) for v in ctx.grid["variantes"]]
    by = {r["nombre"]: r for r in results}
    for r in results:
        ref = by[REF_SAMPLE] if r["muestra"] else by[REF]
        r["vs_referencia"] = {"referencia": ref["nombre"], **compare(r, ref)}
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_block(ctx, gold, started)
    path.write_text(json.dumps({"meta": meta, "variantes": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    text = render_table(results)
    (out_dir / "barrido_calidad.md").write_text(text, encoding="utf-8")
    print(text)


def meta_block(ctx: Context, gold: dict, started: float) -> dict:
    cal_ids, _ = ctx.doc["calibracion"]
    return {
        "config": os.path.relpath(ctx.cfg.source, ROOT), "commit": git_commit(), "encoder": ctx.encoder.name,
        "k": ctx.cfg.archetypes.k, "agregacion": ctx.cfg.ranking.aggregation, "lse_tau": ctx.cfg.ranking.lse_tau,
        "calibracion_n": len(cal_ids), "holdout": "eval/holdout_ids.json (completo) + eval/holdout_calib_test.json",
        "gold_seleccion": f"{sum(not_test(s) for s in gold['split'])} consultas dev + calibration (sin las test)",
        "gold_llm": gold["llm"], "lambdas": LAMBDAS,
        "popularidad": "n.º de reseñas de train sin filtrar (la penalización y el Spearman no cambian con el filtro)",
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED},
        "minutos": round((time.time() - started) / 60, 1),
    }


def render_table(results: list[dict]) -> str:
    L = ["| Variante | Idea | Reseñas que quedan | Mín. por pueblo | λ | MRR macro calib. | Δ vs ref. [IC95] | "
         "nDCG@5 gold | Δ gold [IC95] | Spearman (λ) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r.get("vs_referencia", {})
        m, g = c.get("mrr_macro"), c.get("gold_ndcg_at_5")
        dm = "—" if not m or r["nombre"] == c.get("referencia") else f"{m['diff']:+.4f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]{' *' if m['significant'] else ''}"
        dg = "—" if not g or r["nombre"] == c.get("referencia") else f"{g['diff']:+.4f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]{' *' if g['significant'] else ''}"
        res = r["resenas"]
        L.append(f"| {r['nombre']} | {r['idea']} | {res['quedan']:,} de {res['base']:,} ({100 * res['quedan'] / res['base']:.1f} %) | "
                 f"{res['por_pueblo_min']} ({res['pueblo_min']}) | {r['lambda']} | {r['calibracion']['mrr_macro']:.4f} | {dm} | "
                 f"{r['gold']['ndcg_at_5']:.4f} | {dg} | {r['val']['spearman']:+.3f} |")
    return "\n".join(L) + "\n"


def seeds(ctx: Context) -> list[int]:
    """Semilla del YAML (la del barrido) y las extra del ruido del K-Means."""
    return [int(ctx.cfg.archetypes.seed)] + [int(s) for s in ctx.grid.get("semillas_extra", [])]


def run_seeds(ctx: Context, out_dir: Path) -> None:
    """Las mismas variantes con otras semillas del K-Means: cuánto cambia el resultado solo por
    la inicialización (con 0.1 % de reseñas menos el MRR macro de calibración se movía ±0.03)."""
    path = out_dir / "barrido_semillas.json"
    if path.exists():
        raise SystemExit(f"{path} ya existe: los resultados no se sobrescriben.")
    base = json.loads((out_dir / "barrido_calidad.json").read_text(encoding="utf-8"))
    ctx.holdout()
    val, gold = ctx.validation(), ctx.gold()
    started = time.time()
    runs = {str(seeds(ctx)[0]): base["variantes"]}
    for s in seeds(ctx)[1:]:
        runs[str(s)] = [run_variant(ctx, v, val, gold, seed=s) for v in ctx.grid["variantes"]]
    pooled = pool_runs(ctx, runs)
    path.write_text(json.dumps({"meta": {**meta_block(ctx, gold, started), "semillas": seeds(ctx)},
                                "agregado": pooled, "por_semilla": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
    text = render_pooled(pooled, seeds(ctx))
    (out_dir / "barrido_semillas.md").write_text(text, encoding="utf-8")
    print(text)


def pool_runs(ctx: Context, runs: dict[str, list[dict]], extra: Optional[dict[str, dict]] = None) -> list[dict]:
    """Por variante: media y desviación entre semillas, y Δ frente a su referencia con el
    recíproco del rango de cada consulta promediado sobre las semillas (bootstrap pareado)."""
    names = [r["nombre"] for r in next(iter(runs.values()))] + ([next(iter(extra.values()))["nombre"]] if extra else [])
    by_seed = {s: {r["nombre"]: r for r in rs} for s, rs in runs.items()}
    if extra:
        for s, r in extra.items():
            by_seed[s][r["nombre"]] = r
    out = []
    for name in names:
        rs = [by_seed[s][name] for s in by_seed]
        ref_name = REF_SAMPLE if rs[0]["muestra"] else REF
        refs = [by_seed[s][ref_name] for s in by_seed]
        rr = np.mean([1.0 / np.asarray(r["_ranks"]) for r in rs], axis=0)
        rr_ref = np.mean([1.0 / np.asarray(r["_ranks"]) for r in refs], axis=0)
        g = np.mean([r["_gold_ndcg"] for r in rs], axis=0)
        g_ref = np.mean([r["_gold_ndcg"] for r in refs], axis=0)
        groups = np.asarray(rs[0]["_cal_true"])
        cal = [r["calibracion"]["mrr_macro"] for r in rs]
        d_cal = [r["calibracion"]["mrr_macro"] - f["calibracion"]["mrr_macro"] for r, f in zip(rs, refs)]
        gold_v = [r["gold"]["ndcg_at_5"] for r in rs]
        spear = [r["val"]["spearman"] for r in rs]
        out.append({
            "nombre": name, "idea": rs[0]["idea"], "muestra": rs[0]["muestra"], "review_quality": rs[0]["review_quality"],
            "referencia": ref_name, "resenas": rs[0]["resenas"], "lambdas": [r["lambda"] for r in rs],
            "calib_mrr_macro": {"media": round(float(np.mean(cal)), 4), "de": round(float(np.std(cal, ddof=1)), 4), "por_semilla": cal},
            "delta_calib_por_semilla": [round(d, 4) for d in d_cal],
            "gold_ndcg_at_5": {"media": round(float(np.mean(gold_v)), 4), "de": round(float(np.std(gold_v, ddof=1)), 4), "por_semilla": gold_v},
            "spearman": {"media": round(float(np.mean(spear)), 4), "de": round(float(np.std(spear, ddof=1)), 4), "por_semilla": spear},
            "vs_referencia": None if name == ref_name else {
                "mrr_macro": paired_bootstrap(rr, rr_ref, groups, N_BOOT, BOOT_SEED),
                "gold_ndcg_at_5": paired_bootstrap(g, g_ref, None, N_BOOT, BOOT_SEED)},
        })
    return out


def render_pooled(pooled: list[dict], seed_list: list[int]) -> str:
    L = [f"Media ± desviación entre {len(seed_list)} semillas del K-Means ({', '.join(map(str, seed_list))}). Δ: recíproco "
         "del rango de cada consulta promediado sobre las semillas y bootstrap pareado (IC95; * = sin el 0).", "",
         "| Variante | Idea | Reseñas que quedan | Mín. por pueblo | λ por semilla | MRR macro calib. | Δ vs ref. [IC95] | "
         "Δ por semilla | nDCG@5 gold (25) | Δ gold [IC95] | Spearman |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pooled:
        c, res = r["vs_referencia"], r["resenas"]

        def fmt(b):
            return f"{b['diff']:+.4f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]{' *' if b['significant'] else ''}"
        L.append(
            f"| {r['nombre']} | {r['idea']} | {res['quedan']:,} de {res['base']:,} ({100 * res['quedan'] / res['base']:.1f} %) | "
            f"{res['por_pueblo_min']} ({res['pueblo_min']}) | {', '.join(f'{x:g}' for x in r['lambdas'])} | "
            f"{r['calib_mrr_macro']['media']:.4f} ± {r['calib_mrr_macro']['de']:.4f} | {fmt(c['mrr_macro']) if c else '—'} | "
            f"{' '.join(f'{d:+.3f}' for d in r['delta_calib_por_semilla']) if c else '—'} | "
            f"{r['gold_ndcg_at_5']['media']:.4f} ± {r['gold_ndcg_at_5']['de']:.4f} | {fmt(c['gold_ndcg_at_5']) if c else '—'} | "
            f"{r['spearman']['media']:+.3f} ± {r['spearman']['de']:.3f} |")
    return "\n".join(L) + "\n"


def best_ideas(pooled: list[dict], n: int = 2) -> list[dict]:
    """La mejor variante de cada idea por MRR macro medio de calibración entre semillas
    (desempate: nDCG@5 medio del gold sin las consultas test); luego las `n` ideas con mayor
    Δ promediado sobre semillas frente a su referencia. Se toman aunque Δ no sea > 0 (se
    pidió combinar las 2 mejores); el reporte dice si alguna es mejora real."""
    best: dict[str, dict] = {}
    key = lambda r: (r["calib_mrr_macro"]["media"], r["gold_ndcg_at_5"]["media"])  # noqa: E731
    for r in pooled:
        if r["vs_referencia"] is None:
            continue
        if r["idea"] not in best or key(r) > key(best[r["idea"]]):
            best[r["idea"]] = r
    gain = lambda r: (r["vs_referencia"]["mrr_macro"]["diff"], r["vs_referencia"]["gold_ndcg_at_5"]["diff"])  # noqa: E731
    return sorted(best.values(), key=gain, reverse=True)[:n]


def run_combine(ctx: Context, out_dir: Path) -> None:
    path = out_dir / "combinacion.json"
    if path.exists():
        raise SystemExit(f"{path} ya existe: la combinación no se repite.")
    data = json.loads((out_dir / "barrido_semillas.json").read_text(encoding="utf-8"))
    chosen = best_ideas(data["agregado"])
    merged: dict = {}
    for r in chosen:
        merged.update(r["review_quality"])
    combo = {"nombre": "combinacion", "idea": " + ".join(r["idea"] for r in chosen), "review_quality": merged,
             "muestra": any(r["muestra"] for r in chosen)}
    if combo["muestra"] and not all(r["muestra"] for r in chosen):
        logger.warning("La combinación incluye el enmascarado: se evalúa sobre la muestra de %s por pueblo",
                       ctx.grid["muestra_por_pueblo"])
    ctx.holdout()
    val, gold = ctx.validation(), ctx.gold()
    extra = {str(s): run_variant(ctx, combo, val, gold, seed=s) for s in seeds(ctx)}
    pooled = pool_runs(ctx, data["por_semilla"], extra)
    by = {r["nombre"]: r for r in pooled}
    res = by["combinacion"]
    rr = lambda name: np.mean([1.0 / np.asarray(({**{r["nombre"]: r for r in data["por_semilla"][s]}, "combinacion": extra[s]})[name]["_ranks"])  # noqa: E731
                               for s in extra], axis=0)
    groups = np.asarray(extra[str(seeds(ctx)[0])]["_cal_true"])
    res["vs_componentes"] = {r["nombre"]: paired_bootstrap(rr("combinacion"), rr(r["nombre"]), groups, N_BOOT, BOOT_SEED)
                             for r in chosen}
    out = {"componentes": [r["nombre"] for r in chosen], "regla": best_ideas.__doc__, "variante": res,
           "por_semilla": extra, "commit": git_commit()}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    text = render_pooled([by[res["referencia"]]] + chosen + [res], seeds(ctx))
    (out_dir / "combinacion.md").write_text(text, encoding="utf-8")
    print(text)


def run_test(ctx: Context, out_dir: Path) -> None:
    """Mitad de prueba, una sola vez: combinación contra su referencia con cada semilla; el
    recíproco del rango (y el nDCG@5 del gold test) se promedia por consulta sobre las semillas
    antes del bootstrap pareado."""
    path = out_dir / "prueba.json"
    if path.exists():
        raise SystemExit(f"{path} ya existe: la mitad de prueba se evalúa una sola vez.")
    comb = json.loads((out_dir / "combinacion.json").read_text(encoding="utf-8"))
    runs = json.loads((out_dir / "barrido_semillas.json").read_text(encoding="utf-8"))["por_semilla"]
    combo_runs = comb["por_semilla"]
    ref_name = REF_SAMPLE if comb["variante"]["muestra"] else REF
    ctx.holdout()
    gold = ctx.gold()
    ids, vecs = ctx.doc["prueba"]
    true = np.array([ctx.pos[ctx.hold_town[r]] for r in ids])
    h_ids, h, a, providers = ctx.hyde["prueba"]
    h_true = np.array([ctx.pos[ctx.hold_town[r]] for r in h_ids])

    def evaluate(v: dict, seed: int) -> dict:
        ix = variant_index(ctx, v, seed)[0]
        pen = popularity_penalty(ctx.n_reviews, ctx.n_max, v["lambda"])
        g = gold_metrics(ctx, ix, v["lambda"], gold, lambda s: s == "test")
        return {"doc": positions(scores(ctx, vecs, ix) - pen, true),
                "hyde": positions(scores(ctx, h, ix) - pen, h_true, scores(ctx, a, ix)),
                "gold": np.array([r["ndcg_at_5"] for r in g])}

    per_seed = {}
    for s in seeds(ctx):
        ref = next(r for r in runs[str(s)] if r["nombre"] == ref_name)
        per_seed[s] = {"ref": evaluate(ref, s), "new": evaluate(combo_runs[str(s)], s)}
    result: dict[str, Any] = {
        "combinacion": comb["variante"]["review_quality"], "componentes": comb["componentes"], "referencia": ref_name,
        "semillas": seeds(ctx), "commit": git_commit(), "llm_hyde": providers,
        "lambda": {"referencia": [next(r for r in runs[str(s)] if r["nombre"] == ref_name)["lambda"] for s in seeds(ctx)],
                   "combinacion": [combo_runs[str(s)]["lambda"] for s in seeds(ctx)]},
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "unidad": "consulta; métrica promediada sobre semillas"},
    }
    for key, label, groups in (("doc", "holdout prueba · reseña enmascarada como documento", true),
                               ("hyde", "holdout prueba · HyDE + Anti-HyDE (pipeline web)", h_true)):
        rr_new = np.mean([1.0 / per_seed[s]["new"][key] for s in per_seed], axis=0)
        rr_ref = np.mean([1.0 / per_seed[s]["ref"][key] for s in per_seed], axis=0)
        h5_new = np.mean([(per_seed[s]["new"][key] <= 5).astype(float) for s in per_seed], axis=0)
        h5_ref = np.mean([(per_seed[s]["ref"][key] <= 5).astype(float) for s in per_seed], axis=0)
        result[label] = {
            "n": int(len(rr_new)),
            "combinacion": {"mrr_macro": with_ci(rr_new, groups), "hit_at_5": with_ci(h5_new, None)},
            "referencia": {"mrr_macro": with_ci(rr_ref, groups), "hit_at_5": with_ci(h5_ref, None)},
            "diferencia": {"mrr_macro": paired_bootstrap(rr_new, rr_ref, groups, N_BOOT, BOOT_SEED),
                           "hit_at_5": paired_bootstrap(h5_new, h5_ref, None, N_BOOT, BOOT_SEED)},
            "delta_mrr_macro_por_semilla": {
                str(s): round(float(summary(per_seed[s]["new"][key], groups, ctx.T)["mrr_macro"]
                                    - summary(per_seed[s]["ref"][key], groups, ctx.T)["mrr_macro"]), 4) for s in per_seed},
        }
    g_new = np.mean([per_seed[s]["new"]["gold"] for s in per_seed], axis=0)
    g_ref = np.mean([per_seed[s]["ref"]["gold"] for s in per_seed], axis=0)
    result["gold test · nDCG@5"] = {
        "n": int(len(g_new)), "combinacion": with_ci(g_new, None), "referencia": with_ci(g_ref, None),
        "diferencia": paired_bootstrap(g_new, g_ref, None, N_BOOT, BOOT_SEED),
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


def run_discarded(ctx: Context, out_dir: Path, n: int = 20) -> None:
    """20 reseñas descartadas al azar por la combinación (o, si no descarta, por la mejor
    variante que descarta) para revisarlas a mano."""
    combo = json.loads((out_dir / "combinacion.json").read_text(encoding="utf-8"))["variante"]
    sweep = json.loads((out_dir / "barrido_calidad.json").read_text(encoding="utf-8"))["variantes"]
    candidates = [combo] + sorted(sweep, key=lambda r: -r["calibracion"]["mrr_macro"])
    v = next(r for r in candidates if r["resenas"]["quedan"] < r["resenas"]["base"])
    base_rows = ctx.sample_rows() if v["muestra"] else ctx.train
    sub = ctx.df.iloc[base_rows]
    feats = ctx.train_features().loc[sub.index]
    sel = rq_mod.select_reviews(sub, feats, rq_config(v["review_quality"]))
    out = np.flatnonzero(~sel.keep)
    pick = np.sort(np.random.default_rng(2025).choice(out, size=min(n, len(out)), replace=False))
    L = [f"# {len(pick)} reseñas descartadas al azar · variante `{v['nombre']}`", "",
         f"Filtro: `{json.dumps(v['review_quality'], ensure_ascii=False)}`. Descartadas: {len(out):,} de {len(sub):,} "
         f"reseñas de train. Semilla 2025. Marca en la última columna si el descarte es correcto.", "",
         "| # | row_id | Pueblo | Tipo | Pol. | Palabras de contenido | Genericidad | Motivo | Reseña | ¿Bien descartada? |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, p in enumerate(pick, 1):
        row, f = sub.iloc[p], feats.iloc[p]
        text = row["text"][:350].replace("|", "/").replace("\n", " ")
        L.append(f"| {i} | {row['row_id']} | {row['town']} | {row['type']} | {row['polarity']:.0f} | {f['n_content']} | "
                 f"{f['genericity']:.2f} | {sel.reasons.iloc[p]} | {text} | |")
    (out_dir / "descartadas_muestra.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# ---------------------------------------------------------------- diagnóstico
def run_diagnostic(ctx: Context, out_dir: Path) -> None:
    all_rows = np.arange(len(ctx.df))
    feats = ctx.features(all_rows, "corpus")
    df = ctx.df
    f = feats.copy()
    f["type"], f["polarity"], f["town"], f["text"] = df["type"], df["polarity"].astype(int), df["town"], df["text"]
    p10_words = float(np.percentile(f["n_content"], 10))
    p25_idf = float(np.percentile(f["mean_idf"], 25))
    p75_cos = float(np.percentile(f["cos_global"], 75))
    flags = {
        f"pocas palabras de contenido (≤ P10 = {p10_words:.0f})": f["n_content"] <= p10_words,
        f"IDF medio bajo (< P25 = {p25_idf:.2f})": f["mean_idf"] < p25_idf,
        f"cerca del centroide global (coseno > P75 = {p75_cos:.3f})": f["cos_global"] > p75_cos,
    }
    f["generica"] = np.logical_and.reduce(list(flags.values()))
    f["generica_2de3"] = np.sum(list(flags.values()), axis=0) >= 2
    f["otro_idioma"] = ~f["language"].isin(["es", "?"])
    noise = {
        "genérica (las tres a la vez)": "generica",
        "otro idioma": "otro_idioma",
        "casi duplicada (tiene gemela > 0.95 en su pueblo)": "has_twin",
        "menciona su pueblo o estado": "mentions_place",
        "Hotel/Restaurant solo de servicio": "service_only",
    }
    rng = np.random.default_rng(2025)

    def pct(mask: pd.Series) -> str:
        return f"{100 * mask.mean():.1f} %"

    def breakdown(cols: dict[str, pd.Series]) -> list[str]:
        head = "| Grupo | Reseñas | " + " | ".join(cols) + " |"
        out = [head, "|---" * (len(cols) + 2) + "|"]
        groups = [("Todas", np.ones(len(f), dtype=bool))]
        groups += [(f"Tipo: {t}", (f["type"] == t).to_numpy()) for t in ("Hotel", "Restaurant", "Attractive")]
        groups += [(f"Polaridad {p}", (f["polarity"] == p).to_numpy()) for p in range(1, 6)]
        for name, m in groups:
            out.append(f"| {name} | {int(m.sum()):,} | " + " | ".join(pct(c[m]) for c in cols.values()) + " |")
        return out

    def examples(mask: pd.Series, extra=None) -> list[str]:
        idx = np.flatnonzero(mask.to_numpy())
        pick = np.sort(rng.choice(idx, size=min(N_EXAMPLES, len(idx)), replace=False)) if len(idx) else []
        out = []
        for i in pick:
            r = f.iloc[i]
            more = extra(i) if extra else ""
            out.append(f"{len(out) + 1}. *{r['town']} · {r['type']} · polaridad {r['polarity']} · {r['n_content']} palabras "
                       f"de contenido · idioma {r['language']}*{more} — {r['text'][:300]}")
        return out

    q = lambda s: ", ".join(f"P{p} = {np.percentile(s, p):.0f}" for p in (5, 25, 50, 75, 95))  # noqa: E731
    bins = [-1, 0, 2, 4, 7, 14, 29, np.inf]
    labels = ["0", "1–2", "3–4", "5–7", "8–14", "15–29", "30+"]
    cut = pd.cut(f["n_content"], bins=bins, labels=labels)
    L = [
        "# Diagnóstico de las reseñas que forman los arquetipos", "",
        f"Corpus completo ya limpio y deduplicado ({len(f):,} reseñas, 40 pueblos; train + val), encoder "
        f"`{ctx.encoder.name}` (embeddings de la caché). Generado por `scripts/sweep_quality.py --etapa diagnostico` "
        f"(commit {git_commit()}). Sin tocar el modelo: es solo una descripción del corpus.", "",
        "**Definiciones.** *Palabras de contenido*: tokens de 3+ letras que no son stopwords; la lista de "
        "`src/restmex/stopwords.py` ya incluye los elogios genéricos (excelente, bonito, recomendado, lugar…). "
        "*IDF medio*: ln((1+N)/(1+df))+1 promediado sobre esas palabras. *Centroide global*: media L2 de todos los "
        "embeddings. *Casi duplicada*: coseno > 0.95 con otra reseña del mismo pueblo. *Menciona*: su propio pueblo o "
        "estado (nombre completo, sin espacios, apodos o token distintivo). *Solo servicio*: Hotel o Restaurant con "
        "algún término de servicio (habitación, mesero, recepción, limpieza, precio…) y ninguno de lugar (pueblo, "
        "centro, calle, playa, vista, ubicación…) ni el nombre. Idioma: lingua (offline) entre es/en/fr/de (con portugués e "
        "italiano casi todo lo detectado era español sin acentos), "
        "\"?\" con menos de 3 palabras o sin certeza.", "",
        "## 1. Longitud en palabras de contenido", "",
        f"Todas: {q(f['n_content'])}; media {f['n_content'].mean():.1f}.", "",
        "| Grupo | " + " | ".join(labels) + " | Mediana |", "|---" * (len(labels) + 2) + "|",
    ]
    groups = [("Todas", np.ones(len(f), dtype=bool))] + [(t, (f["type"] == t).to_numpy()) for t in ("Hotel", "Restaurant", "Attractive")] \
        + [(f"Polaridad {p}", (f["polarity"] == p).to_numpy()) for p in range(1, 6)]
    for name, m in groups:
        dist = cut[m].value_counts(normalize=True).reindex(labels, fill_value=0)
        L.append(f"| {name} | " + " | ".join(f"{100 * v:.1f} %" for v in dist) + f" | {np.median(f['n_content'][m]):.0f} |")
    L += ["", "## 2. Reseñas genéricas", "",
          "Tres criterios y su conjunción (genérica = los tres). La puntuación continua de genericidad que usan los "
          "filtros es la media de los percentiles de (−palabras de contenido, −IDF medio, coseno al centroide).", ""]
    L += breakdown({**{k: v for k, v in flags.items()}, "≥ 2 de 3": f["generica_2de3"], "genérica (3 de 3)": f["generica"]})
    L += ["", "Ejemplos (10 al azar entre las genéricas):", ""] + examples(f["generica"])
    lang_counts = f["language"].value_counts()
    L += ["", "## 3. Otro idioma", "",
          "Idiomas detectados: " + ", ".join(f"{k} {v:,} ({100 * v / len(f):.2f} %)" for k, v in lang_counts.items()) + ".", ""]
    L += breakdown({"otro idioma": f["otro_idioma"], "indeterminado (?)": f["language"] == "?"})
    L += ["", "Ejemplos:", ""] + examples(f["otro_idioma"])

    emb = ctx.emb

    def twin(i: int) -> str:
        town_idx = np.flatnonzero((f["town"] == f["town"].iloc[i]).to_numpy())
        sims = np.asarray(emb[town_idx]) @ np.asarray(emb[i])
        sims[town_idx == i] = -1
        j = town_idx[int(np.argmax(sims))]
        return f" · gemela row_id {df['row_id'].iloc[j]} (coseno {sims.max():.3f}): «{df['text'].iloc[j][:160]}»"

    L += ["", "## 4. Casi duplicados dentro del pueblo (coseno > 0.95)", "",
          f"Con gemela: {pct(f['has_twin'])}; sobrantes si se deja una por grupo: {pct(f['near_dup'])} "
          f"({int(f['near_dup'].sum()):,} reseñas). El deduplicado de `data.py` ya quitó los textos idénticos.", ""]
    L += breakdown({"con gemela": f["has_twin"], "sobrante (se quitaría)": f["near_dup"]})
    L += ["", "Ejemplos (con su gemela más cercana):", ""] + examples(f["near_dup"], twin)
    L += ["", "## 5. Mencionan su pueblo o estado", ""]
    L += breakdown({"menciona": f["mentions_place"]})
    L += ["", "Ejemplos de reseñas que **no** lo mencionan:", ""] + examples(~f["mentions_place"])
    business = f["type"].isin(["Hotel", "Restaurant"])
    L += ["", "## 6. Hotel o Restaurant que solo habla del servicio", "",
          f"{pct(f['service_only'][business])} de las reseñas de Hotel y Restaurant "
          f"({int(f['service_only'].sum()):,} reseñas; {pct(f['service_only'])} del corpus).", ""]
    L += breakdown({"solo servicio": f["service_only"]})
    L += ["", "Ejemplos:", ""] + examples(f["service_only"])
    any_noise = f["generica"] | f["otro_idioma"] | f["near_dup"] | f["service_only"]
    L += ["", "## Resumen", "",
          f"- Alguna categoría de ruido (genérica, otro idioma, sobrante duplicada o solo servicio): {pct(any_noise)}.",
          "- " + "; ".join(f"{k}: {pct(f[c])}" for k, c in noise.items()) + ".",
          "- Las cifras por pueblo están en " f"`{os.path.relpath(out_dir, ROOT)}/diagnostico_por_pueblo.csv`.", ""]
    DIAG.write_text("\n".join(L), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_town = f.groupby("town").agg(
        resenas=("n_content", "size"), palabras_contenido_mediana=("n_content", "median"),
        genericas=("generica", "mean"), otro_idioma=("otro_idioma", "mean"), duplicadas=("near_dup", "mean"),
        mencionan=("mentions_place", "mean"), solo_servicio=("service_only", "mean")).round(4)
    by_town.to_csv(out_dir / "diagnostico_por_pueblo.csv")
    print("\n".join(L))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--etapa", required=True, choices=["diagnostico", "barrido", "semillas", "combinar", "prueba", "descartadas"])
    parser.add_argument("--jobs", type=int, default=5, help="K-Means en paralelo (2 hilos cada uno)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    ctx = Context(cfg, args.jobs)
    out_dir = cfg.reports_dir
    {"diagnostico": run_diagnostic, "barrido": run_sweep, "semillas": run_seeds, "combinar": run_combine,
     "prueba": run_test, "descartadas": run_discarded}[args.etapa](ctx, out_dir)


if __name__ == "__main__":
    main()
