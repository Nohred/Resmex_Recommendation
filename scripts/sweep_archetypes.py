"""Barrido de arquetipos: cuántos por pueblo (k) y cómo combinarlos (agregación).

Reutiliza la caché de embeddings del encoder configurado (no codifica reseñas del corpus) y
guarda cada K-Means (pueblo, k) en `<artifacts_dir>/kmeans/`, así que es reanudable.

Protocolo (la rejilla vive en la sección `barrido:` del YAML):
- El holdout enmascarado del reporte (`eval/holdout_ids.json`) se parte a la mitad,
  estratificado por pueblo y con semilla fija, en calibración y prueba
  (`eval/holdout_calib_test.json`; no se regenera).
- Todo se elige en calibración con la variante sin LLM (reseña enmascarada como documento).
- λ se recalibra por combinación con el criterio de `make evaluate` (`choose_lambda`) sobre
  las reseñas de validación con polaridad 4–5, sin las reseñas de la mitad de prueba.
- Las finalistas se confirman con HyDE + Anti-HyDE en una submuestra de calibración.
- La mitad de prueba se evalúa una sola vez (`--etapa prueba` se niega a repetirse).

Uso:
    python scripts/sweep_archetypes.py --config configs/<yaml> --etapa barrido
    python scripts/sweep_archetypes.py --config configs/<yaml> --etapa figura
    python scripts/sweep_archetypes.py --config configs/<yaml> --etapa hyde
    python scripts/sweep_archetypes.py --config configs/<yaml> --etapa prueba
    python scripts/sweep_archetypes.py --config configs/<yaml> --etapa lectura --pueblos Tulum Tequila Tapalpa
"""
from __future__ import annotations

import argparse
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

from evaluate import LAMBDAS  # noqa: E402  (scripts/evaluate.py: la rejilla de λ de `make evaluate`)
from evaluate_report import git_commit, load_holdout  # noqa: E402
from restmex import report as rp  # noqa: E402
from restmex.aggregation import aggregate_per_town, k_log, small_cluster_mask  # noqa: E402
from restmex.bootstrap import macro_mean, paired_bootstrap  # noqa: E402
from restmex.config import load_config, load_raw  # noqa: E402
from restmex.encoders import build_encoder, l2_normalize, locate_embeddings  # noqa: E402
from restmex.evaluate import balanced_sample, choose_lambda, lambda_sweep  # noqa: E402
from restmex.ranking import popularity_penalty  # noqa: E402

logger = logging.getLogger("sweep_archetypes")
SPLIT_FILE = ROOT / "eval/holdout_calib_test.json"
SPLIT_SEED = 2025
BASE = "k10|max"                       # la configuración actual: k = 10 y máximo
AGG_ORDER = ["max", "max_big", "top2", "lse"]   # de la más simple a la menos, para desempates
N_BOOT, BOOT_SEED = 1000, 2025


# ---------------------------------------------------------------- partición del holdout
def split_holdout(sample: pd.DataFrame) -> dict[str, list[int]]:
    """Mitad calibración / mitad prueba, estratificada por pueblo. Se crea una vez y se fija."""
    if SPLIT_FILE.exists():
        saved = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        both = set(saved["calibracion"]) | set(saved["prueba"])
        if both != set(int(r) for r in sample["row_id"]):
            raise SystemExit(f"{SPLIT_FILE.name} no corresponde al holdout de esta configuración; no se regenera.")
        return saved
    rng = np.random.default_rng(SPLIT_SEED)
    calib: list[int] = []
    test: list[int] = []
    for i, (_, group) in enumerate(sample.groupby("town", sort=True)):
        ids = rng.permutation(group["row_id"].to_numpy())
        n_cal = (len(ids) + (i % 2 == 0)) // 2       # 15 por pueblo: 8/7 y 7/8 alternados
        calib += [int(r) for r in ids[:n_cal]]
        test += [int(r) for r in ids[n_cal:]]
    order = {int(r): i for i, r in enumerate(sample["row_id"])}
    saved = {
        "descripcion": (
            "Partición del holdout enmascarado `completo` de eval/holdout_ids.json: la mitad de calibración "
            "sirve para elegir (k, agregación, λ, prompts) y la de prueba se evalúa una sola vez con lo ya "
            "elegido. Estratificada por pueblo; no se edita ni se regenera."
        ),
        "seed": SPLIT_SEED,
        "conjunto": "completo",
        "calibracion": sorted(calib, key=order.get),
        "prueba": sorted(test, key=order.get),
    }
    SPLIT_FILE.write_text(json.dumps(saved, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    logger.info("Partición fijada en %s: %d calibración / %d prueba", SPLIT_FILE.name, len(calib), len(test))
    return saved


# ---------------------------------------------------------------- K-Means con caché
def _fit_kmeans(path: str, X: np.ndarray, k: int, n_init: int, seed: int) -> str:
    """Un K-Means (pueblo, k) con la misma receta que `archetypes.build_archetypes`."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(np.asarray(X))
    tmp = path + ".tmp.npz"
    np.savez(tmp, centroids=l2_normalize(km.cluster_centers_), labels=km.labels_.astype(np.int16))
    Path(tmp).replace(path)
    return path


def _silhouette_curve(X: np.ndarray, ks: list[int], sample: int, seed: int) -> dict[int, float]:
    """Silhouette (coseno) de K-Means en una muestra del pueblo, para cada k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from threadpoolctl import threadpool_limits

    X = np.asarray(X)
    if len(X) > sample:
        X = X[np.sort(np.random.default_rng(seed).choice(len(X), size=sample, replace=False))]
    with threadpool_limits(limits=2):
        dist = np.clip(1.0 - X @ X.T, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        out = {}
        for k in ks:
            if k >= len(X):
                break
            labels = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(X)
            out[k] = float(silhouette_score(dist, labels, metric="precomputed"))
    return out


class Clusterings:
    """K-Means por (pueblo, k) sobre `train`, guardados en disco."""

    def __init__(self, cfg, rows: pd.DataFrame, emb: np.ndarray, jobs: int) -> None:
        self.cfg, self.emb, self.jobs = cfg, emb, jobs
        self.dir = cfg.artifacts_dir / "kmeans"
        self.dir.mkdir(parents=True, exist_ok=True)
        train = (rows["split"] == "train").to_numpy()
        town = rows["town"].to_numpy()
        self.towns = sorted(set(town[train]))
        self.idx = {t: np.flatnonzero(train & (town == t)) for t in self.towns}
        self.n_reviews = np.array([len(self.idx[t]) for t in self.towns], dtype=np.float64)
        self.n_max = int(self.n_reviews.max())

    def path(self, town: str, k: int) -> Path:
        return self.dir / f"{town}__k{k}.npz"

    def ensure(self, pairs) -> None:
        """Ajusta en paralelo los (pueblo, k) que falten, de mayor a menor costo."""
        from joblib import Parallel, delayed

        ac = self.cfg.archetypes
        todo = sorted({(t, k) for t, k in pairs if not self.path(t, k).exists()}, key=lambda tk: -len(self.idx[tk[0]]) * tk[1])
        if todo:
            logger.info("K-Means por ajustar: %d (pueblo, k)", len(todo))
            Parallel(n_jobs=self.jobs, verbose=5)(
                delayed(_fit_kmeans)(str(self.path(t, k)), self.emb[self.idx[t]], min(k, len(self.idx[t])), ac.n_init, ac.seed)
                for t, k in todo
            )

    def silhouette(self, ks: list[int], sample: int) -> dict[str, dict[str, float]]:
        from joblib import Parallel, delayed

        path = self.cfg.artifacts_dir / "silhouette.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        curves = Parallel(n_jobs=self.jobs)(
            delayed(_silhouette_curve)(self.emb[self.idx[t]], ks, sample, self.cfg.archetypes.seed) for t in self.towns
        )
        out = {t: {str(k): round(v, 5) for k, v in c.items()} for t, c in zip(self.towns, curves)}
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        return out

    def index(self, k_by_town: dict[str, int]) -> dict[str, np.ndarray]:
        """Vectores (A, d) de una política de k, con pueblo, tamaño y share de cada arquetipo."""
        self.ensure(k_by_town.items())
        vecs, town_of, sizes, shares = [], [], [], []
        for t_i, t in enumerate(self.towns):
            data = np.load(self.path(t, k_by_town[t]))
            count = np.bincount(data["labels"], minlength=len(data["centroids"]))
            vecs.append(data["centroids"])
            town_of += [t_i] * len(count)
            sizes += count.tolist()
            shares += (count / count.sum()).tolist()
        return {
            "vectors": np.vstack(vecs).astype(np.float32), "town_of": np.asarray(town_of),
            "sizes": np.asarray(sizes), "shares": np.asarray(shares),
        }


def k_policies(grid: dict, cl: Clusterings) -> dict[str, dict[str, int]]:
    """Políticas de k: nombre → k por pueblo."""
    out: dict[str, dict[str, int]] = {f"k{k}": {t: int(k) for t in cl.towns} for k in grid["k_fijo"]}
    lo, hi = grid["k_log"]["k_min"], grid["k_log"]["k_max"]
    for a in grid["k_log"]["a"]:
        out[f"log_a{a:g}"] = {t: k_log(len(cl.idx[t]), a, lo, hi) for t in cl.towns}
    sil = grid.get("silhouette")
    if sil:
        curves = cl.silhouette(list(range(sil["k_min"], sil["k_max"] + 1)), sil["muestra"])
        out["silhouette"] = {t: int(max(c, key=lambda k: (c[k], -int(k)))) for t, c in curves.items()}
    return out


def aggregations(grid: dict) -> list[tuple[str, str, dict]]:
    """(nombre, familia, parámetros) de cada agregación de la rejilla."""
    out: list[tuple[str, str, dict]] = []
    for mode in grid["agregacion"]:
        if mode == "lse":
            out += [(f"lse_t{tau:g}", "lse", {"tau": float(tau)}) for tau in grid["lse_tau"]]
        else:
            out.append((mode, mode, {}))
    return out


def town_scores(q: np.ndarray, ix: dict, n_towns: int, family: str, params: dict, grid: dict) -> np.ndarray:
    keep = None
    if family == "max_big":
        small = grid["max_big"]
        keep = small_cluster_mask(ix["sizes"], ix["shares"], ix["town_of"], small["min_share"], small["min_size"])
    return aggregate_per_town(np.asarray(q, dtype=np.float32) @ ix["vectors"].T, ix["town_of"], n_towns, family, keep=keep, **params)


# ---------------------------------------------------------------- métricas por consulta
def positions(adjusted: np.ndarray, true_town: np.ndarray, sim_anti: Optional[np.ndarray] = None) -> np.ndarray:
    """Posición (1 = mejor) del pueblo correcto, con la regla de orden del reporte del equipo."""
    out = np.empty(len(adjusted), dtype=np.int64)
    for i, row in enumerate(adjusted):
        order = rp.compose_ranking(row, None if sim_anti is None else sim_anti[i])
        out[i] = order.index(int(true_town[i])) + 1
    return out


def summary(ranks: np.ndarray, groups: np.ndarray, n_towns: int) -> dict[str, float]:
    return {
        "mrr_macro": round(macro_mean(1.0 / ranks, groups, n_towns), 4),
        "mrr": round(float(np.mean(1.0 / ranks)), 4),
        "hit_at_1": round(float(np.mean(ranks <= 1)), 4),
        "hit_at_5": round(float(np.mean(ranks <= 5)), 4),
        "bottom_error_rate": round(float(np.mean(ranks > n_towns - 5)), 4),
    }


def with_ci(values: np.ndarray, groups: Optional[np.ndarray]) -> dict:
    """Media (macro si hay `groups`) con IC95 bootstrap por percentiles."""
    b = paired_bootstrap(values, np.zeros(len(values)), groups, N_BOOT, BOOT_SEED)
    return {"value": b["candidate"], "ci95": b["ci95"]}


# ---------------------------------------------------------------- contexto compartido
class Context:
    """Datos comunes a todas las etapas: embeddings en caché, validación y holdout."""

    def __init__(self, cfg, jobs: int) -> None:
        self.cfg = cfg
        self.grid = load_raw(cfg.source)["barrido"]
        base_dir = cfg.embeddings_dir
        state = json.loads((base_dir / "embeddings_state.json").read_text())
        self.encoder = build_encoder(cfg)
        rows = pd.read_csv(base_dir / "embeddings_index.csv")
        if state["encoder"] != self.encoder.name or state["done"] != state["n"] or state["n"] != len(rows):
            raise SystemExit(f"La caché de {base_dir} no es del encoder {self.encoder.name} o está incompleta: no se recodifica aquí.")
        self.emb = np.load(locate_embeddings(base_dir, cfg.root), mmap_mode="r")
        self.rows = rows
        self.cl = Clusterings(cfg, rows, self.emb, jobs)
        self.T = len(self.cl.towns)
        self.pos = {t: i for i, t in enumerate(self.cl.towns)}

        sample, terms = load_holdout(cfg)
        split = split_holdout(sample)
        self.split = split
        masked = [rp.mask_entities(t, terms) for t in sample["text"]]
        self.holdout = {
            int(r): {"town": t, "masked": m} for r, t, m in zip(sample["row_id"], sample["town"], masked) if len(m.split()) >= 2
        }
        self.holdout_order = [int(r) for r in sample["row_id"] if int(r) in self.holdout]

    def part(self, name: str) -> tuple[list[int], np.ndarray]:
        """row_id y pueblo (índice) de `calibracion` o `prueba`, en el orden del holdout."""
        keep = set(self.split[name])
        ids = [r for r in self.holdout_order if r in keep]
        return ids, np.array([self.pos[self.holdout[r]["town"]] for r in ids])

    def doc_vectors(self, name: str) -> np.ndarray:
        """Reseñas enmascaradas de una parte, codificadas como documento (se guardan en disco)."""
        ids, _ = self.part(name)
        path = self.cfg.artifacts_dir / f"holdout_doc_{name}.npz"
        if path.exists():
            data = np.load(path)
            if data["row_id"].tolist() == ids:
                return data["vecs"]
        vecs = self.encoder.encode_documents([self.holdout[r]["masked"] for r in ids])
        np.savez(path, row_id=np.asarray(ids), vecs=vecs)
        return vecs

    def validation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Consultas de `make evaluate` (val, polaridad 4–5) sin las reseñas de la mitad de prueba."""
        rows = self.rows
        mask = ((rows["split"] == "val") & (rows["polarity"] >= 4) & ~rows["row_id"].isin(self.split["prueba"])).to_numpy()
        true_town = rows.loc[mask, "town"].map(self.pos).to_numpy()
        return np.asarray(self.emb[np.flatnonzero(mask)]), true_town, balanced_sample(true_town, self.T, self.cfg.data.seed)

    def penalty(self, lam: float) -> np.ndarray:
        return popularity_penalty(self.cl.n_reviews, self.cl.n_max, lam)


# ---------------------------------------------------------------- etapa 1: barrido
def run_sweep(ctx: Context, out_dir: Path) -> None:
    if (out_dir / "barrido.json").exists():
        raise SystemExit(f"{out_dir}/barrido.json ya existe: los resultados no se sobrescriben.")
    grid, cl, T = ctx.grid, ctx.cl, ctx.T
    started = time.time()
    policies = k_policies(grid, cl)
    cl.ensure(pair for pol in policies.values() for pair in pol.items())   # todos de una vez, en paralelo
    kmeans_min = (time.time() - started) / 60
    logger.info("K-Means listos en %.1f min", kmeans_min)

    val_emb, val_true, bal = ctx.validation()
    cal_ids, cal_true = ctx.part("calibracion")
    cal_vecs = ctx.doc_vectors("calibracion")
    logger.info("Validación para λ: %d consultas (%d balanceadas) · calibración: %d", len(val_true), len(bal), len(cal_ids))

    combos: list[dict[str, Any]] = []
    cal_ranks: dict[str, list[int]] = {}
    for pol_name, pol in policies.items():
        ix = cl.index(pol)
        ks = np.array([pol[t] for t in cl.towns])
        for agg_name, family, params in aggregations(grid):
            if ks.max() == 1 and family != "max":          # con 1 arquetipo todas las agregaciones coinciden
                continue
            s_val = town_scores(val_emb, ix, T, family, params, grid)
            sweep = lambda_sweep(s_val, val_true, cl.n_reviews, cl.n_max, LAMBDAS, bal)
            lam = choose_lambda(sweep)
            row0 = next(r for r in sweep if r["lambda"] == 0.0)
            row = next(r for r in sweep if r["lambda"] == lam)
            s_cal = town_scores(cal_vecs, ix, T, family, params, grid)
            ranks = positions(s_cal - ctx.penalty(lam), cal_true)
            ranks0 = positions(s_cal, cal_true)
            name = f"{pol_name}|{agg_name}"
            cal_ranks[name] = ranks.tolist()
            combos.append({
                "nombre": name, "k": pol_name, "agregacion": agg_name, "familia": family, **params,
                "k_medio": round(float(ks.mean()), 2), "k_min": int(ks.min()), "k_max": int(ks.max()),
                "n_arquetipos": int(len(ix["vectors"])),
                "lambda": lam,
                "val": {
                    "mrr_macro": round(row["macro_mrr"], 4), "mrr_macro_lambda0": round(row0["macro_mrr"], 4),
                    "hit_at_5_macro": round(row["macro_hit@5"], 4),
                    "spearman_lambda0": round(row0["spearman_balanced"], 4), "spearman": round(row["spearman_balanced"], 4),
                },
                "calibracion": summary(ranks, cal_true, T),
                "calibracion_lambda0": summary(ranks0, cal_true, T),
            })
            logger.info("%-22s λ=%-5s val MRR macro %.4f · Spearman λ0 %+.3f → %+.3f · calib MRR macro %.4f Hit@5 %.3f",
                        name, lam, row["macro_mrr"], row0["spearman_balanced"], row["spearman_balanced"],
                        combos[-1]["calibracion"]["mrr_macro"], combos[-1]["calibracion"]["hit_at_5"])

    base_rr = 1.0 / np.asarray(cal_ranks[BASE])
    for c in combos:                                   # diferencia pareada contra k = 10 con máximo
        rr = 1.0 / np.asarray(cal_ranks[c["nombre"]])
        c["calibracion_vs_base"] = paired_bootstrap(rr, base_rr, cal_true, N_BOOT, BOOT_SEED)
    combos.sort(key=lambda c: (-c["calibracion"]["mrr_macro"], c["n_arquetipos"], AGG_ORDER.index(c["familia"])))

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "config": os.path.relpath(ctx.cfg.source, ROOT), "commit": git_commit(), "encoder": ctx.encoder.name,
            "rejilla": grid, "lambdas": LAMBDAS,
            "criterio_lambda": "choose_lambda: menor |Spearman| (consultas balanceadas) entre los λ con MRR macro ≥ 99 % del mejor",
            "consultas_lambda": {"n": int(len(val_true)), "balanceadas": int(len(bal)),
                                 "origen": "val, polaridad 4–5, sin las reseñas de la mitad de prueba"},
            "calibracion": {"n": len(cal_ids), "archivo": str(SPLIT_FILE.relative_to(ROOT)), "seed": SPLIT_SEED,
                            "variante": "reseña enmascarada como documento (sin LLM)"},
            "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "pareado_por": "row_id", "referencia": BASE},
            "kmeans_min": round(kmeans_min, 1), "minutos": round((time.time() - started) / 60, 1),
            "k_por_pueblo": {name: pol for name, pol in policies.items() if not name.startswith("k")},
        },
        "combinaciones": combos,
    }
    (out_dir / "barrido.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "calibracion_por_consulta.json").write_text(
        json.dumps({"row_id": cal_ids, "town": [cl.towns[t] for t in cal_true], "rank": cal_ranks}, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "top10_calibracion.md").write_text(render_top(combos, 10), encoding="utf-8")
    print(render_top(combos, 10))


def render_top(combos: list[dict], n: int) -> str:
    L = [
        f"| # | k | Agregación | Arquetipos | λ | MRR macro calib. | Δ vs {BASE} [IC95] | Hit@5 calib. | MRR macro val | Spearman λ=0 → λ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(combos[:n], 1):
        b, cal, val = c["calibracion_vs_base"], c["calibracion"], c["val"]
        k = c["k"] if c["k_min"] == c["k_max"] else f"{c['k']} ({c['k_min']}–{c['k_max']}, media {c['k_medio']})"
        L.append(
            f"| {i} | {k} | {c['agregacion']} | {c['n_arquetipos']} | {c['lambda']} | {cal['mrr_macro']:.4f} | "
            f"{b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]{' *' if b['significant'] else ''} | "
            f"{cal['hit_at_5']:.3f} | {val['mrr_macro']:.4f} | {val['spearman_lambda0']:+.3f} → {val['spearman']:+.3f} |"
        )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- HyDE (calibración y prueba)
def hyde_vectors(ctx: Context, name: str, n: int, gap: float) -> dict[str, Any]:
    """HyDE y Anti-HyDE (cadena de evaluación + caché) de una submuestra balanceada de una parte."""
    from restmex.llm import build_cache, build_llm_chain
    from restmex.prompts import HYDE_SYSTEM
    from restmex.query_expansion import build_prompt, expand_query

    ids, true = ctx.part(name)
    subset = rp.balanced_subset([ctx.cl.towns[t] for t in true], n)
    path = ctx.cfg.artifacts_dir / f"holdout_hyde_{name}.npz"
    want = [ids[p] for p in subset]
    if path.exists():                                 # ya se generó para esta parte: se reutiliza tal cual
        data = np.load(path)
        if not set(data["row_id"].tolist()) <= set(want):
            raise SystemExit(f"{path} no corresponde a la submuestra pedida (--hyde-n {n}).")
        return {"row_id": data["row_id"].tolist(), "hyde": data["hyde"], "anti": data["anti"],
                "providers": json.loads(str(data["providers"]))}
    providers = build_llm_chain(ctx.cfg, None, names=ctx.cfg.llm.eval_chain)
    cache = build_cache(ctx.cfg)
    done, hydes, antis, used, last = [], [], [], [], 0.0
    for i, rid in enumerate(want, 1):
        query = ctx.holdout[rid]["masked"]
        prompt = build_prompt(query)
        if cache is None or all(cache.get(p.name, p.model, HYDE_SYSTEM, prompt) is None for p in providers):
            time.sleep(max(0.0, gap - (time.time() - last)))      # llamada nueva: respeta los tokens por minuto
        exp = expand_query(query, providers, ctx.cfg.llm, cache, batch=True)
        if not exp.cached:
            last = time.time()
        if exp.fallback:                              # ningún LLM respondió: queda fuera (como en el reporte)
            logger.warning("HyDE pendiente para row_id %d: %s", rid, exp.fallback_reason)
            continue
        done.append(rid)
        hydes.append(exp.hyde)
        antis.append(exp.anti_hyde)
        used.append(rp.llm_tag(exp))
        if i % 20 == 0:
            logger.info("HyDE %s %d/%d (%s)", name, i, len(want), pd.Series(used).value_counts().to_dict())
    vecs = ctx.encoder.encode_documents(hydes + antis)
    providers_used = pd.Series(used).value_counts().to_dict()
    np.savez(path, row_id=np.asarray(done), hyde=vecs[: len(done)], anti=vecs[len(done):], providers=json.dumps(providers_used))
    return {"row_id": done, "hyde": vecs[: len(done)], "anti": vecs[len(done):], "providers": providers_used}


def evaluate_combo(ctx: Context, combo: dict, policies: dict, doc_vecs: np.ndarray, doc_true: np.ndarray,
                   hyde: Optional[dict], ids: list[int]) -> dict[str, np.ndarray]:
    """Posición del pueblo correcto por consulta: variante documento y pipeline web con HyDE."""
    ix = ctx.cl.index(policies[combo["k"]])
    family = combo["familia"]
    params = {"tau": combo["tau"]} if family == "lse" else {}
    pen = ctx.penalty(combo["lambda"])
    out = {"doc": positions(town_scores(doc_vecs, ix, ctx.T, family, params, ctx.grid) - pen, doc_true)}
    if hyde is not None:
        where = {r: i for i, r in enumerate(ids)}
        true = doc_true[[where[r] for r in hyde["row_id"]]]
        s_h = town_scores(hyde["hyde"], ix, ctx.T, family, params, ctx.grid) - pen
        s_a = town_scores(hyde["anti"], ix, ctx.T, family, params, ctx.grid)
        out["hyde"] = positions(s_h, true, s_a)
        out["hyde_true"] = true
    return out


def finalists(combos: list[dict], n: int) -> list[dict]:
    """Las `n` mejores en calibración con (k, familia de agregación) distintos."""
    seen, out = set(), []
    for c in combos:                                   # ya vienen ordenadas por MRR macro de calibración
        key = (c["k"], c["familia"])
        if key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) == n:
            break
    return out


def run_hyde(ctx: Context, out_dir: Path, n: int, n_final: int, gap: float) -> None:
    if (out_dir / "eleccion.json").exists():
        raise SystemExit(f"{out_dir}/eleccion.json ya existe: la elección no se repite.")
    combos = json.loads((out_dir / "barrido.json").read_text(encoding="utf-8"))["combinaciones"]
    by_name = {c["nombre"]: c for c in combos}
    final = finalists(combos, n_final)
    policies = k_policies(ctx.grid, ctx.cl)
    ids, true = ctx.part("calibracion")
    doc = ctx.doc_vectors("calibracion")
    hyde = hyde_vectors(ctx, "calibracion", n, gap)
    names = [c["nombre"] for c in final] + ([BASE] if BASE not in {c["nombre"] for c in final} else [])
    ranks = {name: evaluate_combo(ctx, by_name[name], policies, doc, true, hyde, ids) for name in names}
    rows = {}
    for name in names:
        r = ranks[name]
        rows[name] = {"lambda": by_name[name]["lambda"], "doc": summary(r["doc"], true, ctx.T),
                      "hyde": summary(r["hyde"], r["hyde_true"], ctx.T)}
    leader = final[0]["nombre"]
    chosen, reason = leader, "líder de calibración (reseña como documento); ninguna finalista la supera con HyDE fuera del ruido"
    best_gain = 0.0
    versus = {}
    for name in names:
        if name == leader:
            continue
        b = paired_bootstrap(1.0 / ranks[name]["hyde"], 1.0 / ranks[leader]["hyde"], ranks[leader]["hyde_true"], N_BOOT, BOOT_SEED)
        versus[name] = b
        if name in {c["nombre"] for c in final} and b["significant"] and b["diff"] > best_gain:
            chosen, best_gain = name, b["diff"]
            reason = f"supera a la líder ({leader}) con HyDE en calibración: ΔMRR macro {b['diff']:+.4f}, IC95 {b['ci95']}"
    result = {
        "regla": (
            f"Finalistas = las {n_final} mejores por MRR macro en calibración (reseña como documento) con (k, familia "
            "de agregación) distintos. Se elige la líder salvo que otra finalista la supere en MRR macro con HyDE "
            "(bootstrap pareado, IC95 sin el 0)."
        ),
        "finalistas": [c["nombre"] for c in final], "referencia": BASE,
        "hyde_n": len(hyde["row_id"]), "hyde_solicitadas": n, "llm": hyde["providers"],
        "calibracion": rows, "hyde_vs_lider": versus,
        "elegida": chosen, "motivo": reason, "commit": git_commit(),
    }
    (out_dir / "hyde_calibracion.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "eleccion.json").write_text(
        json.dumps({"elegida": chosen, "motivo": reason, "lambda": by_name[chosen]["lambda"], "referencia": BASE,
                    "lambda_referencia": by_name[BASE]["lambda"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- etapa 3: prueba (una sola vez)
def run_test(ctx: Context, out_dir: Path, n: int, gap: float) -> None:
    if (out_dir / "prueba.json").exists():
        raise SystemExit(f"{out_dir}/prueba.json ya existe: la mitad de prueba se evalúa una sola vez.")
    choice = json.loads((out_dir / "eleccion.json").read_text(encoding="utf-8"))
    by_name = {c["nombre"]: c for c in json.loads((out_dir / "barrido.json").read_text(encoding="utf-8"))["combinaciones"]}
    policies = k_policies(ctx.grid, ctx.cl)
    ids, true = ctx.part("prueba")
    doc = ctx.doc_vectors("prueba")
    hyde = hyde_vectors(ctx, "prueba", n, gap) if n > 0 else None
    names = [choice["elegida"], BASE]
    ranks = {name: evaluate_combo(ctx, by_name[name], policies, doc, true, hyde, ids) for name in dict.fromkeys(names)}
    result: dict[str, Any] = {
        "elegida": choice["elegida"], "referencia": BASE, "commit": git_commit(),
        "lambda": {name: by_name[name]["lambda"] for name in dict.fromkeys(names)},
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "ic": "percentiles 2.5 y 97.5", "pareado_por": "row_id"},
        "variantes": {},
    }
    variants = [("reseña como documento (sin LLM)", "doc", true)]
    if hyde is not None:
        variants.append(("HyDE + Anti-HyDE (pipeline web)", "hyde", ranks[BASE]["hyde_true"]))
        result["llm"] = hyde["providers"]
    for label, key, groups in variants:
        block: dict[str, Any] = {"n": int(len(groups)), "sistemas": {}}
        for name in dict.fromkeys(names):
            r = ranks[name][key]
            block["sistemas"][name] = {
                "mrr_macro": with_ci(1.0 / r, groups), "hit_at_1": with_ci((r <= 1).astype(float), None),
                "hit_at_5": with_ci((r <= 5).astype(float), None),
                "bottom_error_rate": round(float(np.mean(r > ctx.T - 5)), 4),
            }
        if choice["elegida"] != BASE:
            a, b = ranks[choice["elegida"]][key], ranks[BASE][key]
            block["elegida_menos_referencia"] = {
                "mrr_macro": paired_bootstrap(1.0 / a, 1.0 / b, groups, N_BOOT, BOOT_SEED),
                "hit_at_1": paired_bootstrap((a <= 1).astype(float), (b <= 1).astype(float), None, N_BOOT, BOOT_SEED),
                "hit_at_5": paired_bootstrap((a <= 5).astype(float), (b <= 5).astype(float), None, N_BOOT, BOOT_SEED),
            }
        result["variantes"][label] = block
    (out_dir / "prueba.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- lectura a mano de arquetipos
def run_reading(ctx: Context, out_dir: Path, towns: list[str], combo_names: list[str]) -> None:
    """Palabras clave y reseña representativa de los arquetipos de algunos pueblos (Markdown)."""
    from restmex.archetypes import cluster_keywords
    from restmex.data import load_reviews

    by_name = {c["nombre"]: c for c in json.loads((out_dir / "barrido.json").read_text(encoding="utf-8"))["combinaciones"]}
    policies = k_policies(ctx.grid, ctx.cl)
    df = load_reviews(ctx.cfg)
    if df["row_id"].tolist() != ctx.rows["row_id"].tolist():
        raise SystemExit("El corpus no coincide con embeddings_index.csv")
    L = ["# Arquetipos para lectura a mano", ""]
    for name in combo_names:
        pol = policies[by_name[name]["k"]]
        for town in towns:
            idx = ctx.cl.idx[town]
            data = np.load(ctx.cl.path(town, pol[town]))
            labels, cents = data["labels"], data["centroids"]
            texts = df["text"].to_numpy()[idx]
            pol_mean = df["polarity"].to_numpy()[idx]
            kws = cluster_keywords(texts.tolist(), labels, len(cents), ctx.cfg.archetypes.n_keywords)
            X = np.asarray(ctx.emb[idx])
            L += [f"## {town} · {name} · {len(idx):,} reseñas de train · k = {len(cents)}", ""]
            for c in np.argsort(-np.bincount(labels, minlength=len(cents))):
                members = np.flatnonzero(labels == c)
                rep = members[np.argmax(X[members] @ cents[c])]
                L += [
                    f"- **{c}** · {len(members)} reseñas ({100 * len(members) / len(idx):.1f} %) · polaridad media "
                    f"{pol_mean[members].mean():.2f} · {', '.join(kws[c])}",
                    f"  > {texts[rep][:420]}",
                ]
            L.append("")
    path = out_dir / "arquetipos_lectura.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- figura
def run_figure(out_dir: Path) -> None:
    """MRR macro y Spearman contra k: una línea por agregación, un panel por medida."""
    from restmex import plots  # noqa: F401  (aplica el estilo de las figuras del repo)
    import matplotlib.pyplot as plt

    data = json.loads((out_dir / "barrido.json").read_text(encoding="utf-8"))
    combos, meta = data["combinaciones"], data["meta"]
    fixed = [c for c in combos if c["k_min"] == c["k_max"]]
    lse = [c for c in fixed if c["familia"] == "lse"]
    taus = sorted({c["tau"] for c in lse})
    tau = max(taus, key=lambda t: np.mean([c["calibracion"]["mrr_macro"] for c in lse if c["tau"] == t])) if taus else None
    series = [("max", "máximo (actual)", "#2a78d6"), ("top2", "media de los 2 mejores", "#eb6834"),
              ("lse", f"logsumexp, τ = {tau:g}", "#1baf7a"), ("max_big", "máximo sin clústeres pequeños", "#eda100")]
    panels = [
        (lambda c: c["calibracion"]["mrr_macro"], f"MRR macro · calibración (n = {meta['calibracion']['n']})", "MRR macro"),
        (lambda c: c["val"]["mrr_macro"], f"MRR macro · validación (n = {meta['consultas_lambda']['n']:,})", "MRR macro"),
        (lambda c: c["val"]["spearman_lambda0"], "Sesgo de popularidad sin penalización (λ = 0)", "Spearman (Top 5 vs. n.º de reseñas)"),
    ]
    adaptive = [c for c in combos if c["k_min"] != c["k_max"] and c["familia"] == "max"]
    markers = {"log_a1": "s", "log_a2": "D", "log_a3": "^", "silhouette": "v"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    for ax, (value, title, ylabel) in zip(axes, panels):
        for family, label, color in series:
            pts = sorted((c for c in fixed if c["familia"] == family and (family != "lse" or c["tau"] == tau)),
                         key=lambda c: c["k_medio"])
            one = [c for c in fixed if c["k_max"] == 1]          # k = 1: todas las agregaciones coinciden
            pts = (one if family != "max" else []) + pts
            ax.plot([c["k_medio"] for c in pts], [value(c) for c in pts], color=color, lw=2, marker="o", ms=5.5,
                    markeredgecolor=plots.SURFACE, markeredgewidth=1, label=label, zorder=3)
        for c in adaptive:
            ax.scatter([c["k_medio"]], [value(c)], marker=markers.get(c["k"], "X"), s=62, facecolor=plots.SURFACE,
                       edgecolor=plots.INK, linewidth=1.4, zorder=4,
                       label=f"{c['k']} (k {c['k_min']}–{c['k_max']}), máximo")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Arquetipos por pueblo (k; media si es adaptativo)")
        ax.set_xticks(sorted({c["k_max"] for c in fixed}))
    axes[2].axhline(0, color=plots.MUTED, lw=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("¿Cuántos arquetipos por pueblo y cómo combinarlos? λ recalibrado en cada punto (MRR)",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figures" / "mrr_spearman_vs_k.png", dpi=150)
    plt.close(fig)
    logger.info("Figura en %s", out_dir / "figures" / "mrr_spearman_vs_k.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--etapa", required=True, choices=["barrido", "figura", "hyde", "prueba", "lectura"])
    parser.add_argument("--jobs", type=int, default=5, help="K-Means en paralelo (2 hilos cada uno)")
    parser.add_argument("--hyde-n", type=int, default=120, help="Reseñas con HyDE por parte (submuestra balanceada)")
    parser.add_argument("--finalistas", type=int, default=3)
    parser.add_argument("--llm-gap", type=float, default=9.0, help="Segundos entre llamadas nuevas al LLM")
    parser.add_argument("--pueblos", nargs="*", default=[])
    parser.add_argument("--combos", nargs="*", default=[], help="Combinaciones para --etapa lectura (por defecto, la elegida)")
    parser.add_argument("--out", default=None, help="Directorio de salida alterno (pruebas del script)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    cfg.archetypes.label_with_llm = False
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) if args.out else cfg.reports_dir
    if args.etapa == "figura":
        run_figure(out_dir)
        return
    ctx = Context(cfg, args.jobs)
    if args.etapa == "barrido":
        run_sweep(ctx, out_dir)
    elif args.etapa == "hyde":
        run_hyde(ctx, out_dir, args.hyde_n, args.finalistas, args.llm_gap)
    elif args.etapa == "prueba":
        run_test(ctx, out_dir, args.hyde_n, args.llm_gap)
    else:
        combos = args.combos or [json.loads((out_dir / "eleccion.json").read_text(encoding="utf-8"))["elegida"]]
        run_reading(ctx, out_dir, args.pueblos, combos)


if __name__ == "__main__":
    main()
