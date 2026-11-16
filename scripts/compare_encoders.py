"""Compara encoders a partir de los reportes de `make report` de cada experimento.

No evalúa nada: lee de cada `<reports_dir>` los archivos `evaluation_report.json`,
`holdout_per_query.json` y `metrics.json`, y escribe una tabla comparativa con un bootstrap
pareado por consulta contra la configuración de referencia.

Uso:
    python scripts/compare_encoders.py --baseline configs/<ref>.yaml \
        --configs configs/<a>.yaml configs/<b>.yaml --out reports/exp/<nombre>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restmex.bootstrap import macro_mean, paired_bootstrap  # noqa: E402
from restmex.config import load_config  # noqa: E402

DOC = "reseña como documento (sin LLM)"
RAW = "consulta cruda (search_query)"
HYDE = "HyDE + Anti-HyDE (pipeline web)"
N_BOOT, SEED = 1000, 2025
_INDEXED = re.compile(r"Indexado en ([\d.]+) min \(embeddings: ([\d.]+) min")


def load_run(config: str) -> dict:
    cfg = load_config(config)
    rep = cfg.reports_dir
    report = json.loads((rep / "evaluation_report.json").read_text(encoding="utf-8"))
    per_query = json.loads((rep / "holdout_per_query.json").read_text(encoding="utf-8"))
    metrics_path = rep / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    minutes = None
    log = cfg.artifacts_dir / "build.log"
    if log.exists():
        found = _INDEXED.findall(log.read_text(encoding="utf-8", errors="ignore"))
        if found:
            # El log acumula reanudaciones: los minutos de embeddings se suman.
            minutes = {"index": round(sum(float(a) for a, _ in found), 1), "embeddings": round(sum(float(b) for _, b in found), 1)}
    return {"name": Path(config).stem, "cfg": cfg, "report": report, "pq": per_query, "metrics": metrics, "minutes": minutes}


def per_query_values(pq: dict, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row_id, 1/posición, acierto en top 5) de las consultas evaluadas en `variant`."""
    ranks = pq["rank"][variant]
    keep = [i for i, r in enumerate(ranks) if r is not None]
    rank = np.array([ranks[i] for i in keep], dtype=np.float64)
    return np.array([pq["row_id"][i] for i in keep]), 1.0 / rank, (rank <= 5).astype(np.float64)


def summarize(run: dict) -> dict:
    report, pq = run["report"], run["pq"]
    towns = sorted(set(pq["town"]))
    code = {t: i for i, t in enumerate(towns)}
    out: dict = {"holdout": {}}
    for variant in (DOC, RAW, HYDE):
        ids, rr, hit5 = per_query_values(pq, variant)
        if not len(ids):
            continue
        pos = {rid: i for i, rid in enumerate(pq["row_id"])}
        groups = np.array([code[pq["town"][pos[r]]] for r in ids])
        agg = report["holdout"]["variants"][variant]
        out["holdout"][variant] = {
            "n": int(len(ids)),
            "mrr_macro": round(macro_mean(rr, groups, len(towns)), 4),
            "hit_at_5": round(float(hit5.mean()), 4),
            "hit_at_1": agg["metrics"]["hit_at_1"],
            "bottom_error_rate": agg["metrics"]["bottom_error_rate"],
            "spearman_hit5_vs_reviews": agg["spearman_hit5_vs_reviews"],
        }
    gold = report["gold"]
    out["gold"] = {
        "ndcg_at_5": gold["aggregate"]["ndcg_at_5"],
        "mrr": gold["aggregate"]["mrr"],
        "hit_at_5": gold["aggregate"]["hit_at_5"],
        "bottom_error_rate": gold["aggregate"]["bottom_error_rate"],
        "latency_warm_s": gold["latency_s"]["warm"],
        "fallback_queries": gold["fallback_queries"],
    }
    out["spearman_exposure_vs_reviews"] = report["volume_bias"]["spearman_exposure_vs_reviews"]
    out["spearman_balanced_val"] = next(
        (v.get("spearman_balanced") for k, v in run["metrics"].get("variants", {}).items()
         if "arquetipos, λ=" in k), None)
    out["determinism"] = report["determinism"]["identical_rate"]
    meta = report["meta"]
    out["meta"] = {k: meta.get(k) for k in ("encoder", "k", "lambda", "lambda_source", "max_chars",
                                             "max_reviews_per_town", "holdout_set", "commit", "llm_used")}
    out["hyde_providers"] = report["holdout"]["variants"].get(HYDE, {}).get("providers", {})
    out["minutes"] = run["minutes"]
    out["n_train"] = run["metrics"].get("n_train")
    return out


def compare(run: dict, base: dict) -> dict:
    """Bootstrap pareado (candidato − referencia) en las consultas que ambos evaluaron."""
    out = {}
    towns = sorted(set(base["pq"]["town"]))
    code = {t: i for i, t in enumerate(towns)}
    town_of = dict(zip(base["pq"]["row_id"], base["pq"]["town"]))
    for variant in (DOC, HYDE):
        ids_a, rr_a, h_a = per_query_values(run["pq"], variant)
        ids_b, rr_b, h_b = per_query_values(base["pq"], variant)
        common = sorted(set(ids_a.tolist()) & set(ids_b.tolist()))
        if not common:
            continue
        pa, pb = ({r: i for i, r in enumerate(ids)} for ids in (ids_a.tolist(), ids_b.tolist()))
        ia, ib = [pa[r] for r in common], [pb[r] for r in common]
        groups = [code[town_of[r]] for r in common]
        out[variant] = {
            "mrr_macro": paired_bootstrap(rr_a[ia], rr_b[ib], groups, N_BOOT, SEED),
            "hit_at_5": paired_bootstrap(h_a[ia], h_b[ib], None, N_BOOT, SEED),
        }
    return out


def _f(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _delta(b: dict) -> str:
    mark = " *" if b["significant"] else ""
    return f"{b['diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]{mark}"


def render(result: dict) -> str:
    runs, base = result["runs"], result["baseline"]
    first = next(iter(runs.values()))
    n_doc = first["holdout"][DOC]["n"]
    n_hyde = first["holdout"].get(HYDE, {}).get("n", 0)
    L = [
        f"# Comparación de encoders ({result['scope']})",
        "",
        f"Referencia: `{base}`. Todas las filas comparten k, partición, semilla, holdout "
        f"(`{first['meta']['holdout_set']}` de `eval/holdout_ids.json`) y las mismas expansiones HyDE del LLM "
        "(salen de la caché); solo cambian el encoder, sus prefijos, `max_chars` y el λ recalibrado.",
        "",
        "## Métricas",
        "",
        f"Holdout enmascarado: reseña como documento (n = {n_doc}) y HyDE + Anti-HyDE (n = {n_hyde}). "
        "Gold: 40 consultas, relevancia binaria. Latencia: `Recommender.recommend` con el LLM en caché "
        "(embeddings + búsqueda), p50 / p95 en segundos.",
        "",
        "| Experimento | Encoder | max_chars | λ | MRR macro doc | MRR macro HyDE | Hit@5 doc | Hit@5 HyDE | nDCG@5 gold | Error bottom (HyDE / gold) | Spearman sesgo (gold / val) | Latencia p50 / p95 | Embeddings (min) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        doc, hyde, gold = r["holdout"][DOC], r["holdout"].get(HYDE, {}), r["gold"]
        lat = gold["latency_warm_s"]
        L.append(
            f"| {name} | {r['meta']['encoder']} | {r['meta']['max_chars']} | {r['meta']['lambda']} | "
            f"{_f(doc['mrr_macro'])} | {_f(hyde.get('mrr_macro'))} | {_f(doc['hit_at_5'])} | {_f(hyde.get('hit_at_5'))} | "
            f"{_f(gold['ndcg_at_5'])} | {_f(hyde.get('bottom_error_rate'))} / {_f(gold['bottom_error_rate'])} | "
            f"{_f(r['spearman_exposure_vs_reviews'])} / {_f(r['spearman_balanced_val'])} | "
            f"{_f(lat['p50'])} / {_f(lat['p95'])} | {_f((r['minutes'] or {}).get('embeddings'), 1)} |"
        )
    L += [
        "",
        "- **Spearman sesgo**: gold = apariciones en el top 5 del gold contra nº de reseñas del pueblo; "
        "val = frecuencia en el top 5 con consultas balanceadas de validación (`make evaluate`). Cerca de 0 es mejor.",
        "- **Error bottom**: fracción de consultas con un pueblo correcto en el Bottom 5 (azar: "
        "5/40 = 0.125 en el holdout).",
        "",
        f"## Diferencia contra `{base}` (bootstrap pareado, IC95)",
        "",
        "| Experimento | ΔMRR macro doc | ΔHit@5 doc | ΔMRR macro HyDE | ΔHit@5 HyDE |",
        "|---|---|---|---|---|",
    ]
    for name, c in result["vs_baseline"].items():
        cells = [_delta(c[v][m]) if v in c else "—" for v in (DOC, HYDE) for m in ("mrr_macro", "hit_at_5")]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += [
        "",
        "`*` = el IC95 no contiene el 0.",
        "",
        "### Definición del bootstrap",
        "",
        f"- Pareado por consulta: los dos encoders se evalúan con las mismas reseñas del holdout (mismo `row_id`) "
        "y, en HyDE, con el mismo texto generado por el LLM.",
        f"- {N_BOOT:,} remuestreos (semilla {SEED}): en cada uno se sacan n consultas con reemplazo y el mismo "
        "remuestreo se aplica al candidato y a la referencia.",
        "- Estadístico: diferencia candidato − referencia de (a) MRR macro (media de 1/posición por pueblo y "
        "luego media entre pueblos; un pueblo ausente en un remuestreo no cuenta en él) y (b) Hit@5 (media por consulta).",
        "- IC95 por percentiles: percentiles 2.5 y 97.5 de las diferencias. Sin corrección por comparaciones "
        "múltiples.",
        "- Solo mide la incertidumbre por qué consultas cayeron en el holdout; no la del K-Means, la del LLM ni la "
        "de la muestra de reseñas indexadas.",
        "",
        "## Procedencia",
        "",
        "| Experimento | Commit | Holdout | Reseñas train | LLM en HyDE del holdout | Determinismo |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        prov = ", ".join(f"{k} × {v}" for k, v in r["hyde_providers"].items()) or "—"
        L.append(f"| {name} | {r['meta']['commit']} | {r['meta']['holdout_set']} | {r['n_train'] or '—'} | {prov} | {_f(r['determinism'], 2)} |")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="YAML de la referencia")
    parser.add_argument("--configs", nargs="+", required=True, help="YAML de los candidatos")
    parser.add_argument("--out", required=True, help="Directorio de salida (no debe tener ya una comparación)")
    parser.add_argument("--scope", default="muestra", help="Texto del título: p. ej. 'muestra de 1,500 reseñas por pueblo'")
    args = parser.parse_args()

    out = ROOT / args.out
    if (out / "comparacion.json").exists():
        raise SystemExit(f"{out} ya tiene una comparación; usa otro directorio (los resultados no se sobrescriben).")
    base = load_run(args.baseline)
    runs = [base] + [load_run(c) for c in args.configs if Path(c).stem != base["name"]]
    for run in runs[1:]:
        if run["pq"]["row_id"] != base["pq"]["row_id"]:
            raise SystemExit(f"{run['name']} no evaluó el mismo holdout que {base['name']}: no son comparables.")
    result = {
        "scope": args.scope,
        "baseline": base["name"],
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "ci": "percentiles 2.5 y 97.5", "paired_by": "row_id"},
        "runs": {r["name"]: summarize(r) for r in runs},
        "vs_baseline": {r["name"]: compare(r, base) for r in runs[1:]},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparacion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text = render(result)
    (out / "comparacion.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
