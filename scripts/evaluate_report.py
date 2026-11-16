"""Reporte de evaluación con el protocolo del equipo (restmex-orbita).

Corre el pipeline completo de la web (HyDE + Anti-HyDE + λ) sobre el gold del equipo y un
holdout enmascarado de validación, y escribe en `<reports_dir>/`:
    evaluation_report.json, evaluation_report.md y rankings_query_bank.json

Uso:
    python scripts/evaluate_report.py --estimate          # solo cuenta las llamadas nuevas al LLM
    python scripts/evaluate_report.py                     # reporte completo
    python scripts/evaluate_report.py --hyde-holdout 80 --skip-robustness     # modo recortado
    python scripts/evaluate_report.py --hyde-holdout 80 --robustness 10 --max-llm-calls 120
    python scripts/evaluate_report.py --publish           # además copia a docs/ y reports/

El LLM es `llm.eval_chain` con la caché en disco: repetir la corrida no gasta cuota y lo que
quedó pendiente por presupuesto (`--max-llm-calls`) se completa en la siguiente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restmex import report as rp  # noqa: E402
from restmex.config import load_config  # noqa: E402
from restmex.data import load_reviews  # noqa: E402
from restmex.llm import build_cache, build_llm_chain  # noqa: E402
from restmex.ollama_client import OllamaClient  # noqa: E402
from restmex.recommender import Recommender  # noqa: E402
from restmex.retrieval import ArchetypeIndex  # noqa: E402

logger = logging.getLogger("evaluate_report")
GOLD = ROOT / "eval/gold_queries.json"
BANK = ROOT / "eval/query_bank.json"
TEAM_COMMIT = "3ecbcfd"            # restmex-orbita: de ahí salen el gold, el banco y el protocolo
HOLDOUT_IDS = ROOT / "eval/holdout_ids.json"
HOLDOUT_SEED = 2025
HOLDOUT_PER_TOWN = 15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    """Hash corto de HEAD, con `+cambios` si hay modificaciones sin commit en src/ o scripts/."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()

    head = git("rev-parse", "--short", "HEAD") or "sin-git"
    return head + ("+cambios" if git("status", "--porcelain", "--", "src", "scripts") else "")


def holdout_key(cfg) -> str:
    """Nombre del conjunto en `eval/holdout_ids.json`: depende de qué reseñas entran al corpus."""
    cap = cfg.data.max_reviews_per_town
    return f"tope_{cap}" if cap else "completo"


def load_holdout(cfg, freeze: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Holdout (de `val`) con los `row_id` fijos de `eval/holdout_ids.json` y términos a enmascarar.

    El archivo guarda un conjunto por tamaño de corpus (`completo`, `tope_1500`, …) porque la
    partición train/val cambia con `data.max_reviews_per_town`. Si un `row_id` ya no está en
    `val` (cambió la semilla, la partición o el deduplicado), se detiene en vez de remuestrear.
    Con `freeze` se crea el conjunto que falte con el muestreo de siempre; nunca se sobrescribe.
    """
    df = load_reviews(cfg)
    try:
        raw = pd.read_csv(cfg.raw_csv, usecols=["Review"], encoding="utf-8")
    except UnicodeDecodeError:
        raw = pd.read_csv(cfg.raw_csv, usecols=["Review"], encoding="latin-1")
    # La elegibilidad usa la longitud de la reseña original (sin título), como en el equipo.
    df["review_len"] = raw["Review"].fillna("").str.len().to_numpy()[df["row_id"].to_numpy()]
    terms = rp.mask_terms(df[["town", "region"]].drop_duplicates().itertuples(index=False, name=None))

    key = holdout_key(cfg)
    frozen = json.loads(HOLDOUT_IDS.read_text(encoding="utf-8")) if HOLDOUT_IDS.exists() else {
        "descripcion": (
            "row_id (fila del CSV de entrenamiento) de las reseñas del holdout enmascarado, en el orden "
            "en que se evalúan. Un conjunto por tamaño de corpus; no se editan ni se regeneran."
        ),
        "seed": HOLDOUT_SEED,
        "per_town": HOLDOUT_PER_TOWN,
        "criterio": "val, polaridad >= 4, reseña original de 80 a 600 caracteres, estratificado por pueblo",
        "conjuntos": {},
    }
    if key not in frozen["conjuntos"]:
        if not freeze:
            raise SystemExit(
                f"eval/holdout_ids.json no tiene el conjunto `{key}`. Créalo una sola vez con "
                "`python scripts/evaluate_report.py --config <yaml> --freeze-holdout`."
            )
        sample = rp.holdout_sample(df[df["split"] == "val"], HOLDOUT_PER_TOWN, HOLDOUT_SEED)
        frozen["conjuntos"][key] = {
            "max_reviews_per_town": cfg.data.max_reviews_per_town,
            "val_fraction": cfg.data.val_fraction,
            "data_seed": cfg.data.seed,
            "n": len(sample),
            "row_ids": [int(r) for r in sample["row_id"]],
        }
        HOLDOUT_IDS.write_text(json.dumps(frozen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        logger.info("Holdout `%s` fijado en %s (%d reseñas)", key, HOLDOUT_IDS.name, len(sample))
    ids = frozen["conjuntos"][key]["row_ids"]
    by_id = df.set_index("row_id")
    missing = [i for i in ids if i not in by_id.index]
    not_val = [i for i in ids if i in by_id.index and by_id.at[i, "split"] != "val"]
    if missing or not_val:
        raise SystemExit(
            f"El holdout fijo `{key}` ya no corresponde a esta configuración: {len(missing)} row_id "
            f"no están en el corpus y {len(not_val)} no están en `val`. No se remuestrea; revisa "
            "data.seed, data.val_fraction, data.max_reviews_per_town y el preprocesamiento."
        )
    return by_id.loc[ids].reset_index(), terms


def print_estimate(est: dict) -> None:
    free = est["groq_free_tier"]
    print("\nLlamadas nuevas al LLM (prompts que no están en la caché):")
    for name, s in est["sections"].items():
        print(f"  {name:32s} {s['new_calls']:4d} nuevas de {s['queries']} consultas")
    print(f"  {'TOTAL':32s} {est['new_calls']:4d} ≈ {est['estimated_tokens']:,} tokens "
          f"(~{rp.TOKENS_PER_CALL} por llamada)")
    print(f"Capa gratuita de Groq: {free['rpd']:,} solicitudes/día, {free['tpd']:,} tokens/día, "
          f"{free['tpm']:,} tokens/min.")
    print(f"  solicitudes/día: {'caben' if est['fits_requests_per_day'] else 'NO caben'} · "
          f"tokens/día: {'caben' if est['fits_tokens_per_day'] else 'NO caben'} "
          f"(≈ {est['days_by_tokens']} día(s) de cuota) · mínimo ≈ {est['minutes_at_tpm']} min por el límite por minuto")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--estimate", action="store_true", help="Solo estima las llamadas nuevas al LLM y termina")
    parser.add_argument("--hyde-holdout", type=int, default=200, help="Reseñas del holdout con HyDE (0 = omitir)")
    parser.add_argument("--hyde-unmasked", type=int, default=0, help="Reseñas sin enmascarar con HyDE (gasta cuota)")
    parser.add_argument("--robustness", type=int, default=None, help="Consultas del gold para robustez (todas por defecto; 0 = omitir)")
    parser.add_argument("--skip-robustness", action="store_true", help="Omite la robustez (equivale a --robustness 0)")
    parser.add_argument("--no-holdout", action="store_true", help="Omite el holdout (no carga el CSV)")
    parser.add_argument("--max-llm-calls", type=int, default=None,
                        help="Tope de consultas nuevas al LLM fuera del gold; el resto queda pendiente")
    parser.add_argument("--llm-gap", type=float, default=9.0,
                        help="Segundos entre llamadas nuevas al LLM (8K tokens/min ≈ 1 llamada cada 9 s)")
    parser.add_argument("--freeze-holdout", action="store_true",
                        help="Fija en eval/holdout_ids.json el holdout de esta configuración (si falta) y termina")
    parser.add_argument("--calibrated-lambda", action="store_true",
                        help="Usa el λ recalibrado en validación (`lambda_chosen` de <reports_dir>/metrics.json, "
                             "de `make evaluate`) en lugar de `ranking.lambda` del YAML")
    parser.add_argument("--publish", action="store_true",
                        help="Copia el reporte a docs/evaluation_report.md, reports/evaluation_report.json y reports/rankings_query_bank.json")
    args = parser.parse_args()
    if args.skip_robustness:
        args.robustness = 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    started = time.perf_counter()
    cfg = load_config(args.config)
    logger.info("Configuración: %s", cfg.source)
    if args.freeze_holdout:
        sample, _ = load_holdout(cfg, freeze=True)
        print(f"Holdout `{holdout_key(cfg)}`: {len(sample)} reseñas, {sample['town'].nunique()} pueblos")
        return
    lambda_source = "ranking.lambda del YAML"
    if args.calibrated_lambda:
        metrics_path = cfg.reports_dir / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"Falta {metrics_path}: corre antes `make evaluate` con esta configuración.")
        cfg.ranking.lam = float(json.loads(metrics_path.read_text(encoding="utf-8"))["lambda_chosen"])
        lambda_source = "lambda_chosen de metrics.json (recalibrado en validación)"
        logger.info("λ recalibrado: %s", cfg.ranking.lam)
    index = ArchetypeIndex.load(cfg.artifacts_dir)
    centroids_path = cfg.artifacts_dir / "town_centroids.npy"
    centroids = np.load(centroids_path) if centroids_path.exists() else None
    client = OllamaClient(cfg.ollama.host, cfg.ollama.keep_alive)
    # Cadena de evaluación (Groq → qwen): no gasta la cuota de Gemini de la web.
    providers = build_llm_chain(cfg, client, names=cfg.llm.eval_chain)
    rec = Recommender(cfg, index, client=client, providers=providers, cache=build_cache(cfg))
    pipe = rp.Pipeline(rec, centroids, max_new_calls=args.max_llm_calls, gap=args.llm_gap)

    gold = rp.load_gold(GOLD)
    gold_file = json.loads(GOLD.read_text(encoding="utf-8"))
    bank = json.loads(BANK.read_text(encoding="utf-8"))["queries"]
    splits = {q["id"]: q.get("split") for q in bank}

    sample, terms = (None, [])
    if not args.no_holdout:
        sample, terms = load_holdout(cfg)
        logger.info("Holdout: %d reseñas de validación, %d términos enmascarados", len(sample), len(terms))

    # Estimación de cuota (siempre se imprime; con --estimate no se llama a nadie).
    masked, original = [], []
    if sample is not None:
        texts = sample["text"].tolist()
        all_masked = [rp.mask_entities(t, terms) for t in texts]
        ok = [i for i, m in enumerate(all_masked) if len(m.split()) >= 2]
        towns = [sample["town"].iloc[i] for i in ok]
        masked = [all_masked[ok[p]] for p in rp.balanced_subset(towns, args.hyde_holdout)]
        original = [texts[ok[p]] for p in rp.balanced_subset(towns, args.hyde_unmasked)]
    plan = rp.plan_queries(gold, bank, masked, original, args.robustness)
    estimate = rp.estimate_calls(pipe, plan)
    print_estimate(estimate)
    if args.estimate:
        return

    logger.info("Gold: %d consultas con el pipeline completo", len(gold))
    gold_result, runs = rp.gold_suite(pipe, gold, splits)
    determinism = rp.determinism_suite(pipe, gold)
    report: dict = {
        "rubric": rp.rubric(gold_result, determinism),
        "gold": gold_result,
        "baselines": rp.baseline_suite(pipe, gold),
        "ablation": rp.ablation_suite(pipe, [(r, q["relevant"], q.get("excluded", [])) for r, q in zip(runs, gold)]),
        "determinism": determinism,
        "volume_bias": rp.volume_bias_suite(pipe, gold_result["queries"]),
    }
    logger.info("Casos adversariales")
    report["adversarial"] = rp.adversarial_suite(pipe)

    logger.info("Banco de consultas: %d", len(bank))
    exported = rp.bank_rankings(pipe, bank)
    if sample is not None:
        report["holdout"] = rp.holdout_suite(
            pipe, sample, terms, hyde_n=args.hyde_holdout, hyde_unmasked_n=args.hyde_unmasked, progress=logger.info
        )
    if args.robustness != 0:
        logger.info("Robustez")
        report["robustness"] = rp.robustness_suite(pipe, gold_result["queries"], args.robustness)

    rep_dir = cfg.reports_dir
    rep_dir.mkdir(parents=True, exist_ok=True)
    # Resultados por consulta del holdout (para el bootstrap pareado entre encoders): archivo aparte.
    per_query = report.get("holdout", {}).pop("per_query", None)
    if per_query is not None:
        (rep_dir / "holdout_per_query.json").write_text(json.dumps(per_query, ensure_ascii=False), encoding="utf-8")
    k = len(index.meta) // index.n_towns
    chain = " → ".join([p.tag for p in providers] + ["plantilla"])
    meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(cfg.source.relative_to(ROOT)) if cfg.source else "valores por defecto",
        "commit": git_commit(),
        "encoder": index.stats.get("encoder"),
        "k": k,
        "n_archetypes": len(index.meta),
        "n_towns": index.n_towns,
        "lambda": cfg.ranking.lam,
        "lambda_source": lambda_source,
        "max_chars": cfg.data.max_chars,
        "max_reviews_per_town": cfg.data.max_reviews_per_town,
        "holdout_set": holdout_key(cfg),
        "llm_chain": chain,
        "temperature": cfg.llm.temperature,
        "llm_used": ", ".join(f"{name} × {n}" for name, n in pipe.providers_used.most_common()) or "—",
        "llm_new_calls": pipe.new_calls,
        "llm_cached": pipe.cached_calls,
        "llm_pending": pipe.pending,
        "team_repo": "https://github.com/DyAndrade-AI/restmex-orbita",
        "team_commit": TEAM_COMMIT,
        "gold_sha256": _sha256(GOLD),
        "bank_sha256": _sha256(BANK),
        "gold_provenance": gold_file.get("provenance", {}),
        "holdout_seed": HOLDOUT_SEED,
        "holdout_per_town": HOLDOUT_PER_TOWN,
    }
    bank_path = rep_dir / "rankings_query_bank.json"
    bank_path.write_text(
        json.dumps(
            {
                "system": "restmex-hyde-arquetipos",
                "description": (
                    "Ranking completo (40 pueblos) por consulta de eval/query_bank.json. 'ranking' es lo que "
                    "expone el sistema: ranking[:5] = Top 5 y ranking[-5:] = Bottom 5 (Anti-HyDE; el peor al "
                    "final). 'ranking_por_score' ordena solo por score ajustado (su bottom es el reverso)."
                ),
                **{key: meta[key] for key in ("generated_at", "commit", "encoder", "k", "lambda", "llm_chain", "bank_sha256")},
                "towns": list(index.towns),
                **exported,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    report["bank_export"] = {
        "path": str(bank_path.relative_to(ROOT)),
        "queries": len(bank),
        "exported": len(exported["rankings"]),
        "rejected": len(exported["rejected"]),
        "pending": len(exported["pending"]),
    }
    report["llm_estimate"] = estimate
    meta["duration_s"] = round(time.perf_counter() - started, 1)
    report = {"meta": meta, **report}

    json_path, md_path = rep_dir / "evaluation_report.json", rep_dir / "evaluation_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = rp.render_markdown(report)
    md_path.write_text(text, encoding="utf-8")
    if args.publish:
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs/evaluation_report.md").write_text(text, encoding="utf-8")
        for path in (json_path, bank_path):
            target = ROOT / "reports" / path.name
            if target.resolve() != path.resolve():
                shutil.copyfile(path, target)
        logger.info("Publicado en docs/evaluation_report.md y reports/")
    print(text)
    logger.info("Reporte en %s (%d llamadas nuevas al LLM, %d pendientes)", rep_dir, pipe.new_calls, pipe.pending)


if __name__ == "__main__":
    main()
