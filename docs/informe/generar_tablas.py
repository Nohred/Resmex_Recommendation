"""Genera las tablas del informe (`docs/informe/tablas/*.tex`) a partir de los JSON de `reports/`.

Ninguna cifra del informe se transcribe a mano: este script lee los resultados versionados y
escribe las tablas y las macros (`valores.tex`) que `informe.tex` incluye.

Uso (desde la raíz del repo):
    python docs/informe/generar_tablas.py
    cd docs/informe && latexmk -pdf informe.tex
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "reports" / "exp"
OUT = Path(__file__).resolve().parent / "tablas"

DOC = "reseña como documento (sin LLM)"
RAW = "consulta cruda (search_query)"
HYDE = "HyDE + Anti-HyDE (pipeline web)"

ENCODERS = [  # (experimento, nombre en la tabla, max_chars)
    ("enc_nomic", "nomic-embed-text-v2-moe", 1500),
    ("enc_bge_m3", "bge-m3", 1500),
    ("enc_bge_m3_largo", "bge-m3 (largo)", 9000),
    ("enc_qwen3", "qwen3-embedding 0.6b", 1500),
    ("enc_qwen3_largo", "qwen3-embedding 0.6b (largo)", 9000),
    ("enc_gemma", "embeddinggemma", 1500),
    ("enc_gemma_largo", "embeddinggemma (largo)", 6000),
    ("enc_e5_large", "multilingual-e5-large", 1500),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def f3(x, sign: bool = False) -> str:
    if x is None:
        return "---"
    x = round(x + (1e-9 if x >= 0 else -1e-9), 3)   # 0.6775 se guarda como 0.67749…: redondeo al alza
    return f"{x:+.3f}".replace("-", "$-$") if sign else f"{x:.3f}".replace("-", "$-$")


def ci(d: dict) -> str:
    star = "*" if d.get("significant") else ""
    return f"{f3(d['diff'], True)} [{f3(d['ci95'][0], True)}, {f3(d['ci95'][1], True)}]{star}"


def tex(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("≥", r"$\geq$"), ("λ", r"$\lambda$"), ("→", r"$\to$"), ("“", "``"), ("”", "''")):
        s = s.replace(a, b)
    return s


def table(name: str, spec: str, header: list[str], rows: list[list[str]]) -> None:
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [(" & ".join(r) + r" \\") if r else r"\midrule" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def arch_variant(metrics: dict) -> dict:
    """Variante con arquetipos y el λ elegido (su clave dice '10 arquetipos' aunque k sea otro)."""
    return next(v for k, v in metrics["variants"].items() if "arquetipos, λ" in k)


def encoders() -> None:
    comp = load(EXP / "2026-11-16_comparacion_encoders" / "comparacion.json")
    rows_val, rows_hold = [], []
    for exp, label, max_chars in ENCODERS:
        d = EXP / f"2026-11-16_{exp}"
        m, r = load(d / "metrics.json"), load(d / "evaluation_report.json")
        v, hv, g = arch_variant(m), r["holdout"]["variants"], r["gold"]["aggregate"]
        bold = (lambda s: rf"\textbf{{{s}}}") if exp == "enc_gemma" else (lambda s: s)
        rows_val.append([bold(tex(label)), str(max_chars), f"{m['lambda_chosen']:g}", f3(v["hit@1"]), f3(v["hit@5"]),
                         bold(f3(v["macro_mrr"])), f3(v["spearman_balanced"], True)])
        rows_hold.append([bold(tex(label)), f3(hv[DOC]["metrics"]["mrr"]), f3(hv[RAW]["metrics"]["mrr"]),
                          f3(hv[HYDE]["metrics"]["mrr"]), f3(g["hit_at_1"]), f3(g["precision_at_5_wide"]),
                          f3(g["ndcg_at_5"]), f3(g["bottom_error_rate"])])
    table("encoders_val", "lrrrrrr",
          ["Encoder", r"\texttt{max\_chars}", r"$\lambda$", "Hit@1", "Hit@5", "MRR macro", "Spearman"], rows_val)
    table("encoders_holdout", "lrrrrrrr",
          ["Encoder", "MRR doc.", "MRR cruda", "MRR HyDE", "Hit@1", "P@5 amplias", "nDCG@5", "Err. bottom"], rows_hold)
    # Diferencias pareadas contra nomic: se toman tal cual de comparacion.json.
    diffs = comp.get("diferencias") or comp.get("diffs") or comp.get("vs_baseline") or {}
    names = dict((f"2026-11-16_{e}", l) for e, l, _ in ENCODERS)
    rows = []
    for exp, by_variant in diffs.items():
        doc, hyde = by_variant.get(DOC, {}), by_variant.get(HYDE, {})
        rows.append([tex(names.get(exp, exp)), ci(doc["mrr_macro"]), ci(hyde["mrr_macro"])])
    table("encoders_delta", "lll", ["Encoder", r"$\Delta$ MRR macro, documento ($n=600$)",
                                    r"$\Delta$ MRR macro, HyDE ($n=200$)"], rows)


def archetypes() -> None:
    b = load(EXP / "2026-11-16_arquetipos_k_agg" / "barrido.json")
    by = {c["nombre"]: c for c in b["combinaciones"]}
    rows = []
    for k in (1, 3, 5, 10, 15, 20, 30):
        for agg, label in (("max", "máximo"), ("lse_t0.01", r"logsumexp, $\tau=0.01$")):
            c = by.get(f"k{k}|{agg}")
            if c is None:
                continue
            d = c.get("calibracion_vs_base")
            delta = ci(d) if d and c["nombre"] != "k10|max" else "referencia" if c["nombre"] == "k10|max" else "---"
            rows.append([str(k), label, f"{c['lambda']:g}", f3(c["val"]["mrr_macro"]), f3(c["calibracion"]["mrr_macro"]),
                         delta, f3(c["val"]["spearman_lambda0"], True), f3(c["val"]["spearman"], True)])
    table("arquetipos_k", "rlrrrlrr",
          ["$k$", "Agregación", r"$\lambda$", "MRR val.", "MRR calib.", r"$\Delta$ calib.\ vs.\ $k=10$ máx.\ [IC95]",
           r"Spearman $\lambda=0$", r"Spearman $\lambda$"], rows)
    others = []
    for name, label in (("log_a3|lse_t0.01", r"$k = 3\ln n$ (19--30), logsumexp"), ("log_a2|top2", r"$k = 2\ln n$ (13--21), media de 2"),
                        ("silhouette|max", "silhouette, máximo"), ("k30|top2", r"$k=30$, media de 2"),
                        ("k30|max_big", r"$k=30$, máximo sin clústeres pequeños"), ("k30|lse_t0.1", r"$k=30$, logsumexp $\tau=0.1$")):
        c = by.get(name)
        if c:
            others.append([label, str(c["n_arquetipos"]), f"{c['lambda']:g}", f3(c["val"]["mrr_macro"]), f3(c["calibracion"]["mrr_macro"])])
    table("arquetipos_otras", "lrrrr", ["Variante", "Arquetipos", r"$\lambda$", "MRR val.", "MRR calib."], others)

    p = load(EXP / "2026-11-16_arquetipos_k_agg" / "prueba.json")
    rows = []
    for variant, label in ((DOC, "Reseña como documento"), (HYDE, "HyDE + Anti-HyDE")):
        v = p["variantes"][variant]
        d = v["elegida_menos_referencia"]["mrr_macro"]
        rows.append([label, str(v["n"]), f3(d["candidate"]), f3(d["baseline"]), ci(d)])
    table("arquetipos_prueba", "lrrrl", ["Consulta", "$n$", r"$k=30$ lse", r"$k=10$ máx.", r"$\Delta$ MRR macro [IC95]"], rows)


def quality() -> None:
    p = load(EXP / "2026-11-16_calidad_resenas" / "prueba.json")
    rows = []
    for key, d in p.items():
        if isinstance(d, dict) and "diferencia" in d:
            if "diff" in d["diferencia"]:      # gold: la diferencia viene directa (nDCG@5)
                metric, x = "nDCG@5", d["diferencia"]
            else:
                metric, x = "MRR macro", d["diferencia"]["mrr_macro"]
            rows.append([tex(key.split(" · ")[0] + (" · HyDE" if "HyDE" in key else "")), str(d["n"]), metric,
                         f3(x["baseline"]), f3(x["candidate"]), ci(x)])
    table("calidad_prueba", "lrlrrl", ["Conjunto de prueba", "$n$", "Métrica", "Referencia", "Combinación", r"$\Delta$ [IC95]"], rows)


def final_system() -> dict[str, str]:
    """Tablas del sistema adoptado: `reports/metrics.json` y `reports/evaluation_report.json`."""
    values: dict[str, str] = {}
    base, k30 = load(EXP / "2026-11-16_baseline_nomic_k10" / "metrics.json"), load(EXP / "2026-11-16_arq_k30_lse" / "metrics.json")
    runs = [("nomic, $k=10$, máximo (base anterior)", base), ("nomic, $k=30$, logsumexp", k30)]
    gk = EXP / "2026-11-16_gemma_k30_lse" / "metrics.json"
    if gk.exists():
        runs.append(("embeddinggemma, $k=30$, logsumexp", load(gk)))
    best = max(arch_variant(m)["macro_mrr"] for _, m in runs)
    rows = []
    for label, m in runs:
        v = arch_variant(m)
        nol = next(x for k, x in m["variants"].items() if "arquetipos, sin" in k)
        b = (lambda s: rf"\textbf{{{s}}}") if v["macro_mrr"] == best else (lambda s: s)
        rows.append([b(label), str(m["n_archetypes"]), f"{m['lambda_chosen']:g}", f3(v["hit@1"]), f3(v["hit@5"]), b(f3(v["macro_mrr"])),
                     f3(nol["spearman_balanced"], True), f3(v["spearman_balanced"], True)])
    table("verificacion", "lrrrrrrr", ["Configuración (corpus completo)", "Arquetipos", r"$\lambda$", "Hit@1", "Hit@5", "MRR macro",
                                       r"Spearman $\lambda=0$", r"Spearman $\lambda$"], rows)

    rep_path = ROOT / "reports" / "evaluation_report.json"
    if not rep_path.exists():
        return values
    r, old = load(rep_path), load(EXP / "2026-11-16_baseline_nomic_k10" / "evaluation_report.json")
    meta = r["meta"]
    values.update(encoderfinal=tex(meta["encoder"]), kfinal=str(meta["k"]), lambdafinal=f"{meta['lambda']:g}",
                  narquetipos=str(meta["n_archetypes"]), commitreporte=meta["commit"], llmreporte=tex(meta["llm_used"]))
    rows = []
    for item, prev in zip(r["rubric"], old["rubric"]):
        state = {True: "cumple", False: "no cumple", None: "sin medir"}[item["pass"]]
        rows.append([tex(item["check"]), f3(item["value"]), f"{item['threshold']:g}", state, f3(prev["value"])])
    table("final_rubrica", "p{6.6cm}rrlr", ["Criterio", "Sistema final", "Umbral", "Estado", "Base anterior"], rows)

    g, go, bl = r["gold"]["aggregate"], old["gold"]["aggregate"], r["baselines"]
    keys = [("precision_at_5", "Precision@5"), ("precision_at_5_wide", "Precision@5 (consultas amplias)"), ("capped_recall_at_5", "Recall@5 acotado"),
            ("hit_at_1", "Hit@1"), ("hit_at_5", "Hit@5"), ("mrr", "MRR"), ("ndcg_at_5", "nDCG@5"), ("bottom_error_rate", "Error en el bottom 5")]
    table("final_gold", "lrrrr", ["Métrica (gold, 40 consultas)", "Sistema final", "Base anterior", "Popularidad", "Azar"],
          [[lab, f3(g[k]), f3(go[k]), f3(bl["popularity"][k]), f3(bl["random"][k])] for k, lab in keys])

    hv, ho = r["holdout"]["variants"], old["holdout"]["variants"]
    rows = []
    for variant, label in ((DOC, "Reseña como documento (sin LLM)"), (RAW, "Consulta cruda"), (HYDE, "HyDE + Anti-HyDE (web)")):
        v = hv.get(variant)
        if not v or not v.get("n"):
            continue
        m, o = v["metrics"], ho.get(variant, {})
        om = o.get("metrics") if o.get("n") else None
        rows.append([label, str(v["n"]), f3(m["hit_at_1"]), f3(m["hit_at_5"]), f3(m["mrr"]), f3(m["bottom_error_rate"]),
                     f3(v.get("spearman_hit5_vs_reviews"), True), f3(om["mrr"]) if om else "---"])
    rows.append(["Azar (esperado)", "---", "0.025", "0.125", "---", "0.125", "---", "---"])
    table("final_holdout", "lrrrrrrr", ["Consulta", "$n$", "Hit@1", "Hit@5", "MRR", "Err. bottom", "Spearman", "MRR base ant."], rows)

    rows = [[tex(name), f3(a["precision_at_5"]), f3(a["hit_at_1"]), f3(a["hit_at_5"]), f3(a["mrr"]), f3(a["ndcg_at_5"]), f3(a["bottom_error_rate"])]
            for name, a in r["ablation"].items()]
    table("final_ablacion", "p{6.4cm}rrrrrr", ["Configuración (gold)", "P@5", "Hit@1", "Hit@5", "MRR", "nDCG@5", "Err. bottom"], rows)

    split = r["gold"]["by_split"]
    table("final_particiones", "lrrrrr", ["Partición del gold", "Consultas", "P@5", "Hit@1", "MRR", "nDCG@5"],
          [[k, str(v["queries"]), f3(v["precision_at_5"]), f3(v["hit_at_1"]), f3(v["mrr"]), f3(v["ndcg_at_5"])] for k, v in split.items()])
    table("final_adversarial", "lll", ["Caso", "Esperado", "Resultado"],
          [[tex(c["case"]), tex(c["expected"]), "pasa" if c["pass"] else "falla"] for c in r["adversarial"]["cases"]])
    values.update(adversarial=f3(r["adversarial"]["pass_rate"]), determinismo=f3(r["determinism"]["identical_rate"]),
                  sesgogold=f3(r["volume_bias"]["spearman_exposure_vs_reviews"], True),
                  latenciap=f3(r["gold"]["latency_s"]["warm"]["p95"]))
    rob = r.get("robustness") or {}
    values.update(robustez=f3(rob.get("mean_top5_jaccard")), robustezpares=str(rob.get("pairs", "---")))
    return values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    encoders()
    archetypes()
    quality()
    values = final_system()
    (OUT / "valores.tex").write_text(
        "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in sorted(values.items())), encoding="utf-8")
    print(f"Tablas en {OUT.relative_to(ROOT)}: {', '.join(sorted(p.name for p in OUT.glob('*.tex')))}")


if __name__ == "__main__":
    main()
