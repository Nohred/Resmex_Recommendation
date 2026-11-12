"""Figuras de la evaluación (matplotlib) para la exposición."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import display_town  # noqa: E402
from .retrieval import ArchetypeIndex  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"      # par categórico validado (seguro para daltonismo)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def _grouped_bars(ax: Any, groups: list[str], series: dict[str, list[float]], colors: list[str]) -> None:
    x = np.arange(len(groups))
    w = 0.36
    for i, ((name, values), color) in enumerate(zip(series.items(), colors)):
        pos = x + (i - (len(series) - 1) / 2) * (w + 0.03)
        bars = ax.bar(pos, values, w, color=color, label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, max(max(v) for v in series.values()) * 1.36)   # espacio para la leyenda
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2, fontsize=10)


def fig_variants(metrics: dict[str, Any], out: Path) -> None:
    """(a) Un centroide por pueblo vs. 10 arquetipos (sin penalización)."""
    v = metrics["variants"]
    cent, arch = v["1 centroide por pueblo, sin penalización"], v["10 arquetipos, sin penalización"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, prefix, title in zip(axes, ("", "macro_"), ("Promedio por consulta (micro)", "Promedio por pueblo (macro)")):
        keys = [prefix + k for k in ("hit@1", "hit@5", "mrr")]
        _grouped_bars(ax, ["Hit@1", "Hit@5", "MRR"],
                      {"1 centroide por pueblo": [cent[k] for k in keys], "10 arquetipos por pueblo": [arch[k] for k in keys]},
                      [BLUE, ORANGE])
        ax.set_title(title)
    fig.suptitle(f"¿El pueblo real de la reseña aparece arriba? ({metrics['n_queries']:,} reseñas de validación)",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "centroide_vs_arquetipos.png", dpi=150)
    plt.close(fig)


def fig_lambda(sweep: list[dict[str, Any]], lam: float, out: Path) -> None:
    """(b) Calibración de λ: exactitud y sesgo, en dos paneles (un eje cada uno)."""
    lams = [r["lambda"] for r in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(lams, [r["macro_mrr"] for r in sweep], color=ORANGE, lw=2, marker="o", ms=5, label="MRR macro (por pueblo)")
    ax.plot(lams, [r["mrr"] for r in sweep], color=BLUE, lw=2, marker="o", ms=5, label="MRR micro (por consulta)")
    ax.set_title("Exactitud según λ")
    ax.set_ylabel("MRR")
    ax.legend(loc="lower left", fontsize=10)
    ax = axes[1]
    ax.plot(lams, [r["spearman_balanced"] for r in sweep], color=BLUE, lw=2, marker="o", ms=5)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_title("Sesgo de popularidad según λ")
    ax.set_ylabel("Spearman (frecuencia en Top 5 vs. n.º de reseñas)")
    for ax in axes:
        ax.axvline(lam, color=INK, lw=1, ls="--")
        ax.text(lam, ax.get_ylim()[1], f" λ elegido = {lam}", va="top", ha="left", fontsize=9.5, color=INK)
        ax.set_xlabel("λ (penalización por popularidad)")
    fig.tight_layout()
    fig.savefig(out / "calibracion_lambda.png", dpi=150)
    plt.close(fig)


def fig_popularity(sweep: list[dict[str, Any]], index: ArchetypeIndex, lam: float, out: Path) -> None:
    """Frecuencia en el Top 5 vs. número de reseñas, antes y después de la penalización."""
    rows = {r["lambda"]: r for r in sweep}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, lam_, color, title in zip(axes, (0.0, lam), (BLUE, ORANGE), ("Sin penalización (λ = 0)", f"Con penalización (λ = {lam})")):
        freq = np.asarray(rows[lam_]["top5_frequency_balanced"])
        ax.scatter(index.n_reviews, freq, s=46, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_xscale("log")
        ax.set_title(f"{title} · Spearman = {rows[lam_]['spearman_balanced']:+.2f}")
        ax.set_xlabel("Reseñas de entrenamiento del pueblo (escala log)")
        for t in np.argsort(-index.n_reviews)[:3]:
            ax.annotate(display_town(index.towns[t]), (index.n_reviews[t], freq[t]), xytext=(-7, 6),
                        textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    axes[0].set_ylabel("Veces que aparece en el Top 5")
    fig.suptitle("Sesgo de popularidad con consultas balanceadas (mismo número por pueblo)",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sesgo_popularidad.png", dpi=150)
    plt.close(fig)


def fig_hyde(metrics: dict[str, Any], out: Path) -> None:
    """(c) Con y sin HyDE en la muestra pequeña."""
    h = metrics.get("hyde_comparison")
    if not h:
        return
    names = list(h["variants"])
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    y = np.arange(len(names))
    vals = [h["variants"][n]["hit@5"] for n in names]
    bars = ax.barh(y, vals, height=0.5, color=BLUE)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18 + 0.02)
    ax.set_xlabel("Hit@5")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Reseñas usadas como consulta, con y sin HyDE (n = {h['n']})")
    fig.tight_layout()
    fig.savefig(out / "hyde.png", dpi=150)
    plt.close(fig)


def make_figures(metrics: dict[str, Any], sweep: list[dict[str, Any]], index: ArchetypeIndex, lam: float, out: Path) -> None:
    """Genera todas las figuras en `out`."""
    out.mkdir(parents=True, exist_ok=True)
    fig_variants(metrics, out)
    fig_lambda(sweep, lam, out)
    fig_popularity(sweep, index, lam, out)
    fig_hyde(metrics, out)
