"""Precalienta la caché del LLM con las consultas de ejemplo de la página.

Después de ejecutarlo, esas consultas responden al instante en la web y no
gastan cuota (útil antes de la exposición o si no habrá internet).

Uso:
    python scripts/warmup.py                       # consultas de ejemplo
    python scripts/warmup.py "otra consulta" ...   # consultas adicionales
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restmex.config import load_config  # noqa: E402
from restmex.examples import EXAMPLES  # noqa: E402
from restmex.recommender import Recommender  # noqa: E402
from restmex.retrieval import ArchetypeIndex  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    if not cfg.llm.cache:
        raise SystemExit("La caché está apagada (llm.cache: false en config.yaml).")
    rec = Recommender(cfg, ArchetypeIndex.load(cfg.artifacts_dir))
    for query in [*EXAMPLES, *sys.argv[1:]]:
        out = rec.recommend(query)
        exp = out.expansion
        state = "ya estaba en caché" if exp.cached else f"generada en {exp.seconds:.1f} s"
        if exp.fallback:
            state = "SIN LLM (plantilla; no se guarda en caché)"
        print(f"- {query}\n    {exp.provider} {exp.model} · {state}")
        print("    Top 5: " + ", ".join(t.town_display for t in out.top))
    print(f"Caché en {cfg.llm.cache_dir}/")


if __name__ == "__main__":
    main()
