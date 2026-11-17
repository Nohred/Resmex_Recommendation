"""Genera `static/data/mexico.geojson` a partir de Natural Earth (dominio público).

No forma parte del pipeline: solo se corre si hay que regenerar el mapa de la web.
Fuentes (https://www.naturalearthdata.com, repositorio nvkelso/natural-earth-vector):

- `ne_50m_admin_0_countries.geojson`: contorno de México y países vecinos.
- `ne_10m_admin_1_states_provinces.geojson`: límites de los estados.

Uso:
    python scripts/build_map_assets.py <admin0.geojson> <admin1.geojson>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

OUT = Path(__file__).resolve().parents[1] / "src/restmex/app/static/data/mexico.geojson"
BBOX = (-119.5, 13.0, -84.5, 34.0)          # lon_min, lat_min, lon_max, lat_max (con margen)
NEIGHBORS = {"USA", "GTM", "BLZ", "HND", "SLV", "CUB"}
Point = tuple[float, float]


def simplify(points: list[Point], tol: float) -> list[Point]:
    """Douglas-Peucker iterativo (tolerancia en grados)."""
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = points[a], points[b]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5 or 1e-12
        worst, idx = 0.0, -1
        for i in range(a + 1, b):
            x, y = points[i]
            dist = abs(dy * (x - x1) - dx * (y - y1)) / norm if norm > 1e-9 else ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
            if dist > worst:
                worst, idx = dist, i
        if worst > tol and idx > 0:
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    return [p for p, k in zip(points, keep) if k]


def rings(geometry: dict[str, Any]) -> Iterator[list[Point]]:
    """Anillos exteriores de un Polygon o MultiPolygon (los huecos no se dibujan)."""
    polys = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    for poly in polys:
        yield [(float(x), float(y)) for x, y in poly[0]]


def in_bbox(ring: list[Point]) -> bool:
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return not (max(xs) < BBOX[0] or min(xs) > BBOX[2] or max(ys) < BBOX[1] or min(ys) > BBOX[3])


def feature(kind: str, name: str, geometry: dict[str, Any], tol: float, min_points: int) -> dict[str, Any] | None:
    out = []
    for ring in rings(geometry):
        if not in_bbox(ring):
            continue
        simple = simplify(ring, tol)
        if len(simple) >= min_points:
            out.append([[[round(x, 2), round(y, 2)] for x, y in simple]])
    if not out:
        return None
    return {"type": "Feature", "properties": {"kind": kind, "name": name},
            "geometry": {"type": "MultiPolygon", "coordinates": out}}


def main(admin0: Path, admin1: Path) -> None:
    features: list[dict[str, Any]] = []
    for f in json.loads(admin0.read_text(encoding="utf-8"))["features"]:
        code = f["properties"].get("ADM0_A3")
        if code == "MEX":
            features.append(feature("country", "México", f["geometry"], 0.02, 4))
        elif code in NEIGHBORS:
            features.append(feature("neighbor", f["properties"].get("NAME_ES") or code, f["geometry"], 0.06, 6))
    for f in json.loads(admin1.read_text(encoding="utf-8"))["features"]:
        if f["properties"].get("adm0_a3") == "MEX":
            features.append(feature("state", f["properties"].get("name") or "", f["geometry"], 0.03, 6))
    data = {
        "type": "FeatureCollection",
        "source": "Natural Earth 1:50m (países) y 1:10m (estados), dominio público",
        "features": [f for f in features if f],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    kinds = [f["properties"]["kind"] for f in data["features"]]
    print(f"{OUT} · {OUT.stat().st_size / 1024:.0f} KB · "
          + ", ".join(f"{kinds.count(k)} {k}" for k in ("country", "neighbor", "state")))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
