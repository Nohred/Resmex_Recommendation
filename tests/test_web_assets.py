"""Recursos de la web: coordenadas, mapa y funcionamiento sin internet."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from restmex.geo import TOWN_COORDS, town_coords

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src/restmex/app"


def test_coords_cover_40_towns_inside_mexico() -> None:
    assert len(TOWN_COORDS) == 40
    for town, (lat, lon) in TOWN_COORDS.items():
        assert 14.5 < lat < 32.8 and -118.4 < lon < -86.6, town
    assert town_coords("Tulum") == (20.211, -87.465) and town_coords("No existe") is None


def test_coords_match_the_index() -> None:
    stats = ROOT / "artifacts/town_stats.json"
    if not stats.exists():
        pytest.skip("sin índice local (artifacts/ no se versiona)")
    assert set(json.loads(stats.read_text(encoding="utf-8"))["towns"]) == set(TOWN_COORDS)


def test_map_geojson() -> None:
    data = json.loads((APP / "static/data/mexico.geojson").read_text(encoding="utf-8"))
    kinds = [f["properties"]["kind"] for f in data["features"]]
    assert kinds.count("country") == 1 and kinds.count("state") >= 32 and "Natural Earth" in data["source"]


def test_web_works_offline() -> None:
    """Nada de CDN: fuentes y datos se sirven desde static/."""
    for path in [APP / "templates/index.html", APP / "static/app.js", APP / "static/styles.css"]:
        text = path.read_text(encoding="utf-8").replace("http://www.w3.org/2000/svg", "")
        assert not re.search(r"https?://", text), path.name
    css = (APP / "static/styles.css").read_text(encoding="utf-8")
    fonts = re.findall(r'url\("/static/(vendor/fonts/[^"]+)"\)', css)
    assert len(fonts) == 3 and all((APP / "static" / f).exists() for f in fonts)
    assert "prefers-reduced-motion" in css and 'aria-live="polite"' in (APP / "templates/index.html").read_text(encoding="utf-8")
