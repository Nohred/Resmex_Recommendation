"""Coordenadas (latitud, longitud en grados decimales, WGS84) de los 40 Pueblos Mágicos.

Son las de la cabecera o el centro del pueblo, cotejadas contra OpenStreetMap (Nominatim)
el 2026-11-16: todas quedan a menos de 2 km del resultado de OSM. Solo las usa la web
para dibujar el mapa; el modelo no las ve.
"""
from __future__ import annotations

from typing import Optional

TOWN_COORDS: dict[str, tuple[float, float]] = {
    "Ajijic": (20.300, -103.262),
    "Atlixco": (18.909, -98.437),
    "Bacalar": (18.677, -88.395),
    "Bernal": (20.741, -99.941),
    "Chiapa_de_Corzo": (16.707, -93.012),
    "Cholula": (19.063, -98.306),
    "Coatepec": (19.452, -96.961),
    "Creel": (27.752, -107.635),
    "Cuatro_Cienegas": (26.986, -102.066),
    "Cuetzalan": (20.018, -97.522),
    "Dolores_Hidalgo": (21.156, -100.932),
    "Huasca_de_Ocampo": (20.203, -98.576),
    "Isla_Mujeres": (21.232, -86.731),
    "Ixtapan_de_la_Sal": (18.843, -99.676),
    "Izamal": (20.931, -89.018),
    "Loreto": (26.012, -111.348),
    "Malinalco": (18.948, -99.495),
    "Mazunte": (15.666, -96.554),
    "Metepec": (19.251, -99.605),
    "Orizaba": (18.851, -97.100),
    "Palenque": (17.509, -91.982),
    "Parras": (25.437, -102.184),
    "Patzcuaro": (19.516, -101.609),
    "Real_de_Catorce": (23.690, -100.886),
    "San_Cristobal_de_las_Casas": (16.737, -92.638),
    "Sayulita": (20.869, -105.441),
    "Tapalpa": (19.945, -103.758),
    "Taxco": (18.556, -99.605),
    "Teotihuacan": (19.689, -98.861),
    "Tepotzotlan": (19.716, -99.223),
    "Tepoztlan": (18.985, -99.100),
    "Tequila": (20.883, -103.837),
    "Tequisquiapan": (20.521, -99.892),
    "Tlaquepaque": (20.641, -103.312),
    "TodosSantos": (23.449, -110.223),
    "Tulum": (20.211, -87.465),
    "Valladolid": (20.690, -88.202),
    "Valle_de_Bravo": (19.193, -100.131),
    "Xilitla": (21.386, -98.990),
    "Zacatlan": (19.933, -97.961),
}


def town_coords(town: str) -> Optional[tuple[float, float]]:
    """(lat, lon) del pueblo o `None` si no está en la tabla (p. ej. índices de prueba)."""
    return TOWN_COORDS.get(town)
