"""Prompts para el LLM local (qwen2.5:3b): HyDE / Anti-HyDE y etiquetas de arquetipos.

Son prompts cortos y muy directivos, con un solo ejemplo (few-shot) y salida
en JSON, porque el modelo es pequeño.
"""
from __future__ import annotations

import re

HYDE_SYSTEM = (
    "Eres un viajero mexicano que escribe reseñas de destinos turísticos. "
    'Respondes únicamente con un JSON válido con las claves "busca", "opuesto", "hyde" y "anti_hyde".'
)

# Las claves "busca" y "opuesto" obligan al modelo pequeño a planear antes de
# escribir: así el HyDE no olvida partes de la consulta y el Anti-HyDE invierte
# las características del destino (no describe "el mismo lugar pero malo").
HYDE_PROMPT = """Un viajero describe el destino que busca. Responde con un JSON de 4 claves, en este orden:

1. "busca": lista de 3 a 5 características que pide el viajero (usa sus palabras).
2. "opuesto": lista con lo contrario de cada característica. Nombra lo que SÍ hay en ese otro lugar (otro ambiente, otro entorno, otras actividades); no uses "sin" ni "no".
3. "hyde": reseña entusiasta en primera persona, de 6 a 8 oraciones (unas 100 palabras), de un destino que tiene TODAS las características de "busca". Menciona cada una con detalles concretos de paisaje, actividades, comida y gente.
4. "anti_hyde": reseña en primera persona, de 6 a 8 oraciones (unas 100 palabras), que describe un destino que tiene TODAS las características de "opuesto", con detalles concretos. Solo describe ese lugar: no lo compares, no uses las palabras "opuesto" ni "contraste" y no menciones lo que busca el viajero.

Reglas: español de México. Nunca escribas nombres propios de pueblos, ciudades ni estados. No copies frases del ejemplo.

Ejemplo
Viajero: "Quiero un pueblo de montaña con clima frío, cabañas y chocolate caliente"
{{"busca": ["pueblo de montaña", "clima frío", "cabañas", "chocolate caliente"], "opuesto": ["costa plana junto al mar", "calor intenso y húmedo", "hoteles enormes todo incluido", "cocteles con hielo en la alberca"], "hyde": "Nos encantó este pueblito en lo alto de la montaña. Hace frío todo el día y en las mañanas la neblina cubre el bosque de pinos. Nos quedamos en una cabaña de madera con chimenea, muy acogedora y con vista a la sierra. En la plaza venden chocolate caliente espumoso y pan recién horneado, perfecto para el clima. Caminamos por las calles empedradas, bien abrigados, y la gente fue muy amable. Si te gusta el frío, la leña y la tranquilidad de la montaña, este lugar es ideal para un fin de semana.", "anti_hyde": "Este destino es de costa, plano y pegado al mar. El calor es intenso y húmedo desde temprano, se suda todo el día. Nos hospedamos en un hotel enorme todo incluido, con cientos de habitaciones, bufet y varias albercas. La gente pasa el día en camastros tomando cocteles con hielo y cerveza helada junto a la alberca. Hay música, animadores y mucho movimiento. Todo gira alrededor del sol, la arena y el bar de la playa. Es un lugar de calor, traje de baño y hoteles grandes."}}

Viajero: "{query}"
"""

LABEL_SYSTEM = (
    "Clasificas temas de reseñas turísticas. "
    'Respondes únicamente con un JSON válido con la clave "etiqueta".'
)

LABEL_PROMPT = """Pon una etiqueta corta en español (1 a 3 palabras, tipo categoría turística) al tema de un grupo de reseñas.

Ejemplo
Palabras clave: tacos, mariscos, sabor, mesero, platillos, pescado
Reseña: "La comida deliciosa, pedimos tacos de pescado y ceviche. El mesero muy atento."
{{"etiqueta": "Mariscos y tacos"}}

Ejemplo
Palabras clave: pirámide, ruinas, guía, mayas, zona arqueológica, subir
Reseña: "Impresionante zona arqueológica, vale la pena contratar un guía."
{{"etiqueta": "Zona arqueológica"}}

Palabras clave: {keywords}
Reseña: "{review}"
"""

# Plantilla de antónimos para el fallback del Anti-HyDE (sin LLM).
ANTONYM_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"tranquil|relaj|paz|descans|silenci|calma", re.I),
     "muy ruidoso y masivo, lleno de turistas, con fiesta y música a todo volumen toda la noche"),
    (re.compile(r"fiesta|antro|vida nocturna|bares|reventón|party", re.I),
     "muy callado y solitario, sin bares ni vida nocturna, donde todo cierra temprano"),
    (re.compile(r"playa|mar\b|caribe|costa|snorkel|buce", re.I),
     "en plena sierra, lejos del mar, con clima frío, neblina y calles empedradas"),
    (re.compile(r"fr[ií]o|monta[ñn]a|bosque|sierra|caba[ñn]a", re.I),
     "de playa, con calor intenso, humedad y sol todo el día"),
    (re.compile(r"arqueol|ruinas|pir[áa]mide|histori|colonial|cultur|museo|iglesia", re.I),
     "moderno y sin historia, puro hotel grande, plaza comercial y edificios nuevos"),
    (re.compile(r"comida|gastronom|tradicional|t[ií]pic|antojitos|mercado", re.I),
     "con comida de cadenas internacionales, cara y sin sabor local"),
    (re.compile(r"naturaleza|cascada|sender|laguna|cenote|ecotur", re.I),
     "completamente urbano, puro concreto, tráfico y contaminación"),
    (re.compile(r"artesan|textil|alfarer|talavera", re.I),
     "sin artesanos ni mercados, solo tiendas de recuerdos importados"),
    (re.compile(r"barat|econ[óo]mic|accesible", re.I),
     "carísimo, con precios inflados para turistas extranjeros"),
    (re.compile(r"lujo|exclusiv|boutique|spa\b", re.I),
     "muy sencillo, sin servicios ni comodidades"),
]

ANTONYM_DEFAULT = "masivo, ruidoso, caro y sin identidad propia, igual a cualquier zona turística comercial"

ANTONYM_TEMPLATE = (
    "Este destino resultó ser todo lo contrario a lo que yo buscaba. Es un lugar {opuestos}. "
    "El ambiente, el paisaje y las actividades son de otro tipo de viaje por completo."
)
