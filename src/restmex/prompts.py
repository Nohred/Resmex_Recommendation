"""Prompts para el LLM local (qwen2.5:3b): HyDE / Anti-HyDE y etiquetas de arquetipos.

Son prompts cortos y muy directivos, con un solo ejemplo (few-shot) y salida
en JSON, porque el modelo es pequeño.
"""
from __future__ import annotations

import re

HYDE_SYSTEM = (
    "Eres un viajero mexicano que escribe reseñas de destinos turísticos en un sitio de viajes. "
    'Respondes únicamente con un JSON válido con las claves "busca", "opuesto", "hyde" y "anti_hyde".'
)

# Diseño del prompt:
# - Sin reseña de ejemplo: los modelos (sobre todo los pequeños) la copiaban y el
#   HyDE perdía partes de la consulta. Solo se da una guía de pares de opuestos.
# - "busca" y "opuesto" obligan a planear antes de escribir: el HyDE cubre toda
#   la consulta y el Anti-HyDE invierte las características del destino.
# - El Anti-HyDE lo escribe "otro viajero al que sí le gusta ese lugar": así el
#   texto describe rasgos opuestos del destino y no una mala experiencia (queja
#   de servicio, suciedad), que se parecería a las reseñas negativas de cualquier pueblo.
HYDE_PROMPT = """Un viajero describe el destino que busca en México. Devuelve un JSON con 4 claves, en este orden:

1. "busca": lista de 3 a 5 características del destino que pide el viajero (ambiente, entorno, actividades, comida, clima), con sus propias palabras.
2. "opuesto": lista con el contrario de cada característica, como rasgos de OTRO tipo de destino. Nombra lo que sí hay allá; nunca escribas "sin ..." ni "no hay ...". Guía de opuestos (aplica en ambos sentidos): tranquilo ↔ masivo y fiestero; arqueológico o colonial ↔ moderno y comercial; playa y calor ↔ sierra y frío; naturaleza ↔ ciudad y concreto; comida tradicional ↔ cadenas y comida rápida; económico ↔ lujo exclusivo; artesanías ↔ plazas comerciales.
3. "hyde": reseña entusiasta de 80 a 150 palabras, en primera persona, de un viajero que visitó un destino que tiene TODAS las características de "busca". Menciona cada una con detalles concretos de lo que vio, hizo y comió.
4. "anti_hyde": reseña de 80 a 150 palabras, en primera persona, de OTRO viajero que visitó un destino que tiene TODAS las características de "opuesto" y lo describe con detalles concretos (a él sí le gusta ese tipo de lugar). Sería la pesadilla del primer viajero, pero la reseña solo describe ese lugar: no lo compara con nada ni menciona lo que buscaba el primer viajero.

Reglas:
- Español de México, natural y coloquial, como una reseña real de un sitio de viajes.
- Nunca escribas nombres propios de pueblos, ciudades, estados, hoteles ni atracciones.
- "hyde" y "anti_hyde" no deben compartir ninguna característica del destino.

Viajero: "{query}"
"""

LABEL_SYSTEM = (
    "Clasificas temas de reseñas turísticas de Pueblos Mágicos de México. "
    'Respondes únicamente con un JSON válido con la clave "etiquetas".'
)

LABEL_PROMPT = """Estos son {k} grupos temáticos de reseñas de turistas sobre {town}. Da a cada grupo una etiqueta corta en español: 1 a 3 palabras, tipo categoría turística (por ejemplo "Zona arqueológica", "Mariscos frente al mar", "Hoteles boutique", "Vida nocturna").

Reglas: las etiquetas deben distinguir a los grupos entre sí (no repitas); describe el tema, no la opinión; no incluyas el nombre del pueblo.

{groups}

Devuelve {{"etiquetas": [...]}} con exactamente {k} etiquetas, en el mismo orden de los grupos.
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
