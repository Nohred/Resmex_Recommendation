"""Stopwords en español (y unas pocas en inglés) para las palabras clave TF-IDF."""
from __future__ import annotations

_ES = """
a al algo algunas algunos allá allí ante antes aquel aquella aquí así aun aunque año años bien cada casi como con
contra cosas cual cuando cualquier de del desde después día días donde dos durante el ella ellas ellos en entre era
eran es esa esas ese eso esos esta estaba estaban estamos están estar este esto estos estuvo fue fueron fuimos gran
ha había han hace hacer hasta hay hizo hoy la las le les lo los luego mas más me mejor menos mi mientras mis mucha
muchas mucho muchos muy nada ni no nos nosotros nuestra nuestro nunca o otra otras otro otros para pero poco por
porque pues que qué quien se sea ser si sí sido siempre sin sobre solo sólo son su sus también tan tanto te tenemos
tener tiene tienen toda todas todo todos tres tu tus un una unas uno unos usted ustedes va vamos van ver vez veces
ya yo él e u les fui vas ir iba puede pueden puedes podemos hacia cuando cómo dónde ahí etc
lugar lugares sitio muy súper super vale pena recomiendo recomendable recomendado recomendamos excelente bueno
buena buenos buenas buen bonito bonita hermoso hermosa lindo linda increíble visita visitar visitamos fuimos
experiencia tiempo además realmente verdad parte forma manera hora horas personas persona gente estuvimos
definitivamente simplemente totalmente bastante demasiado embargo cuenta tal tipo así estás estábamos era dan da
dio ti les nuestro nuestra nuestros nuestras mismo misma hicimos puedo pude pudimos tuvimos tuve tenía tenían
hacen hacía haber sino aún ello algún alguna alguno quienes cuales cuál ahora entonces
"""
_EN = """
the and was were with for this that are is it of to in on at we you our very from but not have had they their
there great good place nice all one out can will just would about which when what also more really get got
"""

SPANISH_STOPWORDS: frozenset[str] = frozenset((_ES + _EN).split())
