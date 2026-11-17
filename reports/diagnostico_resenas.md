# Diagnóstico de las reseñas que forman los arquetipos

Corpus completo ya limpio y deduplicado (207,867 reseñas, 40 pueblos; train + val), encoder `ollama:nomic-embed-text-v2-moe` (embeddings de la caché). Generado por `scripts/sweep_quality.py --etapa diagnostico` (commit b83a790+cambios). Sin tocar el modelo: es solo una descripción del corpus.

**Definiciones.** *Palabras de contenido*: tokens de 3+ letras que no son stopwords; la lista de `src/restmex/stopwords.py` ya incluye los elogios genéricos (excelente, bonito, recomendado, lugar…). *IDF medio*: ln((1+N)/(1+df))+1 promediado sobre esas palabras. *Centroide global*: media L2 de todos los embeddings. *Casi duplicada*: coseno > 0.95 con otra reseña del mismo pueblo. *Menciona*: su propio pueblo o estado (nombre completo, sin espacios, apodos o token distintivo). *Solo servicio*: Hotel o Restaurant con algún término de servicio (habitación, mesero, recepción, limpieza, precio…) y ninguno de lugar (pueblo, centro, calle, playa, vista, ubicación…) ni el nombre. Idioma: lingua (offline) entre es/en/fr/de (con portugués e italiano casi todo lo detectado era español sin acentos), "?" con menos de 3 palabras o sin certeza.

## 1. Longitud en palabras de contenido

Todas: P5 = 9, P25 = 14, P50 = 20, P75 = 34, P95 = 63; media 26.8.

| Grupo | 0 | 1–2 | 3–4 | 5–7 | 8–14 | 15–29 | 30+ | Mediana |
|---|---|---|---|---|---|---|---|---|
| Todas | 0.0 % | 0.0 % | 0.1 % | 2.2 % | 24.8 % | 43.4 % | 29.5 % | 20 |
| Hotel | 0.0 % | 0.0 % | 0.0 % | 0.1 % | 4.2 % | 41.7 % | 54.0 % | 32 |
| Restaurant | 0.0 % | 0.0 % | 0.1 % | 3.0 % | 35.0 % | 47.2 % | 14.7 % | 17 |
| Attractive | 0.0 % | 0.0 % | 0.2 % | 2.7 % | 27.3 % | 39.8 % | 29.9 % | 20 |
| Polaridad 1 | 0.0 % | 0.0 % | 0.0 % | 0.6 % | 11.5 % | 43.7 % | 44.1 % | 26 |
| Polaridad 2 | 0.0 % | 0.0 % | 0.0 % | 0.7 % | 12.4 % | 41.5 % | 45.4 % | 26 |
| Polaridad 3 | 0.0 % | 0.0 % | 0.1 % | 1.1 % | 19.3 % | 42.8 % | 36.7 % | 22 |
| Polaridad 4 | 0.0 % | 0.0 % | 0.1 % | 1.6 % | 22.2 % | 43.8 % | 32.4 % | 21 |
| Polaridad 5 | 0.0 % | 0.0 % | 0.1 % | 2.6 % | 27.3 % | 43.4 % | 26.6 % | 19 |

## 2. Reseñas genéricas

Tres criterios y su conjunción (genérica = los tres). La puntuación continua de genericidad que usan los filtros es la media de los percentiles de (−palabras de contenido, −IDF medio, coseno al centroide).

| Grupo | Reseñas | pocas palabras de contenido (≤ P10 = 10) | IDF medio bajo (< P25 = 5.46) | cerca del centroide global (coseno > P75 = 0.783) | ≥ 2 de 3 | genérica (3 de 3) |
|---|---|---|---|---|---|---|
| Todas | 207,867 | 11.2 % | 25.0 % | 25.0 % | 15.8 % | 3.6 % |
| Tipo: Hotel | 51,305 | 0.4 % | 27.4 % | 24.4 % | 10.9 % | 0.1 % |
| Tipo: Restaurant | 86,714 | 16.5 % | 33.2 % | 38.4 % | 25.8 % | 7.5 % |
| Tipo: Attractive | 69,848 | 12.7 % | 13.1 % | 8.6 % | 6.9 % | 1.4 % |
| Polaridad 1 | 5,438 | 3.6 % | 5.0 % | 7.5 % | 2.2 % | 0.4 % |
| Polaridad 2 | 5,492 | 4.4 % | 8.6 % | 12.6 % | 4.2 % | 0.7 % |
| Polaridad 3 | 15,501 | 7.5 % | 17.3 % | 20.0 % | 9.3 % | 1.4 % |
| Polaridad 4 | 44,980 | 9.0 % | 25.3 % | 23.6 % | 14.1 % | 2.6 % |
| Polaridad 5 | 136,456 | 13.0 % | 27.2 % | 27.2 % | 18.0 % | 4.4 % |

Ejemplos (10 al azar entre las genéricas):

1. *Tulum · Attractive · polaridad 5 · 10 palabras de contenido · idioma es* — Lugar increíble! Este lugar es de este mundo y 100% vale la pena el viaje.Me encanta el agua, de un buceo libre, y este lugar no me defraudó.El agua es cristalina y realmente es como estar en otro mundo cuando allí!
2. *Cuetzalan · Restaurant · polaridad 3 · 9 palabras de contenido · idioma es* — Vacaciones. Muy buena vista ,muy buena decoración y bien ubicado..servicio lentooo...bebidas buenas y agradable ambiente
3. *Metepec · Restaurant · polaridad 5 · 9 palabras de contenido · idioma es* — Altamente recomendado. Excelente servicio, todo el personal muy atento y la comida deliciosa además de ser un restaurante Pet friendlyMás
4. *Valle_de_Bravo · Restaurant · polaridad 5 · 10 palabras de contenido · idioma es* — Increíble lugar para pasar unos días en pareja o en familia. Recomendado 100% las actividades, la vegetación, las cabañas y la comida. Muchas gracias por todas sus atenciones. Regresaremos!
5. *Valle_de_Bravo · Restaurant · polaridad 5 · 10 palabras de contenido · idioma es* — Desayuno. La vista del lugar esta exelente, el servicio es un poco lento y el desayuno esta muy bueno. Se la pasa uno super bien desayunando con una vista preciosa
6. *Valladolid · Restaurant · polaridad 5 · 8 palabras de contenido · idioma es* — Increible. Gran restaurante El lugar es muy bonito y la comida muy buena. Nos lo pasamos muy bien. Gracias a Rodrigo que fue muy amable y atento! Recomendamos !
7. *Isla_Mujeres · Restaurant · polaridad 3 · 9 palabras de contenido · idioma es* — Buen desayuno, pero no el mejor. Un lugar altamente recomendado en línea, así que llegamos con altas expectativas. El desayuno estaba ok, pero no más. El cuadro es agradable.
8. *Patzcuaro · Restaurant · polaridad 5 · 8 palabras de contenido · idioma es* — Cena. Muy amables , muy buen servicio, rápida la comida y rica . Recomendado para turistas a patzcuaro ...
9. *Valle_de_Bravo · Restaurant · polaridad 5 · 8 palabras de contenido · idioma es* — Comida. Excelente servicio y atención. Muy buena la comida y el ambiente es muy agradable, tranquilo y acogedor.
10. *Isla_Mujeres · Restaurant · polaridad 5 · 10 palabras de contenido · idioma es* — Súper rico! La comida muy rica con mucho sazón, los antojitos woow, muy buen servicio y muy rápido volvería diario! Muy recomendable !

## 3. Otro idioma

Idiomas detectados: es 207,572 (99.86 %), en 285 (0.14 %), ? 7 (0.00 %), fr 3 (0.00 %).

| Grupo | Reseñas | otro idioma | indeterminado (?) |
|---|---|---|---|
| Todas | 207,867 | 0.1 % | 0.0 % |
| Tipo: Hotel | 51,305 | 0.0 % | 0.0 % |
| Tipo: Restaurant | 86,714 | 0.0 % | 0.0 % |
| Tipo: Attractive | 69,848 | 0.3 % | 0.0 % |
| Polaridad 1 | 5,438 | 0.1 % | 0.0 % |
| Polaridad 2 | 5,492 | 0.2 % | 0.0 % |
| Polaridad 3 | 15,501 | 0.0 % | 0.0 % |
| Polaridad 4 | 44,980 | 0.0 % | 0.0 % |
| Polaridad 5 | 136,456 | 0.2 % | 0.0 % |

Ejemplos:

1. *Isla_Mujeres · Attractive · polaridad 5 · 29 palabras de contenido · idioma en* — Cold beer, strong margaritas. Had so much fun here - we stopped in for a few drinks, and the place was hopping. My husband couldn't help himself and ordered nachos and chicken wings, both of which were perfect. Very spicy, which we loved. Great atmosphere, very friendly staff. A great spot to spend 
2. *Isla_Mujeres · Attractive · polaridad 5 · 47 palabras de contenido · idioma en* — Best Spa Experience Ever! This was the second time my daughter and I visited Miile Spa. The spa is located in a separate building from the resort hotel and the atmosphere is so luxurious and relaxing. We booked the hydrotherapy massage, a hot stone massage and a facial. For three hours we were total
3. *Isla_Mujeres · Attractive · polaridad 5 · 22 palabras de contenido · idioma en* — Great little place. You have to go if you like cold beer, great people, and great food. Met a couple of locals and a few tourists there. They're all there for the same reason you are. Relaxation and fun. Can't wait for August 2015 for another week long stay on the island.
4. *Tulum · Attractive · polaridad 5 · 13 palabras de contenido · idioma en* — Gran Cenote! Beautiful cenote and wish we could have stayed longer! It's a must for anyone visiting Tulum. Thanks for the memories!!!
5. *Isla_Mujeres · Attractive · polaridad 5 · 26 palabras de contenido · idioma en* — MUST visit. We loved finding the soggy peso and visiting with a large group. I almost expected Jimmy Buffet to be playing. We dove in the water, saw iguanas and the cutest dog that lives here that has a coconut as a toy! A must do!
6. *Isla_Mujeres · Hotel · polaridad 5 · 79 palabras de contenido · idioma en* — Beautiful Island and even more beautiful people. Mia Reef Isla Mujeres and all it's staff honestly provided one of the most welcoming stays I've ever had. From the first day, the service we received from Roberto from day 1 was amazing, the attention to detail and treatment was exceptional. Miguel de
7. *Loreto · Attractive · polaridad 5 · 67 palabras de contenido · idioma en* — Whales, whales and more whales. Sea Kayak Adventures provided a wonderful whale watching experience that exceeded our expectations. Diego Vega, our guide, seemed to be a whale whisperer. The whales appeared magically, it seemed, and one played near and around our boat. Diego also gave two evening wh
8. *Isla_Mujeres · Attractive · polaridad 5 · 74 palabras de contenido · idioma en* — They call it the Mayan Sacrifice for a reason!!! :-). Oh, Soggy Peso, how do I love you? Let me count the ways...#1 - best tortilla chips we ate in Mexico. No joke.#2 - deliciously and deceptively murderous margaritas#3 - slammers - a bubbling concoction of tequila and soda taken as a shot. Also kno
9. *Tulum · Restaurant · polaridad 5 · 13 palabras de contenido · idioma en* — Bachelorette party...Gibran Lavoe. Lo mejor de lo mejor. The best service ever!!! Gibran Lavoie y Rodolfo el mejor servicio!!! Xoxo xoxo
10. *Loreto · Attractive · polaridad 4 · 104 palabras de contenido · idioma en* — 5 days on Islands of Loreto tour. My friend and I really enjoyed our 5 day Islands of Loreto tour with ROW (listed as 7 days but no activities on the extra 2, only travel or the pre-trip meeting). We paddled for about 2 hours in the morning and after lunch and snorkeled every day. Our leaders, Maria

## 4. Casi duplicados dentro del pueblo (coseno > 0.95)

Con gemela: 0.1 %; sobrantes si se deja una por grupo: 0.1 % (106 reseñas). El deduplicado de `data.py` ya quitó los textos idénticos.

| Grupo | Reseñas | con gemela | sobrante (se quitaría) |
|---|---|---|---|
| Todas | 207,867 | 0.1 % | 0.1 % |
| Tipo: Hotel | 51,305 | 0.0 % | 0.0 % |
| Tipo: Restaurant | 86,714 | 0.1 % | 0.1 % |
| Tipo: Attractive | 69,848 | 0.1 % | 0.0 % |
| Polaridad 1 | 5,438 | 0.0 % | 0.0 % |
| Polaridad 2 | 5,492 | 0.1 % | 0.0 % |
| Polaridad 3 | 15,501 | 0.0 % | 0.0 % |
| Polaridad 4 | 44,980 | 0.1 % | 0.0 % |
| Polaridad 5 | 136,456 | 0.1 % | 0.1 % |

Ejemplos (con su gemela más cercana):

1. *Teotihuacan · Attractive · polaridad 4 · 39 palabras de contenido · idioma es* · gemela row_id 7785 (coseno 0.959): «Lugar unico. Es un lugar que realmente impacta a los sentidos. Se puede ingresar a templos y admirar pinturas y altares ceremoniales. En la calzada de los muert» — Vale la pena, es unico! Es un lugar que realmente impacta a los sentidos. Se puede ingresar a templos y admirar pinturas y altares ceremoniales. En la calzada de los muertos se puede revivir el paso de la civilizacion, admirar los murales a los laterales (los que aun se conservan) y tener la posibil
2. *Teotihuacan · Attractive · polaridad 5 · 40 palabras de contenido · idioma es* · gemela row_id 46791 (coseno 0.958): «Anímate!!! Se me rompió a llorar de alegría, este lugar es tan hermoso que es irreal. Se tarda un par de horas para ver todo el sitio, y es un poco caliente, es» — Merece la pena, pero ya sabes que. Se me rompió a llorar de alegría, este lugar es tan hermoso que es irreal. Se tarda un par de horas para ver todo el sitio, y es un poco caliente, especialmente en la cima de la pirámide del sol. La vista desde la cima es impresionante y la pirámide es un poco esti
3. *Atlixco · Restaurant · polaridad 5 · 11 palabras de contenido · idioma es* · gemela row_id 33835 (coseno 0.952): «Desayuno. Muy buen desayuno y excelente atención de Arturo Valencia. El pan está delicioso también. Recomendado» — Desayuno delicioso. Muy rico el desayuno como siempre. Desde el pan recién horneado hasta los platillos. La atención de Arturo Valencia excelente.
4. *Teotihuacan · Attractive · polaridad 4 · 46 palabras de contenido · idioma es* · gemela row_id 24743 (coseno 0.990): «La experiencia de la visita amerita una mejor gestión. Es la segunda vez que tengo la oportunidad de hacer la visita (1984 fue la primera), y lamentablemente fr» — La experiencia amerita una mejor gestión del servicio. Es la segunda vez que tengo la oportunidad de hacer la visita (1984 fue la primera), y lamentablemente frente a los restos arqueológicos (muy bien conservados) la gestión de la visita brilla por su ausencia, desorden, tumulto, aglomeración y fal
5. *Palenque · Attractive · polaridad 5 · 51 palabras de contenido · idioma es* · gemela row_id 67902 (coseno 0.953): «Tesoro mexicano. NO TENGO palabras! Este es un lugar que debes visitar! Se puede llegar en coche, puedes ir con un tour o puedes ir en transporte público (desde» — Tesoro mexicano. NO TENGO palabras! Este es un lugar que debes visitar! Se puede llegar en coche, puedes ir con un tour o puedes ir en transporte público (desde Palenque, o Ocosiongo dejan en el cruce, desde allí se debe tomar un taxi (como 15 minutos de trayecto).Hay baños, restaurantes, artesanía.
6. *Chiapa_de_Corzo · Attractive · polaridad 5 · 24 palabras de contenido · idioma es* · gemela row_id 117849 (coseno 0.959): «Espectacular! Chiapas es maravillosa y hay que empezar con Cañón del sumidero, un lugar verdaderamente mágico. La mejor idea es volar a Tuxtla y vaya directamen» — Lugar maravilloso. Chiapas es maravilloso y hay que iniciar con el Cañón del Sumidero, un lugar mágico. La idea es volar a Tuxtla e ir directamente a San Cristobal de la Casas. Se puede tomar autobus desde el aeropuerto o compartir un taxi. El tour se organiza desde el hotel en San Cristóbal.
7. *Teotihuacan · Attractive · polaridad 5 · 80 palabras de contenido · idioma es* · gemela row_id 93869 (coseno 0.964): «Tercero" de la pirámide" en el mundo! Llegamos aquí en autobús local que paraba justo fuera de las puertas. Hay muchas guías oficiales de trabajo allí (en las p» — Interesante y buen ejercicio! Llegamos aquí en autobús local que paraba justo fuera de las puertas. Hay muchas guías oficiales de trabajo allí (en las puertas y desplegado más freelancer escribe dentro; ambos con tienen las credenciales oficiales colgando alrededor del cuello). Elegimos la segunda o
8. *Valladolid · Attractive · polaridad 4 · 45 palabras de contenido · idioma es* · gemela row_id 120755 (coseno 0.971): «Vale la pena la visita. Leímos algunas evaluaciones antes de ir, por lo que sabíamos que habría personas en la puerta de entrada tratando de que los acompañara » — Increíble lugar, pero la gente intenta engañarte. Leímos algunas evaluaciones antes de ir, por lo que sabíamos que habría personas en la puerta de entrada tratando de que los acompañara y luego cobrándolo. Realmente los hay y se visten igual que los guías del lugar, pero no lo son. Además, si rentas
9. *Isla_Mujeres · Restaurant · polaridad 5 · 13 palabras de contenido · idioma es* · gemela row_id 52596 (coseno 0.953): «Desayuno increíble. Restaurante muy acogedor. Sirve un desayuno fantástico - porciones enormes! Sin duda volveremos. Me encantó todo.» — Desayuno increíble! Restaurante me encantó. Teníamos un gran desayuno con la familia. El personal era muy amable y la comida era increíble y fresco. Buenas porciones de tamaño medio.
10. *Teotihuacan · Attractive · polaridad 5 · 7 palabras de contenido · idioma es* · gemela row_id 86672 (coseno 0.995): «INCREIBLE. Deben visitarlo, el precio de la excursión no es caro y vale totalmente la pena! No se lo pierdan!!!» — increible. Deben visitarlo, el precio de la excursión no es caro y vale totalmente la pena! No se lo pierdan!!!

## 5. Mencionan su pueblo o estado

| Grupo | Reseñas | menciona |
|---|---|---|
| Todas | 207,867 | 27.6 % |
| Tipo: Hotel | 51,305 | 31.5 % |
| Tipo: Restaurant | 86,714 | 23.3 % |
| Tipo: Attractive | 69,848 | 30.2 % |
| Polaridad 1 | 5,438 | 14.6 % |
| Polaridad 2 | 5,492 | 17.2 % |
| Polaridad 3 | 15,501 | 21.4 % |
| Polaridad 4 | 44,980 | 27.9 % |
| Polaridad 5 | 136,456 | 29.2 % |

Ejemplos de reseñas que **no** lo mencionan:

1. *Valladolid · Restaurant · polaridad 5 · 7 palabras de contenido · idioma es* — POV. Muy buen servicio, el ambiente esta muy aseado y agradable la comida a tiempo y el trato muy bueno..
2. *Isla_Mujeres · Hotel · polaridad 2 · 54 palabras de contenido · idioma es* — Mala opción. Gran ubicación, con una piscina. Colchón fantástico y A/C funcionó a la perfección. Ahora a los problemas. Agua caliente en la ducha, normalmente nunca caliente, y a veces guay. Servicio de atención al cliente en la recepción muy pobre. Nos hizo sentir interrumpías mientras que navegué 
3. *Valladolid · Restaurant · polaridad 5 · 10 palabras de contenido · idioma es* — Muy buen servicio. Muy cálido y cómodo el lugar, super limpio y la comida deliciosa, pero lo que mas me gusta es el servicio porque son muy amables y atenciosos
4. *Tulum · Attractive · polaridad 5 · 15 palabras de contenido · idioma es* — Maravilla !! Un lugar paradisiaco . Para ver . Está en un enclave espectacular. Cada edificio tiene su historia y con un guía como el nuestro " pajarito " pues el no va más. Todo un profesional que ama su trabajo y al pueblo Maya. No os lo perdais.
5. *Tulum · Restaurant · polaridad 5 · 16 palabras de contenido · idioma es* — Comida deliciosa, servicio excelente! El servicio es muy bueno. En este caso nos atendió Azael, muy atento, que nos recomendó varios platillos. Le hicimos caso y fue una gran idea. La comida estaba deliciosa y en muy buena cantidad. 100% recomendable.
6. *Tulum · Attractive · polaridad 5 · 21 palabras de contenido · idioma es* — La playa es espectacular. Las ruinas son pequeñas, y no son gran cosa. Lo espectacular es que son las únicas que están al lado del Mar Caribe, y esto genera unos paisajes grandiosos. Bajar a la playa y bañarse con las ruinas mayas de escenario es grandioso, sólo por esto último vale la pena.
7. *San_Cristobal_de_las_Casas · Attractive · polaridad 2 · 41 palabras de contenido · idioma es* — No abierto a hora indicada. Tenía muchas ganas de visitar este museo y doble ychecked que sería abierto en lunes. Varios sitios web oficiales y dos guías de acuerdo en que fue abierto cada día de 10. 00. Llegué a las 10. 20 de la mañana sólo para encontrar todo y no hay información en horario public
8. *Ajijic · Restaurant · polaridad 5 · 42 palabras de contenido · idioma es* — Maravilloso entorno para comer al aire libre. Genuina cordial bienvenida. El entorno de mesas de comedor sobre una piscina y un bonito césped verde todo sobre hace un ambiente encantador. Yves paga una visita a diners les aseguro un buen tiempo en su lugar, disfrutando de un delicioso menú y barra d
9. *Sayulita · Restaurant · polaridad 5 · 12 palabras de contenido · idioma es* — tacos increíbles! Este restaurante tiene uno de los mejores tacos que he probado jamás. Buen precio, buena cantidad de comida por plato ... y guacamole increíble!
10. *Tequisquiapan · Hotel · polaridad 5 · 19 palabras de contenido · idioma es* — Un lugar valioso en medio del tiempo. Es un lugar detenido en el tiempo, perfecto para descansar y conectar con la naturaleza, las habitaciones están en armonía con el lugar, te transportan, el personal es maravilloso, amable te hacen parte de la experiencia, el restaurante también fue una agradable

## 6. Hotel o Restaurant que solo habla del servicio

36.0 % de las reseñas de Hotel y Restaurant (49,710 reseñas; 23.9 % del corpus).

| Grupo | Reseñas | solo servicio |
|---|---|---|
| Todas | 207,867 | 23.9 % |
| Tipo: Hotel | 51,305 | 22.1 % |
| Tipo: Restaurant | 86,714 | 44.3 % |
| Tipo: Attractive | 69,848 | 0.0 % |
| Polaridad 1 | 5,438 | 47.9 % |
| Polaridad 2 | 5,492 | 36.9 % |
| Polaridad 3 | 15,501 | 25.5 % |
| Polaridad 4 | 44,980 | 18.9 % |
| Polaridad 5 | 136,456 | 23.9 % |

Ejemplos:

1. *Valle_de_Bravo · Restaurant · polaridad 2 · 10 palabras de contenido · idioma es* — Me decepcionó. La verdad no entendí porque esta tan bien calificado. Para mi la comida me decepcionó, mi expectativa era mucho alta de acuerdo a la calificación y al precio
2. *Metepec · Restaurant · polaridad 3 · 12 palabras de contenido · idioma es* — Mucho de poco. Mucho de poco significa que el lugar no tiene mucho que ofrecer en cuanto a concepto. Sin embargo; el sabor de los platillos es muy interesante y vale la pena el lugar por su atmósfera. LO MALO: El servicio y la atención es promedio, con
3. *Tlaquepaque · Restaurant · polaridad 4 · 19 palabras de contenido · idioma es* — Comida entre amigos. Los mejores chiles en nogada que he probado en mi vida. La opinión de la comida de todo el grupo fue: excelente. El servicio un tanto lento y el amargor llego en el postre, un hermoso tirsmisu en un frasco que al abrirlo hizo "plop",
4. *San_Cristobal_de_las_Casas · Restaurant · polaridad 5 · 8 palabras de contenido · idioma es* — Excelente lugar. Un gran sitio para comer y pasar un rato en familia , recomendable. Los domingos se pone a tope recomiendo hacer reserva!!
5. *Isla_Mujeres · Restaurant · polaridad 5 · 12 palabras de contenido · idioma es* — Cita obliga!!! Calidez en. Cita obliga!!! Calidez en la ambientación, comida inigualable y la atención de Geor hizo de nuestra visita PERFECTA.
6. *Tlaquepaque · Hotel · polaridad 2 · 53 palabras de contenido · idioma es* — Publicidad engañosa. Fui de visita a guadalajara en noviembre solo por negocios, entonces buscaba algo del concepto del city express solo una noche, entiendo que no es un hotel de lujo pero que sea económico no tiene que ver con que cuando llegue había palomitas tiradas por todo el suelo!, otra cosa
7. *Isla_Mujeres · Restaurant · polaridad 5 · 8 palabras de contenido · idioma es* — Excelente Atención. Lugar paradisíaco y su gente es lo mejor! Fueron vacaciones soñadas. La atención de Camila fue excelente. Meritoria su preocupación por nosotros
8. *Tlaquepaque · Restaurant · polaridad 4 · 9 palabras de contenido · idioma es* — Lugar para visitar. Muy buen servicio de parte del mesero Zárate, muy agradable ambientación y buen detalle la música mexicana en vivo
9. *Tepoztlan · Hotel · polaridad 4 · 47 palabras de contenido · idioma es* — "Buen fin de semana". Es un lindo hotel para descansar. No es tan maravilloso como su página sugiere, pero es un buen lugar. Lo que decepciona un poco es que hay muchas cosas que anuncian en su página, pero no siempre están disponibles ni los cursos ni talleres ni el observatorio astronomico. Es un 
10. *Metepec · Restaurant · polaridad 3 · 18 palabras de contenido · idioma es* — Opción para ver partidos. Es un lugar con mobiliario viejo y, si tienes niños tiene un espacio donde se pueden divertir y tienen un par de nanas que hacen lo posible por estar al tanto; falta que circule el Aire en esta área, y obvio un espacio para dejar

## Resumen

- Alguna categoría de ruido (genérica, otro idioma, sobrante duplicada o solo servicio): 25.6 %.
- genérica (las tres a la vez): 3.6 %; otro idioma: 0.1 %; casi duplicada (tiene gemela > 0.95 en su pueblo): 0.1 %; menciona su pueblo o estado: 27.6 %; Hotel/Restaurant solo de servicio: 23.9 %.
- Las cifras por pueblo están en `reports/exp/2026-11-16_calidad_resenas/diagnostico_por_pueblo.csv`.
