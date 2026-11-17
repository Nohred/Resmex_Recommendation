"use strict";

/* ---------- utilidades ---------- */
const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => Number(x).toFixed(d);
const num = (x) => Number(x).toLocaleString("es-MX");
const ms = (x) => (x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${x < 10 ? x.toFixed(1) : Math.round(x)} ms`);
const stars = (p) => "★".repeat(Math.round(p)) + "☆".repeat(5 - Math.round(p));
const sleep = (t) => new Promise((r) => setTimeout(r, t));
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const narrow = () => window.matchMedia("(max-width: 760px)").matches;

const form = $("form"), query = $("query"), submit = $("submit");
const alertBox = $("alert"), results = $("results"), reviews = $("reviews"), live = $("live");

const FALLBACK_EXAMPLES = [
  "Quiero un lugar tranquilo con zonas arqueológicas y buena comida tradicional",
  "Busco fiesta, playa y vida nocturna",
  "Me interesan pueblos coloniales con artesanías y clima frío",
  "Naturaleza, cascadas y aventura lejos de las multitudes",
];
const PROVIDER_NAMES = { gemini: "Gemini", groq: "Groq", ollama: "el modelo local (Ollama)", plantilla: "una plantilla, sin modelo de lenguaje" };

const state = { towns: [], geo: null, result: null, byTown: new Map(), busy: false };

/* ---------- avisos ---------- */
function showAlert(html, kind = "error") {
  alertBox.className = "alert" + (kind === "warn" ? " alert--warn" : "");
  alertBox.innerHTML = html;
  alertBox.hidden = false;
}
const say = (text) => { live.textContent = text; };

/* ---------- estado del sistema y ejemplos ---------- */
function renderExamples(list) {
  const ul = $("examples");
  ul.innerHTML = "";
  list.forEach((text) => {
    const li = document.createElement("li"), b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", () => { query.value = text; form.requestSubmit(); });
    li.appendChild(b);
    ul.appendChild(li);
  });
}

async function checkHealth() {
  const el = $("status"), label = el.querySelector("span");
  const set = (cls, text) => { el.className = "status " + cls; label.textContent = text; };
  try {
    const h = await (await fetch("/health")).json();
    renderExamples(h.examples?.length ? h.examples : FALLBACK_EXAMPLES);
    if (h.index?.loaded) {
      $("stat-towns").textContent = h.index.towns;
      $("stat-arch").textContent = h.index.archetypes;
      if (h.index.n_train) $("stat-reviews").textContent = `${Math.round(h.index.n_train / 1000)} mil`;
    }
    const chain = h.llm?.chain || [], first = chain[0];
    if (!h.index?.loaded) set("status--bad", "Falta el índice: ejecuta make index");
    else if (!h.ollama?.ok) set("status--bad", "Ollama está apagado: inicia ollama serve");
    else if (!h.ollama?.embedding_model) set("status--bad", "Falta el modelo de embeddings en Ollama");
    else if (first) {
      const active = chain.find((p) => p.available && !p.exhausted) || first;
      const name = active.provider === "ollama" ? "modelo local" : (PROVIDER_NAMES[active.provider] || active.provider);
      if (active === first) set("status--ok", `Listo, escribe ${name} (${active.model})`);
      else set("status--warn", `Listo con respaldo: escribe ${name} (${active.model})`);
    } else set("status--warn", "Listo, sin modelo de lenguaje: se usará la plantilla");
  } catch {
    renderExamples(FALLBACK_EXAMPLES);
    set("status--bad", "El servidor no responde");
  }
}

/* ---------- geometría del mapa ---------- */
const COS = Math.cos(23.5 * Math.PI / 180);
const MAIN = { lon0: -118.6, lon1: -86.4, lat0: 14.2, lat1: 33.0 };
const CENTER = { lon0: -100.4, lon1: -96.7, lat0: 18.35, lat1: 21.0 };   // recuadro ampliado
const INSET_BOX = { left: 0.668, top: 0.0625, width: 0.30 };             // debe coincidir con el CSS
const inCenter = (t) => t.lon > CENTER.lon0 && t.lon < CENTER.lon1 && t.lat > CENTER.lat0 && t.lat < CENTER.lat1;

function projector(view, width) {
  const k = width / ((view.lon1 - view.lon0) * COS);
  const fn = (lon, lat) => [(lon - view.lon0) * COS * k, (view.lat1 - lat) * k];
  fn.height = (view.lat1 - view.lat0) * k;
  return fn;
}

function geoPath(kind, proj) {
  let d = "";
  for (const f of state.geo?.features || []) {
    if (f.properties.kind !== kind) continue;
    for (const poly of f.geometry.coordinates) {
      for (const ring of poly) {
        d += ring.map(([lon, lat], i) => {
          const [x, y] = proj(lon, lat);
          return `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
        }).join("") + "Z";
      }
    }
  }
  return d;
}

/* Coloca rótulos alrededor de sus puntos: prueba ocho posiciones y se queda con la que menos estorba. */
function placeLabels(items, fontPx, width, height, blocked = []) {
  const boxes = items.map((it) => ({ x0: it.x - 6, x1: it.x + 6, y0: it.y - 6, y1: it.y + 6 }));
  const overlap = (a, b) => Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) * Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  return items.map((it, i) => {
    if (!it.text) return null;
    const w = it.text.length * fontPx * 0.58 + 6, h = fontPx + 3, g = it.gap ?? 9, d = g * 0.7;
    const at = (x0, y0, anchor) => ({
      anchor, box: { x0, x1: x0 + w, y0, y1: y0 + h },
      x: anchor === "start" ? x0 : anchor === "end" ? x0 + w : x0 + w / 2, y: y0 + h - 3 - fontPx * 0.12,
    });
    const options = [
      at(it.x + g, it.y - h / 2, "start"), at(it.x - g - w, it.y - h / 2, "end"),
      at(it.x - w / 2, it.y - g - h, "middle"), at(it.x - w / 2, it.y + g, "middle"),
      at(it.x + d, it.y - d - h, "start"), at(it.x + d, it.y + d, "start"),
      at(it.x - d - w, it.y - d - h, "end"), at(it.x - d - w, it.y + d, "end"),
    ];
    const cost = (o, k) => {
      const b = o.box;
      const out = Math.max(0, 2 - b.x0) + Math.max(0, b.x1 - width + 2) + Math.max(0, 2 - b.y0) + Math.max(0, b.y1 - height + 2);
      const banned = blocked.reduce((sum, other) => sum + overlap(b, other), 0);
      return out * 1000 + banned * 50 + boxes.reduce((sum, other, j) => (j === i ? sum : sum + overlap(b, other)), 0) + k * 0.5;
    };
    const chosen = options.map((o, k) => [cost(o, k), o]).sort((a, b) => a[0] - b[0])[0][1];
    boxes.push(chosen.box);
    return chosen;
  });
}

/* ---------- mapa ---------- */
function townInfo(town) {
  const base = state.byTown.get(town) || {};
  const scored = state.result?.towns?.find((t) => t.town === town);
  return { ...base, ...(scored || {}), scored: Boolean(scored) };
}

function glowRange() {
  const scores = (state.result?.towns || []).map((t) => t.score);
  return scores.length ? [Math.min(...scores), Math.max(...scores)] : [0, 1];
}

function describe(t) {
  const where = `${t.town_display}, ${t.region}`;
  if (!t.scored) {
    const themes = (t.archetypes || []).slice(0, 3).join(", ");
    return `${where}. ${num(t.n_reviews)} reseñas.${themes ? ` De lo que más se habla: ${themes}.` : ""}`;
  }
  const group = t.group === "top" ? `Número ${t.rank} de tu top 5. `
    : t.group === "bottom" ? `Número ${t.rank} entre los que se parecen a tu pesadilla. ` : "";
  const label = t.group === "bottom" ? t.anti_archetype_label : t.archetype_label;
  return `${where}. ${group}Arquetipo: ${label}. Parecido con tu viaje ideal ${fmt(t.sim_hyde)}, con tu pesadilla ${fmt(t.sim_anti)}.`;
}

function tipHtml(t) {
  const head = `<strong>${esc(t.town_display)}</strong><span>${esc(t.region)}</span>`;
  if (!t.scored) {
    const themes = (t.archetypes || []).slice(0, 3).map((a) => `<li>${esc(a)}</li>`).join("");
    return `${head}<p>${num(t.n_reviews)} reseñas. De lo que más se habla:</p><ul>${themes}</ul>`;
  }
  const badge = t.group === "top" ? `<p class="tip-top">N.º ${t.rank} de tu top 5</p>`
    : t.group === "bottom" ? `<p class="tip-bottom">N.º ${t.rank} de los que evitarías</p>` : "";
  const label = t.group === "bottom" ? t.anti_archetype_label : t.archetype_label;
  return `${head}${badge}<p class="tip-arch">${esc(label)}</p>
    <dl><dt>Ideal</dt><dd>${fmt(t.sim_hyde)}</dd><dt>Pesadilla</dt><dd>${fmt(t.sim_anti)}</dd><dt>Puntaje</dt><dd>${fmt(t.score)}</dd></dl>`;
}

function showTip(tip, frame, target, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const f = frame.getBoundingClientRect(), r = target.getBoundingClientRect();
  const cx = r.left + r.width / 2 - f.left, cy = r.top + r.height / 2 - f.top;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  const left = Math.max(4, Math.min(f.width - w - 4, cx - w / 2));
  const top = cy - h - 14 < 0 ? cy + 16 : cy - h - 14;
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}

function pointSvg(t, x, y, opts) {
  const info = townInfo(t.town);
  const [lo, hi] = glowRange();
  const glow = info.scored ? 0.25 + 0.75 * (info.score - lo) / Math.max(hi - lo, 1e-9) : 0.6;
  const cls = info.group === "top" ? "pt pt--top" : info.group === "bottom" ? "pt pt--bottom" : "pt";
  const r = opts.r;
  const delay = info.group === "top" ? (info.rank - 1) * 140 : Math.round((x * 7 + y * 13) % 1800);
  let inner = "";
  if (info.group === "top") inner += `<circle class="pt-halo" r="${r * 5}" fill="url(#${opts.halo})"/>`;
  if (info.group === "bottom") {
    const c = r * 0.62;
    inner += `<circle class="pt-ring" r="${r + 1.5}"/><path class="pt-x" d="M${-c} ${-c}L${c} ${c}M${c} ${-c}L${-c} ${c}"/>`;
  } else inner += `<circle class="pt-core" r="${info.group === "top" ? r + 1 : r}"/>`;
  if (!opts.interactive) {
    return `<g class="${cls} pt--echo" transform="translate(${x.toFixed(1)} ${y.toFixed(1)})" style="--glow:${glow.toFixed(2)};--d:${delay}ms" aria-hidden="true">${inner}</g>`;
  }
  return `<g class="${cls}" data-town="${esc(t.town)}" transform="translate(${x.toFixed(1)} ${y.toFixed(1)})"
    style="--glow:${glow.toFixed(2)};--d:${delay}ms" tabindex="-1" role="img" aria-label="${esc(describe(info))}">
    <circle class="pt-hit" r="${opts.hit}"/>${inner}</g>`;
}

function labelsSvg(points, fontPx, width, height, numbersOnly, blocked = []) {
  const marked = points
    .map((p) => ({ ...p, info: townInfo(p.t.town) }))
    .filter((p) => p.info.group && !(numbersOnly && p.info.group === "bottom"))
    .sort((a, b) => (a.info.group === b.info.group ? a.info.rank - b.info.rank : a.info.group === "top" ? -1 : 1));
  const items = marked.map((p) => ({
    x: p.x, y: p.y, gap: p.info.group === "top" ? 11 : 9,
    text: p.info.group === "top"
      ? (numbersOnly ? String(p.info.rank) : `${p.info.rank}  ${p.info.town_display}`)
      : p.info.town_display,
  }));
  // Los puntos sin rótulo también estorban: se agregan como obstáculos.
  const obstacles = points.filter((p) => !marked.some((m) => m.t.town === p.t.town))
    .map((p) => ({ x: p.x, y: p.y, text: "", gap: 0 }));
  const placed = placeLabels([...items, ...obstacles], fontPx, width, height, blocked).slice(0, items.length);
  return placed.map((pos, i) => {
    const p = marked[i];
    const cls = p.info.group === "top" ? "lbl lbl--top" : "lbl lbl--bottom";
    const text = p.info.group === "top" && !numbersOnly
      ? `<tspan class="lbl-rank">${p.info.rank}</tspan><tspan dx="5">${esc(p.info.town_display)}</tspan>`
      : esc(items[i].text);
    return `<text class="${cls}" x="${pos.x.toFixed(1)}" y="${pos.y.toFixed(1)}" text-anchor="${pos.anchor}" font-size="${fontPx}" aria-hidden="true">${text}</text>`;
  }).join("");
}

const defs = (id, w = 1, h = 1) => `<defs><radialGradient id="fade-${id}" r=".66"><stop offset=".62" stop-color="#fff"/><stop offset="1" stop-color="#000"/></radialGradient>
  <mask id="mask-${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="${w}" height="${h}"><rect width="${w}" height="${h}" fill="url(#fade-${id})"/></mask><radialGradient id="${id}"><stop offset="0" stop-color="#FFB627" stop-opacity=".55"/>
  <stop offset=".45" stop-color="#FFB627" stop-opacity=".16"/><stop offset="1" stop-color="#FFB627" stop-opacity="0"/></radialGradient></defs>`;

function renderMap() {
  const svg = $("map"), frame = $("map-frame");
  const width = Math.max(300, Math.round(frame.clientWidth));
  const proj = projector(MAIN, width), height = Math.round(proj.height);
  const small = narrow();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const outside = state.towns.filter((t) => !inCenter(t)), inside = state.towns.filter(inCenter);
  const pts = outside.map((t) => { const [x, y] = proj(t.lon, t.lat); return { t, x, y }; });
  const [cx0, cy0] = proj(CENTER.lon0, CENTER.lat1), [cx1, cy1] = proj(CENTER.lon1, CENTER.lat0);
  const leaders = small ? "" : (() => {
    const ix0 = INSET_BOX.left * width, ix1 = (INSET_BOX.left + INSET_BOX.width) * width;
    const iy1 = INSET_BOX.top * height + INSET_BOX.width * width * (234 / 300);
    return `<path class="leader" d="M${cx0} ${cy0}L${ix0} ${iy1}M${cx1} ${cy0}L${ix1} ${iy1}"/>`;
  })();
  const opts = { r: small ? 3 : 3.6, hit: small ? 11 : 13, halo: "halo-main", interactive: true };
  svg.innerHTML = defs("halo-main", width, height)
    + `<g mask="url(#mask-halo-main)"><path class="land land--neighbor" d="${geoPath("neighbor", proj)}"/></g>`
    + `<path class="land land--mx" d="${geoPath("country", proj)}"/>`
    + `<path class="states" d="${geoPath("state", proj)}"/>`
    + `<path class="coast" d="${geoPath("country", proj)}"/>`
    + leaders
    + `<rect class="zoom-box" x="${cx0}" y="${cy0}" width="${cx1 - cx0}" height="${cy1 - cy0}"/>`
    + inside.map((t) => { const [x, y] = proj(t.lon, t.lat); return pointSvg(t, x, y, { ...opts, r: small ? 1.6 : 2, interactive: false }); }).join("")
    + pts.map((p) => pointSvg(p.t, p.x, p.y, opts)).join("")
    + labelsSvg(pts, small ? 12 : 13, width, height, small, small ? [] : [{
      x0: INSET_BOX.left * width - 4, x1: (INSET_BOX.left + INSET_BOX.width) * width + 4,
      y0: INSET_BOX.top * height - 22, y1: INSET_BOX.top * height + INSET_BOX.width * width * (234 / 300) + 4,
    }]);

  // Recuadro ampliado del centro del país
  const inset = $("inset"), iw = Math.max(200, Math.round($("inset-frame").clientWidth));
  const iproj = projector(CENTER, iw), ih = Math.round(iproj.height);
  inset.setAttribute("viewBox", `0 0 ${iw} ${ih}`);
  const ipts = inside.map((t) => { const [x, y] = iproj(t.lon, t.lat); return { t, x, y }; });
  inset.innerHTML = defs("halo-inset")
    + `<path class="land land--mx" d="${geoPath("country", iproj)}"/>`
    + `<path class="states" d="${geoPath("state", iproj)}"/>`
    + ipts.map((p) => pointSvg(p.t, p.x, p.y, { r: 3.4, hit: 10, halo: "halo-inset", interactive: true })).join("")
    + labelsSvg(ipts, small ? 12 : 11, iw, ih, false);

  const first = frame.querySelector(".pt--top[data-town]") || frame.querySelector(".pt[data-town]");
  if (first) first.setAttribute("tabindex", "0");
}

function setupMapInteraction() {
  const frame = $("map-frame"), tip = $("tip");
  const pointOf = (e) => e.target.closest?.(".pt[data-town]");
  const open = (g) => { showTip(tip, frame, g.querySelector(".pt-hit"), tipHtml(townInfo(g.dataset.town))); mark(g.dataset.town, true); };
  const close = (g) => { tip.hidden = true; if (g) mark(g.dataset.town, false); };
  frame.addEventListener("pointerover", (e) => { const g = pointOf(e); if (g) open(g); });
  frame.addEventListener("pointerout", (e) => { const g = pointOf(e); if (g) close(g); });
  frame.addEventListener("focusin", (e) => { const g = pointOf(e); if (g) open(g); });
  frame.addEventListener("focusout", (e) => { const g = pointOf(e); if (g) close(g); });
  frame.addEventListener("click", (e) => { const g = pointOf(e); if (g) { g.focus(); jumpToCard(g.dataset.town); } });
  frame.addEventListener("keydown", (e) => {
    const g = pointOf(e);
    if (!g) return;
    if (e.key === "Escape") { tip.hidden = true; return; }
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); jumpToCard(g.dataset.town); return; }
    const dir = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    if (!dir) return;
    e.preventDefault();
    const center = (el) => { const r = el.querySelector(".pt-hit").getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };
    const [x, y] = center(g);
    let best = null, bestCost = Infinity;
    frame.querySelectorAll(".pt[data-town]").forEach((other) => {
      if (other === g) return;
      const [ox, oy] = center(other), dx = ox - x, dy = oy - y;
      const along = dx * dir[0] + dy * dir[1], across = Math.abs(dx * dir[1]) + Math.abs(dy * dir[0]);
      if (along <= 1) return;
      const cost = along + across * 2.5;      // prefiere al vecino más alineado con la flecha
      if (cost < bestCost) { bestCost = cost; best = other; }
    });
    if (best) {
      frame.querySelectorAll('.pt[tabindex="0"]').forEach((el) => el.setAttribute("tabindex", "-1"));
      best.setAttribute("tabindex", "0");
      best.focus();
    }
  });
}

function jumpToCard(town) {
  const card = document.querySelector(`.card[data-town="${CSS.escape(town)}"]`);
  if (!card) return;
  card.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
  card.querySelector("h3")?.focus({ preventScroll: true });
}

/* Resalta el mismo pueblo en el mapa, el plano y las tarjetas. */
function mark(town, on) {
  document.querySelectorAll(`[data-town="${CSS.escape(town)}"]`).forEach((el) => el.classList.toggle("is-hot", on));
}

/* ---------- plano afinidad / rechazo ---------- */
function renderPlane() {
  const data = state.result;
  if (!data?.towns?.length) return;
  const svg = $("plane"), frame = $("plane-frame");
  const size = Math.max(280, Math.min(560, Math.round(frame.clientWidth)));
  const m = { l: 46, r: 14, t: 14, b: 44 }, inner = size - m.l - m.r;
  const all = data.towns.flatMap((t) => [t.sim_hyde, t.sim_anti]);
  const span = Math.max(...all) - Math.min(...all) || 0.1;
  const lo = Math.min(...all) - span * 0.08, hi = Math.max(...all) + span * 0.1;
  const sx = (v) => m.l + (v - lo) / (hi - lo) * inner, sy = (v) => m.t + inner - (v - lo) / (hi - lo) * inner;
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.style.height = `${size}px`;

  const step = span > 0.25 ? 0.1 : 0.05, ticks = [];
  for (let v = Math.ceil(lo / step) * step; v < hi; v += step) ticks.push(Number(v.toFixed(2)));
  const grid = ticks.map((v) => `<path class="grid" d="M${sx(v)} ${m.t}V${m.t + inner}M${m.l} ${sy(v)}H${m.l + inner}"/>
    <text class="tick" x="${sx(v)}" y="${m.t + inner + 16}" text-anchor="middle">${v.toFixed(2)}</text>
    <text class="tick" x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`).join("");

  const pts = data.towns.map((t) => ({ t, x: sx(t.sim_hyde), y: sy(t.sim_anti) }));
  const order = (p) => (p.t.group === "top" ? 2 : p.t.group === "bottom" ? 1 : 0);
  const dots = [...pts].sort((a, b) => order(a) - order(b)).map((p) => {
    const cls = p.t.group === "top" ? "pp pp--top" : p.t.group === "bottom" ? "pp pp--bottom" : "pp";
    const halo = p.t.group === "top" ? `<circle class="pp-halo" r="16" fill="url(#halo-plane)"/>` : "";
    return `<g class="${cls}" data-town="${esc(p.t.town)}" transform="translate(${p.x.toFixed(1)} ${p.y.toFixed(1)})">
      <circle class="pp-hit" r="11"/>${halo}<circle class="pp-dot" r="${p.t.group ? 5 : 3.5}"/></g>`;
  }).join("");
  const marked = pts.filter((p) => p.t.group).sort((a, b) => order(b) - order(a) || a.t.rank - b.t.rank);
  const others = pts.filter((p) => !p.t.group).map((p) => ({ x: p.x, y: p.y, text: "", gap: 0 }));
  const fontPx = size < 400 ? 11 : 12.5;
  const placed = placeLabels([...marked.map((p) => ({ x: p.x, y: p.y, text: p.t.town_display, gap: 9 })), ...others], fontPx, size, size);
  const labels = marked.map((p, i) => `<text class="lbl ${p.t.group === "top" ? "lbl--top" : "lbl--bottom"}" x="${placed[i].x.toFixed(1)}"
    y="${placed[i].y.toFixed(1)}" text-anchor="${placed[i].anchor}" font-size="${fontPx}">${esc(p.t.town_display)}</text>`).join("");

  svg.innerHTML = defs("halo-plane")
    + `<path class="half half--cold" d="M${m.l} ${m.t}H${m.l + inner}L${m.l} ${m.t + inner}Z"/>`
    + `<path class="half half--warm" d="M${m.l + inner} ${m.t}V${m.t + inner}H${m.l}Z"/>`
    + grid
    + `<path class="diag" d="M${m.l} ${m.t + inner}L${m.l + inner} ${m.t}"/>`
    + `<text class="corner corner--cold" x="${m.l + 10}" y="${m.t + inner * 0.56}">Más cerca de tu pesadilla</text>`
    + `<text class="corner corner--warm" x="${m.l + inner - 10}" y="${m.t + inner - 12}" text-anchor="end">Más cerca de tu viaje ideal</text>`
    + `<text class="axis" x="${m.l + inner / 2}" y="${size - 6}" text-anchor="middle">Parecido con tu viaje ideal</text>`
    + `<text class="axis" transform="translate(13 ${m.t + inner / 2}) rotate(-90)" text-anchor="middle">Parecido con tu pesadilla</text>`
    + dots + labels;

  const below = data.towns.filter((t) => t.sim_hyde > t.sim_anti).length;
  $("plane-desc").textContent = `${below} de ${data.towns.length} pueblos se parecen más a tu viaje ideal que a tu pesadilla. `
    + `Los de tu top 5 son: ${data.top.map((t) => t.town_display).join(", ")}. La tabla de abajo tiene todos los valores.`;

  $("table-n").textContent = data.towns.length;
  $("table").innerHTML = [...data.towns].sort((a, b) => b.score - a.score).map((t) => `<tr data-town="${esc(t.town)}"
    class="${t.group ? "row--" + t.group : ""}"><th scope="row">${esc(t.town_display)}</th><td>${esc(t.region)}</td>
    <td class="num">${fmt(t.sim_hyde)}</td><td class="num">${fmt(t.sim_anti)}</td><td class="num">${fmt(t.score)}</td>
    <td>${t.group === "top" ? `Top ${t.rank}` : t.group === "bottom" ? `Bottom ${t.rank}` : ""}</td></tr>`).join("");
}

function setupPlaneInteraction() {
  const frame = $("plane-frame"), tip = $("plane-tip");
  frame.addEventListener("pointerover", (e) => {
    const g = e.target.closest?.(".pp");
    if (!g) return;
    showTip(tip, frame, g.querySelector(".pp-hit"), tipHtml(townInfo(g.dataset.town)));
    mark(g.dataset.town, true);
  });
  frame.addEventListener("pointerout", (e) => {
    const g = e.target.closest?.(".pp");
    if (g) { tip.hidden = true; mark(g.dataset.town, false); }
  });
}

/* ---------- tarjetas ---------- */
/* Tira con los 40 pueblos: dónde cae este y cuánto lo movió la penalización. */
function stripSvg(t, kind) {
  const towns = state.result.towns, key = kind === "top" ? "score" : "sim_anti";
  const values = towns.map((x) => x[key]).concat(kind === "top" ? towns.map((x) => x.sim_hyde) : []);
  const lo = Math.min(...values), hi = Math.max(...values), W = 300, pad = 8;
  const sx = (v) => pad + (v - lo) / Math.max(hi - lo, 1e-9) * (W - 2 * pad);
  const ticks = towns.map((x) => `<path class="st-tick" d="M${sx(x[key]).toFixed(1)} 9V19"/>`).join("");
  const here = sx(t[key]);
  const pull = kind === "top"
    ? `<path class="st-pull" d="M${sx(t.sim_hyde).toFixed(1)} 14H${here.toFixed(1)}"/><circle class="st-raw" cx="${sx(t.sim_hyde).toFixed(1)}" cy="14" r="4"/>`
    : "";
  return `<svg class="strip" viewBox="0 0 ${W} 28" preserveAspectRatio="none" aria-hidden="true">
    <path class="st-base" d="M${pad} 14H${W - pad}"/>${ticks}${pull}<path class="st-here" d="M${here.toFixed(1)} 3V25"/></svg>`;
}

function cardHtml(t, kind) {
  const isTop = kind === "top";
  const chips = (t.keywords || []).slice(0, isTop ? 7 : 5).map((k) => `<li>${esc(k)}</li>`).join("");
  const math = isTop
    ? `<span><b>${fmt(t.sim_hyde)}</b> de parecido</span> <span class="op">−</span> <span><b>${fmt(t.penalty)}</b> por popularidad</span> <span class="op">=</span> <span class="total"><b>${fmt(t.score)}</b></span>`
    : `<span><b>${fmt(t.sim_anti)}</b> de parecido con tu pesadilla</span>, <span><b>${fmt(t.sim_hyde)}</b> con tu viaje ideal</span>`;
  const stripNote = isTop
    ? "Cada raya es uno de los pueblos. El círculo hueco es el parecido antes de la penalización."
    : "Cada raya es uno de los pueblos, ordenados por su parecido con tu pesadilla.";
  const review = t.review ? `<details class="card-review"${isTop && t.rank === 1 ? " open" : ""}>
      <summary>Reseña representativa <span class="stars" aria-label="${fmt(t.review_polarity, 0)} de 5 estrellas">${stars(t.review_polarity)}</span></summary>
      <blockquote>${esc(t.review)}</blockquote></details>` : "";
  return `<li class="card card--${kind}" data-town="${esc(t.town)}">
    <div class="card-head">
      <span class="card-rank" aria-hidden="true">${t.rank}</span>
      <div class="card-name"><h3 tabindex="-1"><span class="sr-only">${t.rank}. </span>${esc(t.town_display)}</h3>
        <p>${esc(t.region)}, ${num(t.n_reviews)} reseñas</p></div>
      <p class="card-score"><b>${fmt(isTop ? t.score : t.sim_anti)}</b><span>${isTop ? "puntaje" : "parecido"}</span></p>
    </div>
    <p class="card-arch">${isTop ? "Coincide por" : "Choca por"} <b>${esc(t.archetype_label)}</b></p>
    <ul class="chips" aria-label="Palabras clave del arquetipo">${chips}</ul>
    <div class="card-math">${stripSvg(t, kind)}<p>${math}</p><p class="sr-only">${stripNote}</p></div>
    ${review}
  </li>`;
}

function renderCards() {
  const data = state.result;
  $("top").innerHTML = data.top.map((t) => cardHtml(t, "top")).join("");
  $("bottom").innerHTML = data.bottom.map((t) => cardHtml(t, "bottom")).join("");
  results.querySelectorAll(".card").forEach((card) => {
    card.addEventListener("pointerenter", () => mark(card.dataset.town, true));
    card.addEventListener("pointerleave", () => mark(card.dataset.town, false));
    card.addEventListener("focusin", () => mark(card.dataset.town, true));
    card.addEventListener("focusout", () => mark(card.dataset.town, false));
  });
}

/* ---------- reseñas que se escriben ---------- */
function prepareReview(el, text) {
  el.innerHTML = `<span class="typed"></span><span class="caret" aria-hidden="true"></span><span class="untyped"></span>`;
  el.querySelector(".untyped").textContent = text;
}

async function typeReviews(pairs, duration) {
  if (reducedMotion()) {
    pairs.forEach(([el, text]) => { el.textContent = text; });
    return;
  }
  const items = pairs.map(([el, text]) => { prepareReview(el, text); return { el, words: text.split(" ") }; });
  const start = performance.now();
  await new Promise((done) => {
    const frame = (now) => {
      const p = Math.min(1, (now - start) / duration);
      items.forEach(({ el, words }) => {
        const n = Math.round(words.length * p);
        el.querySelector(".typed").textContent = words.slice(0, n).join(" ");
        el.querySelector(".untyped").textContent = (n ? " " : "") + words.slice(n).join(" ");
      });
      if (p < 1) requestAnimationFrame(frame); else done();
    };
    requestAnimationFrame(frame);
  });
  pairs.forEach(([el, text]) => { el.textContent = text; });
}

function metaText(data) {
  const p = data.llm_provider, t = data.timings_ms || {};
  const who = data.fallback ? "Sin modelo de lenguaje: tu consulta tal cual y una plantilla de opuestos"
    : `Escritas por ${PROVIDER_NAMES[p] || p}${data.llm_model ? ` (${data.llm_model})` : ""}`;
  const when = data.fallback ? "" : (data.cached ? ", desde la caché" : `, en ${ms(t.llm)}`);
  return `${who}${when}. Embeddings en ${ms(t.embedding)}, búsqueda en ${ms(t.busqueda)}, λ = ${data.lambda}.`;
}

/* ---------- consulta ---------- */
async function* streamEvents(text) {
  const resp = await fetch("/recommend/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: text }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Error ${resp.status}`);
  }
  const reader = resp.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
    if (done) break;
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

function resetStage() {
  alertBox.hidden = true;
  results.hidden = true;
  state.result = null;
  $("meta").textContent = "";
  ["hyde", "anti"].forEach((id) => { $(id).innerHTML = `<span class="caret" aria-hidden="true"></span>`; });
  reviews.hidden = false;
  reviews.classList.add("is-writing");
  $("map-frame").classList.add("is-thinking");
  renderMap();
}

async function recommend(text) {
  if (state.busy) return;
  state.busy = true;
  submit.disabled = true;
  submit.textContent = "Escribiendo…";
  resetStage();
  say("Escribiendo las dos reseñas: tu viaje ideal y tu viaje de pesadilla.");
  if (narrow()) reviews.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  let typing = Promise.resolve(), gotResult = false;
  try {
    for await (const ev of streamEvents(text)) {
      if (ev.event === "expansion") {
        say("Reseñas listas. Comparando con los arquetipos de cada pueblo.");
        typing = typeReviews([[$("hyde"), ev.hyde], [$("anti"), ev.anti_hyde]], 2200);
      } else if (ev.event === "result") {
        await typing;
        gotResult = true;
        showResult(ev.data);
      } else if (ev.event === "error") {
        await typing;
        throw new Error(ev.detail || "Error del servidor");
      }
    }
    if (!gotResult) throw new Error("El servidor cortó la respuesta antes de terminar.");
  } catch (err) {
    const msg = err instanceof TypeError ? "No hay conexión con el servidor. Revisa que `make serve` siga activo y vuelve a intentar." : err.message;
    if (!$("hyde").textContent.trim()) reviews.hidden = true;
    showAlert(`<b>No se pudo recomendar.</b> ${esc(msg).replace(/`([^`]+)`/g, "<code>$1</code>")}`);
    alertBox.scrollIntoView({ behavior: "auto", block: "nearest" });
    say("");
    checkHealth();
  } finally {
    reviews.classList.remove("is-writing");
    $("map-frame").classList.remove("is-thinking");
    state.busy = false;
    submit.disabled = false;
    submit.textContent = "Encender pueblos";
  }
}

function showResult(data) {
  state.result = data;
  const p = data.llm_provider;
  if (data.fallback) {
    showAlert("<b>Ningún modelo de lenguaje respondió.</b> Usamos tu consulta tal cual como viaje ideal y una plantilla de opuestos como pesadilla. El resultado es menos fino; vuelve a intentar en un momento.", "warn");
  } else if (data.degraded) {
    showAlert(`<b>El modelo principal no respondió.</b> Las reseñas las escribió ${esc(PROVIDER_NAMES[p] || p)} (${esc(data.llm_model)}). El resultado sigue siendo válido.`, "warn");
  }
  $("meta").textContent = metaText(data);
  $("map-frame").classList.remove("is-thinking");
  $("map-frame").classList.add("is-lighting");
  renderMap();
  setTimeout(() => $("map-frame").classList.remove("is-lighting"), 2200);
  results.hidden = false;
  renderCards();
  renderPlane();
  say(`Listo. Tu top 5: ${data.top.map((t) => t.town_display).join(", ")}. `
    + `Los que evitarías: ${data.bottom.map((t) => t.town_display).join(", ")}.`);
  if (narrow()) document.querySelector(".map").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
}

/* ---------- arranque ---------- */
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = query.value.trim();
  if (text.length < 3) {
    showAlert("<b>Falta tu consulta.</b> Cuenta en unas palabras qué buscas en el viaje.");
    query.focus();
    return;
  }
  recommend(text);
});
query.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) form.requestSubmit();
});

let resizeTimer = 0, lastWidth = window.innerWidth;
window.addEventListener("resize", () => {
  if (window.innerWidth === lastWidth) return;      // en móvil, la barra del navegador cambia solo el alto
  lastWidth = window.innerWidth;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { renderMap(); renderPlane(); }, 150);
});

async function loadMap() {
  try {
    const [geo, towns] = await Promise.all([
      fetch("/static/data/mexico.geojson").then((r) => r.json()),
      fetch("/towns").then((r) => (r.ok ? r.json() : { towns: [] })),
    ]);
    state.geo = geo;
    state.towns = (towns.towns || []).filter((t) => t.lat != null && t.lon != null);
    state.byTown = new Map((towns.towns || []).map((t) => [t.town, t]));
  } catch {
    $("map-help").textContent = "No se pudo cargar el mapa. Las recomendaciones siguen funcionando.";
  }
  renderMap();
}

setupMapInteraction();
setupPlaneInteraction();
checkHealth();
loadMap().then(() => {
  const params = new URLSearchParams(location.search), preset = params.get("q");
  if (preset) { query.value = preset; recommend(preset); }
});
