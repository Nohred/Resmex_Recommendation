"use strict";

const $ = (id) => document.getElementById(id);
const form = $("form"), query = $("query"), submit = $("submit");
const alertBox = $("alert"), loading = $("loading"), results = $("results");

const FALLBACK_EXAMPLES = [
  "Quiero un lugar tranquilo con zonas arqueológicas y buena comida tradicional",
  "Busco fiesta, playa y vida nocturna",
  "Me interesan pueblos coloniales con artesanías y clima frío",
  "Naturaleza, cascadas y aventura lejos de las multitudes",
];
const PROVIDER_NAMES = { gemini: "Gemini", groq: "Groq", ollama: "Modelo local (Ollama)", plantilla: "Plantilla sin LLM" };
const LOADING_STEPS = [
  "Imaginando tu viaje ideal…",
  "Escribiendo también tu viaje de pesadilla…",
  "Comparando con los arquetipos de cada pueblo…",
];

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => Number(x).toFixed(d);
const ms = (x) => (x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${x < 10 ? x.toFixed(1) : Math.round(x)} ms`);
const stars = (p) => "★".repeat(Math.round(p)) + "☆".repeat(5 - Math.round(p));

function showAlert(html, kind = "error") {
  alertBox.className = "alert" + (kind === "warn" ? " warn" : "");
  alertBox.innerHTML = html;
  alertBox.hidden = false;
}

function renderExamples(list) {
  $("examples").innerHTML = "";
  list.forEach((text) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", () => { query.value = text; query.focus(); });
    $("examples").appendChild(b);
  });
}

async function checkHealth() {
  const el = $("status"), label = el.querySelector("span");
  try {
    const h = await (await fetch("/health")).json();
    renderExamples(h.examples?.length ? h.examples : FALLBACK_EXAMPLES);
    if (h.index?.loaded) {
      $("stat-towns").textContent = h.index.towns;
      $("stat-arch").textContent = h.index.archetypes;
    }
    const first = h.llm?.chain?.[0];
    if (!h.index?.loaded) {
      el.className = "status bad";
      label.textContent = "Índice no cargado (ejecuta make index)";
    } else if (!h.ollama?.ok || !h.ollama?.embedding_model) {
      el.className = "status bad";
      label.textContent = "Ollama o el modelo de embeddings no están disponibles";
    } else if (first) {
      const active = h.llm.chain.find((p) => p.available) || first;
      el.className = "status " + (active === first && first.available ? "ok" : "warn");
      label.textContent = `Listo · LLM: ${PROVIDER_NAMES[active.provider] || active.provider} (${active.model})`
        + (active === first ? "" : " · respaldo");
    } else {
      el.className = "status warn";
      label.textContent = "Listo · sin LLM (se usará la plantilla)";
    }
  } catch {
    renderExamples(FALLBACK_EXAMPLES);
    el.className = "status bad";
    label.textContent = "No se pudo consultar el estado del servidor";
  }
}

function reviewHtml(t) {
  const text = t.review || "";
  const short = text.length > 300 ? text.slice(0, 300).replace(/\s+\S*$/, "") + "…" : text;
  const more = text.length > 300
    ? ` <button type="button" class="more" data-full="${esc(text)}">ver completa</button>` : "";
  return `<blockquote><span class="rtext">“${esc(short)}”</span>${more}
    <footer><span class="stars" aria-label="${t.review_polarity} de 5">${stars(t.review_polarity)}</span>
    · reseña representativa del arquetipo</footer></blockquote>`;
}

function cardHtml(t, kind, lo, hi) {
  const isTop = kind === "top";
  const value = isTop ? t.score : t.sim_anti;
  const width = Math.max(8, Math.round(100 * (value - lo) / Math.max(hi - lo, 1e-9)));
  const kws = (t.keywords || []).slice(0, 6).map((k) => `<span class="kw">${esc(k)}</span>`).join("");
  const numbers = isTop
    ? `Similitud HyDE ${fmt(t.sim_hyde)} − penalización ${fmt(t.penalty)} = <b>${fmt(t.score)}</b> · ${t.n_reviews.toLocaleString("es-MX")} reseñas`
    : `Similitud Anti-HyDE <b>${fmt(t.sim_anti)}</b> · similitud HyDE ${fmt(t.sim_hyde)} · ${t.n_reviews.toLocaleString("es-MX")} reseñas`;
  return `<article class="town card">
    <div class="rank" aria-label="Lugar ${t.rank}">${t.rank}</div>
    <header>
      <h3>${esc(t.town_display)}</h3><span class="region">${esc(t.region)}</span>
      <div class="score"><b>${fmt(value)}</b><small>${isTop ? "puntaje ajustado" : "parecido a tu pesadilla"}</small></div>
    </header>
    <div class="bar" aria-hidden="true"><i style="width:${width}%"></i></div>
    <div class="body">
      <div class="why"><span class="tag">${esc(t.archetype_label)}</span>${kws}</div>
      ${reviewHtml(t)}
      <p class="numbers">${numbers}</p>
    </div>
  </article>`;
}

function render(data) {
  // Aviso de proveedor / fallback
  const p = data.llm_provider;
  if (data.fallback) {
    showAlert("<b>Ningún modelo de lenguaje respondió.</b> Se usó tu consulta tal cual como HyDE y una plantilla de antónimos como Anti-HyDE; los resultados pueden ser menos precisos.", "warn");
  } else if (data.degraded) {
    showAlert(`<b>El modelo principal no respondió.</b> Las reseñas hipotéticas las generó ${esc(PROVIDER_NAMES[p] || p)} (${esc(data.llm_model)}).`, "warn");
  }
  const cls = (p === "gemini" || p === "groq") ? "" : (p === "ollama" ? " local" : " template");
  const t = data.timings_ms || {};
  $("meta").innerHTML = `
    <span class="pill provider${cls}">Generado por <b>${esc(PROVIDER_NAMES[p] || p)}</b>${data.llm_model ? " · " + esc(data.llm_model) : ""}</span>
    <span class="pill">LLM <b>${data.cached ? "caché" : ms(t.llm)}</b></span>
    <span class="pill">Embeddings <b>${ms(t.embedding)}</b></span>
    <span class="pill">Búsqueda <b>${ms(t.busqueda)}</b></span>
    <span class="pill">λ = <b>${data.lambda}</b></span>`;

  const range = (arr, key) => {
    const v = arr.map((x) => x[key]);
    const hi = Math.max(...v), lo = Math.min(...v);
    return [lo - (hi - lo) * 0.6 - 0.01, hi];
  };
  const [tl, th] = range(data.top, "score"), [bl, bh] = range(data.bottom, "sim_anti");
  $("top").innerHTML = data.top.map((x) => cardHtml(x, "top", tl, th)).join("");
  $("bottom").innerHTML = data.bottom.map((x) => cardHtml(x, "bottom", bl, bh)).join("");
  $("hyde").textContent = data.hyde;
  $("anti").textContent = data.anti_hyde;
  $("texts-provider").textContent = `(${PROVIDER_NAMES[p] || p})`;
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function recommend(text) {
  alertBox.hidden = true;
  results.hidden = true;
  loading.hidden = false;
  submit.disabled = true;
  let step = 0;
  $("loading-step").textContent = LOADING_STEPS[0];
  const timer = setInterval(() => {
    step = Math.min(step + 1, LOADING_STEPS.length - 1);
    $("loading-step").textContent = LOADING_STEPS[step];
  }, 2500);
  try {
    const resp = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
      throw new Error(detail || `Error ${resp.status}`);
    }
    render(data);
  } catch (err) {
    const msg = err instanceof TypeError ? "No se pudo conectar con el servidor. ¿Sigue activo?" : err.message;
    showAlert(`<b>No se pudo obtener la recomendación.</b> ${esc(msg)}`);
  } finally {
    clearInterval(timer);
    loading.hidden = true;
    submit.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = query.value.trim();
  if (text.length < 3) { showAlert("Escribe qué buscas en tu viaje (al menos unas palabras)."); return; }
  recommend(text);
});
query.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) form.requestSubmit();
});
document.addEventListener("click", (e) => {
  const b = e.target.closest(".more");
  if (!b) return;
  b.parentElement.querySelector(".rtext").textContent = `“${b.dataset.full}”`;
  b.remove();
});

checkHealth();
const preset = new URLSearchParams(location.search).get("q");
if (preset) { query.value = preset; recommend(preset); }
