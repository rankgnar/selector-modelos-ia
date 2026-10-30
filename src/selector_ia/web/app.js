// Selector de Modelos IA — page controller. Search logic lives in finder.js; this file only renders.
// With the local server (/api/...) everything is live. Opened without it (the design mock-up), it reads
// snapshot.json and «Mis equipos» / «Probar» use clearly labelled sample data.
import {
  CAPABILITIES, WHERE, emptyQuery, search, reason, prosCons, isFree, requirementsFor, sortsFor, capLabel, reqLabel,
  harnessSteps, fmtNumber, fmtUsd, fmtTokens, fmtParams, fmtDate,
} from "./finder.js";

const PRICE_STEPS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50];
const PAGE = 60;
const BENCH_SHORT = {
  intelligence: "Inteligencia", coding: "Programación", agentic: "Agentes", asr_wer: "Error (WER)",
  arena_image: "ELO imagen", arena_edit: "ELO edición",
};
const SORT_RANK = {
  quality: "Mejor calidad", value: "Mejor calidad-precio", cheap: "Más barato", popular: "Más usado",
  new: "Más nuevo", light: "Más ligero",
};

const state = {
  mode: "live",            // "live" (server) | "mockup" (snapshot.json, no server)
  records: [],
  meta: {},
  query: emptyQuery(),
  fit: null,               // id -> {verdict, need_gb, text}
  profiles: [],
  templates: [],
  profile: null,
  harnesses: {},           // id -> agent (harnesses.json, from official docs)
  harnessMeta: {},
  status: null,            // /api/status: freshness of the catalog
  checks: {},              // "profile|model" -> exact check (/api/check)
  extra: new Map(),        // models found by the live search in all of Hugging Face (id -> record)
  live: { text: "", busy: false, added: 0, error: null },
  priceStep: 4,
  shown: PAGE,
  result: null,
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ICON = {
  check: '<svg viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  minus: '<svg viewBox="0 0 24 24"><path d="M5 12h14"/></svg>',
  copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
  link: '<svg viewBox="0 0 24 24"><path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="m6 4 14 8-14 8z"/></svg>',
  star: '<svg viewBox="0 0 24 24"><path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/></svg>',
};

// --- data access ---

const api = {
  async get(path) {
    const res = await fetch(path, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `Error ${res.status}`);
    return res.json();
  },
  async post(path, body) {
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Error ${res.status}`);
    return data;
  },
};

async function loadCatalog() {
  try {
    const status = await api.get("/api/status");
    state.mode = "live";
    while (!status.ready) {
      $("#count").textContent = "Preparando el catálogo…";
      $("#method").textContent = status.progress || "Descargando modelos de OpenRouter y Hugging Face (solo la primera vez).";
      await new Promise((r) => setTimeout(r, 700));
      Object.assign(status, await api.get("/api/status"));
      if (status.error) throw new Error(status.error);
    }
    state.status = status;
    const data = await api.get("/api/catalog");
    state.records = data.records; state.meta = data;
    const eq = await api.get("/api/profiles");
    state.profiles = eq.profiles; state.templates = eq.templates;
  } catch (err) {
    if (state.mode === "live" && !(err instanceof TypeError) && !String(err.message).startsWith("Error 404")) {
      showFatal(err.message);
      return false;
    }
    const data = await (await fetch("snapshot.json")).json();
    state.mode = "mockup"; state.records = data.records; state.meta = data;
    state.profiles = [SAMPLE_PROFILE];
  }
  await loadHarnesses();
  let saved = null;
  try { saved = localStorage.getItem("profile"); } catch (e) { /* ignore */ }
  state.profile = state.profiles.find((p) => p.name === saved) || state.profiles.find((p) => p.kind === "local")
    || state.profiles[0] || null;
  await loadFit();
  return true;
}

async function loadHarnesses() {
  try {
    const data = await (await fetch("harnesses.json")).json();
    state.harnessMeta = data;
    state.harnesses = Object.fromEntries(data.harnesses.map((h) => [h.id, h]));
  } catch (e) { state.harnesses = {}; }
}

// Sample PC for the mock-up: the teacher's own computer (docs/ESTADO.md), labelled as an example.
const SAMPLE_PROFILE = { name: "Mi PC (ejemplo)", kind: "local", machine: "pc", ram_gb: 22.8, gpus: [], sample: true,
                         summary: "22,8 GB de RAM · GPU integrada (sin memoria propia)" };

function sampleFit(records, profile) {
  // Same rule as hardware/fit.py: memory ≈ weights × 1.15; Q4_K_M = 0.614 bytes/param for LLMs
  // (measured on unsloth/Qwen3-8B-GGUF), F16 = 2 bytes for image/video/voice; 80 % of RAM usable.
  const usable = profile.ram_gb * 0.8;
  const fit = {};
  for (const r of records) {
    if (!r.hf_id) continue;
    if (!r.params) { fit[r.id] = { verdict: "unknown", text: "Sin datos de tamaño" }; continue; }
    const llm = (r.capabilities || []).some((c) => ["chat", "code", "vision"].includes(c));
    const need = (r.params * (llm ? 0.614 : 2.0) * 1.15) / 1073741824;
    const verdict = need <= 2 ? "ok" : need <= usable ? "limited" : "no";
    const text = verdict === "ok" ? `Cabe y va bien: ~${fmtNumber(need, 1)} GB`
      : verdict === "limited" ? `Cabe en la RAM (~${fmtNumber(need, 1)} GB de ${fmtNumber(usable, 1)}), lento sin GPU`
      : `No cabe: necesita ~${fmtNumber(need, 1)} GB y hay ~${fmtNumber(usable, 1)} utilizables`;
    fit[r.id] = { verdict, need_gb: need, text, estimate: true };
  }
  return fit;
}

async function loadFit() {
  if (!state.profile) { state.fit = null; return; }
  if (state.mode === "mockup") { state.fit = sampleFit(state.records, state.profile); return; }
  state.fit = (await api.get(`/api/fit?profile=${encodeURIComponent(state.profile.name)}`)).fit;
}

// --- persistence (per-viewer convenience only) ---

// The query also lives in the address (#caps=pdf,tools&where=openrouter…) so a teacher can share a link
// with the filters already ticked.
function queryToHash(q) {
  const p = new URLSearchParams();
  if (q.text) p.set("q", q.text);
  if (q.caps.length) p.set("caps", q.caps.join(","));
  if (q.reqs.length) p.set("reqs", q.reqs.join(","));
  if (q.where !== "openrouter") p.set("where", q.where);
  if (q.sort !== "quality") p.set("sort", q.sort);
  if (q.maxPrice !== null) p.set("max", String(q.maxPrice));
  if (q.includeOld) p.set("old", "1");
  if (!q.onlyFits) p.set("fits", "0");
  if (q.harness) p.set("app", q.harness);
  if (!q.harnessStrict) p.set("all", "1");
  if (q.community) p.set("com", "1");
  return p.toString();
}

function hashToQuery(hash) {
  const p = new URLSearchParams(hash.replace(/^#/, ""));
  const list = (k) => (p.get(k) || "").split(",").filter(Boolean);
  return { ...emptyQuery(), text: p.get("q") || "", caps: list("caps"), reqs: list("reqs"),
           where: p.get("where") || "openrouter", sort: p.get("sort") || "quality",
           maxPrice: p.has("max") ? Number(p.get("max")) : null, includeOld: p.get("old") === "1",
           onlyFits: p.get("fits") !== "0", harness: p.get("app") || null, harnessStrict: p.get("all") !== "1",
           community: p.get("com") === "1" };
}

function saveQuery() {
  const hash = queryToHash(state.query);
  history.replaceState(null, "", hash ? `#${hash}` : location.pathname);
  try { localStorage.setItem("query", JSON.stringify({ ...state.query, text: "" })); } catch (e) { /* private mode */ }
}

function restoreQuery() {
  if (location.hash.length > 1) { state.query = hashToQuery(location.hash); return; }
  try {
    const q = JSON.parse(localStorage.getItem("query") || "null");
    if (q) state.query = { ...emptyQuery(), ...q, text: "" };
  } catch (e) { /* ignore */ }
}

// --- rendering ---

function update({ keepPage = false } = {}) {
  if (!keepPage) state.shown = PAGE;
  const q = state.query;
  const records = q.text && state.extra.size ? state.records.concat([...state.extra.values()]) : state.records;
  state.result = search(records, q, state.fit, state.harnesses);
  renderSidebar(); renderHeader(); renderFreshness(); renderChips(); renderTop(); renderList(); renderFooter();
  saveQuery();
}

function opt({ id, label, help, checked, count, disabled }) {
  return `<label class="opt${disabled ? " disabled" : ""}" title="${esc(help || "")}">
    <input type="checkbox" data-id="${esc(id)}" ${checked ? "checked" : ""} ${disabled ? "disabled" : ""}>
    <span class="opt-label">${esc(label)}</span>
    ${count === null ? "" : `<span class="opt-count">${fmtNumber(count, 0)}</span>`}
  </label>`;
}

function renderSidebar() {
  const q = state.query;
  const c = state.result.counts;
  const where = WHERE.map((w) => `<button type="button" data-where="${w.key}" aria-pressed="${q.where === w.key}" title="${esc(w.help)}">
      <span class="seg-label">${esc(w.label)}</span><span class="seg-count">${fmtNumber(c.where[w.key], 0)} modelos</span></button>`).join("");
  const profileOpts = state.profiles.map((p) => `<option value="${esc(p.name)}" ${state.profile && p.name === state.profile.name ? "selected" : ""}>${esc(p.name)}</option>`).join("");
  const local = q.where === "local" ? `<div class="sub">
      ${state.profiles.length ? `<select class="select" id="profile" aria-label="Equipo">${profileOpts}</select>
        <div class="muted small-text">${esc(state.profile ? state.profile.summary || "" : "")}</div>` : ""}
      <button class="btn small" type="button" data-open="pc">${state.profiles.length ? "Gestionar equipos" : "Añadir un equipo"}</button>
      ${opt({ id: "onlyFits", label: "Solo los que caben", help: "Quita los que no caben en este equipo o cuyo tamaño no se conoce.", checked: q.onlyFits, count: null })}
    </div>` : "";
  const h = q.harness && state.harnesses[q.harness];
  const harnessOpts = [`<option value="">Cualquier programa (${fmtNumber(c.harness[""] ?? 0, 0)})</option>`]
    .concat(Object.values(state.harnesses).map((x) => `<option value="${esc(x.id)}" ${q.harness === x.id ? "selected" : ""}>${esc(x.name)} (${fmtNumber(c.harness[x.id] ?? 0, 0)})</option>`)).join("");
  const harness = q.where === "openrouter" && Object.keys(state.harnesses).length ? `<div class="sub">
      <label class="field-label" for="harness">¿Con qué programa lo vas a usar?</label>
      <select class="select" id="harness">${harnessOpts}</select>
      ${h ? `<div class="muted small-text">${esc(h.kind)} · <a href="${esc(h.docs)}" target="_blank" rel="noopener">documentación</a></div>` : ""}
      ${h && h.requires_text ? `<div class="hint-box">${esc(h.requires_text)}</div>
        ${opt({ id: "harnessAll", label: "Ver también los modelos sin garantía", help: "Modelos que su documentación no garantiza para este programa.", checked: !q.harnessStrict, count: q.harnessStrict ? c.strict : null })}` : ""}
    </div>` : "";
  const caps = CAPABILITIES.map((cap) => {
    const on = q.caps.includes(cap.key);
    return opt({ id: `cap:${cap.key}`, label: cap.label, help: cap.help, checked: on,
                 count: on ? null : c.caps[cap.key], disabled: !on && c.caps[cap.key] === 0 });
  }).join("");
  const reqs = requirementsFor(q.where).map((r) => {
    const on = q.reqs.includes(r.key);
    return opt({ id: `req:${r.key}`, label: r.label, help: r.help, checked: on,
                 count: on ? null : c.reqs[r.key], disabled: !on && c.reqs[r.key] === 0 });
  }).join("");
  const limit = q.maxPrice !== null;
  const priceBox = q.where === "openrouter" ? `<div class="group">
      <p class="group-title">Precio en OpenRouter</p>
      ${opt({ id: "limitPrice", label: "Poner un precio máximo", help: "Precio por millón de tokens (3 de entrada por 1 de salida).", checked: limit, count: null })}
      ${limit ? `<div class="sub"><input class="range" id="price" type="range" min="0" max="${PRICE_STEPS.length - 1}" step="1" value="${state.priceStep}">
        <div class="range-row"><span>Hasta</span><b>${fmtUsd(PRICE_STEPS[state.priceStep])} / millón</b></div></div>` : ""}
    </div>` : "";
  $("#sidebar").innerHTML = `
    <div class="group"><p class="group-title">Dónde lo vas a usar</p><div class="seg">${where}</div>${local}${harness}</div>
    <div class="group"><p class="group-title">Qué tiene que hacer</p>
      <p class="group-hint">Marca todo lo que necesites: solo salen los que lo cumplen todo.</p>${caps}</div>
    <div class="group"><p class="group-title">Requisitos</p>${reqs}</div>
    ${priceBox}
    <div class="group"><p class="group-title">Más opciones</p>
      ${opt({ id: "includeOld", label: "Incluir modelos antiguos", help: "Publicados y sin actualizar desde hace más de 2 años.", checked: q.includeOld, count: q.includeOld ? null : c.old })}
      ${opt({ id: "community", label: "Incluir versiones de la comunidad", help: "Ajustes, mezclas y versiones de un modelo hechas por otras personas (no por su autor original). Calidad variable.", checked: q.community, count: q.community ? null : c.community })}
    </div>`;
}

function renderHeader() {
  const q = state.query; const r = state.result;
  $("#count").textContent = r.count === 1 ? "1 modelo" : `${fmtNumber(r.count, 0)} modelos`;
  $("#method").textContent = r.method;
  const sel = $("#sort");
  const sorts = sortsFor(q.where);
  if (!sorts.some((s) => s.key === q.sort)) q.sort = "quality";
  sel.innerHTML = sorts.map((s) => `<option value="${s.key}" ${s.key === q.sort ? "selected" : ""}>${esc(s.label)}</option>`).join("");
  const notice = [];
  if (state.mode === "mockup") {
    notice.push(`<div class="notice"><span><b>Maqueta</b> con datos reales de OpenRouter y Hugging Face
      (${fmtDate(state.meta.openrouter_date)}). «Mis equipos» usa un equipo de ejemplo y «Probar» no envía nada.</span></div>`);
  }
  if (q.reqs.includes("uncensored")) {
    const adult = state.result.results.some((s) => s.rec.adult);
    notice.push(`<div class="notice"><span><b>Sin censura:</b> casi siempre son versiones de la comunidad de un modelo
      original: la calidad varía y se aplica la licencia del original. Úsalos con responsabilidad.${adult
        ? " Algunos de esta lista están marcados por su autor como <b>contenido adulto (+18)</b>." : ""}</span></div>`);
  } else if (q.community) {
    notice.push(`<div class="notice"><span><b>Versiones de la comunidad:</b> hechas por otras personas a partir de
      un modelo original (ajustes, mezclas). No son oficiales y la calidad varía.</span></div>`);
  }
  $("#notice").innerHTML = notice.join("");
  const lv = state.live;
  if (q.text && state.mode === "live") {
    $("#method").textContent += lv.busy ? " · Buscando también en todo Hugging Face…"
      : lv.error ? ` · No se ha podido buscar en Hugging Face: ${lv.error}`
      : lv.text === q.text ? ` · Incluye ${fmtNumber(lv.added, 0)} modelos más encontrados en todo Hugging Face.` : "";
  }
}

function ago(iso) {
  if (!iso) return "sin fecha";
  const min = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 60000));
  if (min < 1) return "ahora mismo";
  if (min < 60) return `hace ${min} min`;
  const h = Math.round(min / 60);
  return h < 48 ? `hace ${h} h` : `el ${fmtDate(iso)}`;
}

function renderFreshness() {
  const el = $("#freshness");
  if (state.mode !== "live" || !state.status) { el.hidden = true; return; }
  const s = state.status;
  el.hidden = false;
  el.classList.toggle("busy", Boolean(s.building));
  el.classList.toggle("error", Boolean(s.error) && !s.building);
  el.innerHTML = `<span class="dot"></span>${s.building ? esc(s.progress || "Actualizando datos…")
    : `Datos ${esc(ago(s.built_at))}`}`;
  el.title = s.error ? `No se han podido actualizar: ${s.error}`
    : `Catálogo de OpenRouter y Hugging Face. Se actualiza solo cada ${fmtNumber(s.refresh_hours, 0)} h. Pulsa para actualizar ahora.`;
}

function renderChips() {
  const q = state.query;
  const chips = [];
  if (q.text) chips.push({ key: "text", label: `«${q.text}»` });
  for (const c of q.caps) chips.push({ key: `cap:${c}`, label: capLabel(c) });
  const shown = new Set(requirementsFor(q.where).map((r) => r.key));
  for (const r of q.reqs) if (shown.has(r)) chips.push({ key: `req:${r}`, label: reqLabel(r) });
  if (q.maxPrice !== null && q.where === "openrouter") chips.push({ key: "price", label: `Hasta ${fmtUsd(q.maxPrice)} / millón` });
  if (q.includeOld) chips.push({ key: "includeOld", label: "Incluye antiguos" });
  if (q.community) chips.push({ key: "community", label: "Incluye versiones de la comunidad" });
  if (q.harness && q.where === "openrouter" && state.harnesses[q.harness]) {
    chips.push({ key: "harness", label: `Para ${state.harnesses[q.harness].name}` });
  }
  $("#chips").innerHTML = chips.map((c) => `<button class="chip" type="button" data-chip="${esc(c.key)}" title="Quitar">${esc(c.label)} ${ICON.x}</button>`).join("")
    + (chips.length > 1 ? `<button class="chip clear" type="button" data-chip="all">Limpiar todo</button>` : "");
}

function qualityShort(s) {
  if (!s.parts.length) return `<span class="muted" title="Sin benchmark publicado para lo que has marcado">—</span>`;
  return s.parts.map((p) => `<span title="${esc(BENCH_SHORT[p.key])}">${esc(fmtNumber(p.value))}${p.key === "asr_wer" ? " %" : ""}</span>
    <span class="muted">${esc(BENCH_SHORT[p.key])}</span>`).join(" · ");
}

const per = (s) => (s.price.unit.startsWith("por segundo") ? "/s" : "/M");

function priceShort(s) {
  if (state.query.where === "local") return s.rec.params ? fmtParams(s.rec.params) : `<span class="muted">sin datos</span>`;
  if (s.price.value === null) return `<span class="muted">${s.rec.openrouter ? "sin datos" : "—"}</span>`;
  const free = isFree(s.rec) && s.price.value > 0 ? ` <span class="badge ok">gratis</span>` : "";
  return `${esc(fmtUsd(s.price.value))}<span class="muted"> ${per(s)}</span>${free}`;
}

function fitBadge(rec) {
  const f = state.fit && state.fit[rec.id];
  if (!rec.hf_id) return `<span class="muted">—</span>`;
  if (!f || f.verdict === "unknown") return `<span class="badge" title="${esc(f ? f.text : "Sin datos")}">¿?</span>`;
  const cls = { ok: "ok", limited: "warn", no: "bad" }[f.verdict];
  const text = { ok: "Cabe", limited: "Lento", no: "No cabe" }[f.verdict];
  return `<span class="badge ${cls}" title="${esc(f.text)}${f.estimate ? " (estimación)" : ""}">${text}</span>`;
}

function whereShort(rec) {
  const out = [];
  if (rec.openrouter) out.push("OpenRouter");
  if ((rec.providers || []).length) out.push(rec.providers.length > 1 ? `${rec.providers[0]} +${rec.providers.length - 1}` : rec.providers[0]);
  if (rec.hf_id) out.push("Descargable");
  return out.map((x) => `<span class="badge">${esc(x)}</span>`).join(" ");
}

function ticks(rec) {
  return state.query.caps.map((c) => `<span class="badge ok" title="${esc(capLabel(c))}">${ICON.check}${esc(capLabel(c))}</span>`).join("");
}

function renderTop() {
  const r = state.result; const q = state.query;
  if (!r.count) {
    const s = r.suggestion;
    $("#top").innerHTML = `<div class="empty" style="grid-column: 1 / -1">
      <h2>Ningún modelo cumple todo lo que has marcado</h2>
      <p>${s ? `${esc(s.label)}: tendrás ${fmtNumber(s.n, 0)} modelos.` : "Prueba a quitar algún filtro."}</p>
      ${s ? `<button class="btn primary" type="button" data-suggest="1">${esc(s.label)}</button>` : ""}</div>`;
    return;
  }
  $("#top").innerHTML = r.results.slice(0, 3).map((s, i) => {
    const ctx = s.rec.openrouter && s.rec.openrouter.context;
    const third = q.where === "local"
      ? `<div><div class="metric-label">¿Cabe?</div><div class="metric-value">${fitBadge(s.rec)}</div></div>`
      : `<div><div class="metric-label">Contexto</div><div class="metric-value">${ctx ? fmtTokens(ctx) : "—"}</div></div>`;
    return `<article class="card${i === 0 ? " first" : ""}" data-open-model="${esc(s.rec.id)}" tabindex="0">
      <div class="rank">${i === 0 ? ICON.star : ""}${i + 1} · ${esc(q.sort === "quality" && !s.parts.length ? "Destacado" : SORT_RANK[q.sort])}</div>
      <div><div class="card-name">${esc(s.rec.name)}</div><div class="card-author">${esc(s.rec.author)}${s.rec.community ? ' · <span class="badge">comunidad</span>' : ""}${s.rec.adult ? ' <span class="badge bad">+18</span>' : ""}</div></div>
      <div class="card-reason">${esc(reason(s, q, r.count))}</div>
      ${q.caps.length ? `<div class="model-sub">${ticks(s.rec)}</div>` : ""}
      <div class="metrics">
        <div><div class="metric-label">Calidad</div><div class="metric-value">${s.parts.length ? s.parts.map((p) => fmtNumber(p.value) + (p.key === "asr_wer" ? " %" : "")).join(" · ") : "—"}</div></div>
        <div><div class="metric-label">${q.where === "local" ? "Tamaño" : "Precio"}</div><div class="metric-value">${q.where === "local" ? (s.rec.params ? fmtParams(s.rec.params) : "—") : (s.rec.openrouter ? (s.price.value === null ? "sin datos" : `${fmtUsd(s.price.value)} ${per(s)}`) : "—")}</div></div>
        ${third}
      </div></article>`;
  }).join("");
}

function renderList() {
  const r = state.result; const q = state.query;
  const list = $("#list");
  if (r.count <= 3) { list.innerHTML = ""; list.hidden = true; return; }
  list.hidden = false;
  const withFit = q.where === "local";
  list.classList.toggle("with-fit", withFit);
  const head = `<div class="row head"><span>#</span><span>Modelo</span><span>Calidad</span>
    <span>${withFit ? "Tamaño" : "Precio"}</span><span class="hide-md">Contexto</span><span class="hide-md">Licencia</span>
    ${withFit ? "<span>¿Cabe?</span>" : ""}<span class="hide-md">Dónde</span></div>`;
  const rows = r.results.slice(0, state.shown).map((s) => {
    const rec = s.rec;
    const ctx = rec.openrouter && rec.openrouter.context;
    const lic = rec.hf_id ? (rec.license || "sin datos") : "API";
    return `<div class="row" data-open-model="${esc(rec.id)}" tabindex="0">
      <span class="pos">${s.position}</span>
      <div style="min-width:0"><div class="model-name">${esc(rec.name)}</div>
        <div class="model-sub"><span>${esc(rec.author)}</span>${rec.old ? '<span class="badge warn">antiguo</span>' : ""}${rec.community ? '<span class="badge">comunidad</span>' : ""}${rec.adult ? '<span class="badge bad">+18</span>' : ""}${rec.live ? '<span class="badge accent">de Hugging Face</span>' : ""}</div></div>
      <span class="cell">${qualityShort(s)}</span>
      <span class="cell">${priceShort(s)}</span>
      <span class="cell hide-md">${ctx ? fmtTokens(ctx) : '<span class="muted">—</span>'}</span>
      <span class="cell hide-md${rec.hf_id ? "" : " muted"}">${esc(lic)}</span>
      ${withFit ? `<span class="cell">${fitBadge(rec)}</span>` : ""}
      <span class="cell hide-md">${whereShort(rec)}</span></div>`;
  }).join("");
  const more = r.count > state.shown
    ? `<div class="more"><button class="btn" type="button" data-more="1">Mostrar más (${fmtNumber(r.count - state.shown, 0)} restantes)</button></div>` : "";
  list.innerHTML = head + rows + more;
}

function renderFooter() {
  const m = state.meta;
  $("#footer").innerHTML = `Datos: OpenRouter del ${fmtDate(m.openrouter_date)} · Hugging Face del ${fmtDate(m.hf_date)}.
    Benchmarks: Artificial Analysis y Design Arena (publicados en OpenRouter) y Open ASR Leaderboard (fichas de Hugging Face).
    Todo sale de esas fuentes; lo que falta aparece como «sin datos» o «—».`;
}

// --- detail drawer ---

function openModel(id) {
  const r = state.result;
  const s = r.results.find((x) => x.rec.id === id);
  if (!s) return;
  const rec = s.rec; const q = state.query;
  const pc = prosCons(s, q, r.results);
  const or = rec.openrouter;
  const facts = [
    ["Precio", s.price.value === null ? (or ? "sin datos" : "—") : `${fmtUsd(s.price.value)} ${s.price.unit}`],
    ["Contexto", or && or.context ? `${fmtTokens(or.context)} tokens` : "sin datos"],
    ["Tamaño", rec.params ? `${fmtParams(rec.params)} parámetros` : "sin datos"],
    ["Licencia", rec.hf_id ? (rec.license || "sin datos") : "Términos del proveedor"],
    ["Publicado", fmtDate(rec.created)],
    ["Descargas / mes", rec.downloads ? fmtNumber(rec.downloads, 0) : "sin datos"],
  ];
  const places = [];
  const skus = or && or.video_skus && Object.keys(or.video_skus).length
    ? `<section><p class="section-title">Precios de vídeo publicados por OpenRouter</p>
        <div class="fit-box"><span class="code">${esc(Object.entries(or.video_skus).map(([k, v]) => `${k}: ${v}`).join(" · "))}</span>
        <div class="muted">Copiados tal cual. OpenRouter no documenta sus unidades («pueden variar según el proveedor»):
        confírmalos en <a href="${esc(or.url)}" target="_blank" rel="noopener">su ficha</a>.</div></div></section>` : "";
  if (or) {
    places.push(place("OpenRouter", `<span class="code">${esc(or.id)}</span>${or.free_id ? ` · gratis: <span class="code">${esc(or.free_id)}</span>` : ""}`,
      `<button class="btn small" data-copy="${esc(or.id)}">${ICON.copy}Copiar</button><a class="btn small" href="${esc(or.url)}" target="_blank" rel="noopener">${ICON.link}Abrir</a>`));
  }
  if ((rec.providers || []).length) {
    places.push(place("Otros proveedores", esc(rec.providers.join(", ")) + " (según Hugging Face)",
      `<a class="btn small" href="https://huggingface.co/${esc(rec.hf_id)}" target="_blank" rel="noopener">${ICON.link}Ver</a>`));
  }
  if (rec.hf_id) {
    places.push(place("En tu equipo", `Descarga desde Hugging Face: <span class="code">${esc(rec.hf_id)}</span>`,
      `<button class="btn small" data-copy="${esc(rec.hf_id)}">${ICON.copy}Copiar</button><a class="btn small" href="https://huggingface.co/${esc(rec.hf_id)}" target="_blank" rel="noopener">${ICON.link}Ficha</a>`));
  }
  const f = state.fit && state.fit[rec.id];
  const fitHtml = rec.hf_id && state.profile ? `<section><p class="section-title">En ${esc(state.profile.name)}</p>
      <div class="fit-box">${f ? `${fitBadge(rec)} ${esc(f.text)}` : "Sin datos de tamaño"}
      <div class="muted">${state.mode === "mockup" ? "Estimación de ejemplo: memoria ≈ parámetros × bytes por parámetro × 1,15 (Q4 en modelos de texto, F16 en imagen, vídeo y voz). La app comprobará los archivos reales y te dará el comando de Ollama o LM Studio."
        : "Estimación por el número de parámetros: memoria ≈ pesos × 1,15."}</div></div>
      ${state.mode === "live" ? `<div id="exact-check" class="exact"><span class="muted">Comprobando con los archivos reales…</span></div>` : ""}</section>` : "";
  const canTry = or && (rec.capabilities || []).includes("chat");
  const harnessIds = Object.keys(state.harnesses);
  const chosen = q.harness && state.harnesses[q.harness] ? q.harness : "openai-compatible";
  const harnessHtml = canTry && harnessIds.length ? `<section><p class="section-title">Cómo usarlo con tu programa</p>
      <select class="select" id="drawer-harness" aria-label="Programa">${Object.values(state.harnesses).map((h) =>
        `<option value="${esc(h.id)}" ${h.id === chosen ? "selected" : ""}>${esc(h.name)}</option>`).join("")}</select>
      <div id="harness-steps" class="steps"></div></section>` : "";
  $("#drawer").innerHTML = `
    <div class="drawer-head"><div style="min-width:0"><h2>${esc(rec.name)}</h2><div class="mono">${esc(rec.id)}</div></div>
      <button class="btn icon" type="button" data-close="1" aria-label="Cerrar">${ICON.x}</button></div>
    <div class="drawer-body">
      <section><p class="section-title">Por qué está aquí</p><div>${esc(reason(s, q, r.count))}</div>
        <div class="model-sub" style="margin-top:8px">${ticks(rec)}${(rec.capabilities || []).filter((c) => !q.caps.includes(c)).map((c) => `<span class="badge">${esc(capLabel(c))}</span>`).join("")}</div></section>
      ${""}<section class="facts">${facts.map(([k, v]) => `<div class="fact"><div class="metric-label">${esc(k)}</div><div class="metric-value">${esc(v)}</div></div>`).join("")}</section>
      ${skus}
      <section class="pc">
        <div><p class="section-title">A favor</p><ul>${pc.pros.map((x) => `<li class="pro">${ICON.check}<span>${esc(x)}</span></li>`).join("") || '<li class="muted">—</li>'}</ul></div>
        <div><p class="section-title">En contra</p><ul>${pc.cons.map((x) => `<li class="con">${ICON.x}<span>${esc(x)}</span></li>`).join("") || '<li class="muted">—</li>'}</ul></div>
      </section>
      ${fitHtml}
      ${harnessHtml}
      <section><p class="section-title">Dónde usarlo</p><div class="places">${places.join("")}</div></section>
      <section class="actions">
        ${canTry ? `<button class="btn primary" type="button" data-try="${esc(rec.id)}">${ICON.play}Probar</button>` : ""}
        <button class="btn" type="button" data-copy="${esc(or ? or.id : rec.id)}">${ICON.copy}Copiar nombre</button>
      </section>
    </div>`;
  $("#drawer").hidden = false; $("#scrim").hidden = false;
  $("#drawer").scrollTop = 0;
  state.openId = rec.id;
  if (harnessHtml) renderSteps(chosen, rec);
  if (fitHtml && state.mode === "live") loadCheck(rec);
}

function renderSteps(harnessId, rec) {
  const h = state.harnesses[harnessId];
  const box = $("#harness-steps");
  if (!h || !box) return;
  const warn = h.requires && h.requires.authors && h.requires.authors.length
    && !h.requires.authors.includes(rec.openrouter.id.split("/")[0])
    ? `<div class="hint-box">${esc(h.requires_text || "")}</div>` : "";
  box.innerHTML = warn + harnessSteps(h, rec).map((st, i) => `<div class="step"><div class="step-title"><span class="step-n">${i + 1}</span>${esc(st.title)}</div>
      ${st.text ? `<p>${esc(st.text)}</p>` : ""}
      ${st.code ? `<div class="codeblock"><pre>${esc(st.code)}</pre><button class="btn small" type="button" data-copy="${esc(st.code)}">${ICON.copy}Copiar</button></div>` : ""}</div>`).join("")
    + `<p class="muted small-text">Según la <a href="${esc(h.docs)}" target="_blank" rel="noopener">documentación oficial</a> (comprobado el ${fmtDate(state.harnessMeta.checked)}).
       Usa tu clave de OpenRouter en lugar de «sk-or-...».</p>`;
}

async function loadCheck(rec) {
  const key = `${state.profile.name}|${rec.id}`;
  const show = (html) => { const el = $("#exact-check"); if (el && state.openId === rec.id) el.innerHTML = html; };
  try {
    if (!state.checks[key]) state.checks[key] = await api.get(`/api/check?id=${encodeURIComponent(rec.id)}&profile=${encodeURIComponent(state.profile.name)}`);
    const c = state.checks[key];
    const cls = { ok: "ok", limited: "warn", no: "bad" }[c.verdict] || "";
    const label = { ok: "Cabe", limited: "Lento", no: "No cabe" }[c.verdict] || "¿?";
    const runners = c.runners.map((r) => `<div class="step"><div class="step-title">${esc(r.tool)}
        <a class="muted small-text" href="${esc(r.url)}" target="_blank" rel="noopener">${ICON.link}</a></div>
        <p>${esc(r.note)}</p>${r.command ? `<div class="codeblock"><pre>${esc(r.command)}</pre><button class="btn small" type="button" data-copy="${esc(r.command)}">${ICON.copy}Copiar</button></div>` : ""}</div>`).join("");
    show(`<div class="fit-box"><b>Comprobación con los archivos reales:</b> <span class="badge ${cls}">${label}</span> ${esc(c.reason)}
        ${c.variant ? `<div class="muted">Versión ${esc(c.variant)} · ${esc(c.memory_source || "")}</div>` : ""}
        ${c.notes.map((n) => `<div class="muted">${esc(n)}</div>`).join("")}</div>
      ${runners ? `<p class="section-title" style="margin-top:14px">Cómo arrancarlo</p><div class="steps">${runners}</div>` : ""}`);
  } catch (err) {
    show(`<span class="muted">No se ha podido comprobar: ${esc(err.message)}</span>`);
  }
}

function place(title, sub, actions) {
  return `<div class="place"><div class="place-main"><div class="place-title">${esc(title)}</div><div class="place-sub">${sub}</div></div>${actions}</div>`;
}

function closeDrawer() { $("#drawer").hidden = true; $("#scrim").hidden = true; }

// --- modals ---

function modal(title, body, { wide = false } = {}) {
  $("#modal").innerHTML = `<div class="modal-card${wide ? " wide" : ""}"><div class="modal-head"><h2>${esc(title)}</h2>
    <button class="btn icon" type="button" data-close-modal="1" aria-label="Cerrar">${ICON.x}</button></div>
    <div class="modal-body">${body}</div></div>`;
  $("#modal").hidden = false;
}
function closeModal() { $("#modal").hidden = true; $("#modal").innerHTML = ""; }

const TEST_PROMPT = "Explícame en 5 líneas, para alguien que empieza, qué es un modelo de lenguaje y pon un ejemplo cotidiano.";

function openTry(id) {
  const r = state.result;
  const first = r.results.find((s) => s.rec.id === id);
  const others = r.results.filter((s) => s.rec.id !== id && s.rec.openrouter && (s.rec.capabilities || []).includes("chat")).slice(0, 5);
  const options = [first, ...others].map((s, i) => `<label class="opt"><input type="checkbox" data-try-model="${esc(s.rec.openrouter.id)}" ${i < 3 ? "checked" : ""}>
    <span class="opt-label">${esc(s.rec.name)} <span class="muted">${esc(s.rec.openrouter.id)}</span></span><span class="opt-count">${fmtUsd(s.price.value)} ${per(s)}</span></label>`).join("");
  modal("Probar: mismo mensaje, respuestas lado a lado", `
    ${state.mode === "mockup" ? '<div class="notice"><span><b>Maqueta:</b> aquí no se envía nada. En la app se enviará con tu clave de OpenRouter (Ajustes) y verás tiempo, tokens y coste real.</span></div>' : ""}
    <div class="field"><label for="try-prompt">Mensaje de prueba</label><textarea class="input" id="try-prompt">${esc(TEST_PROMPT)}</textarea></div>
    <div class="field"><label>Modelos (hasta 3)</label><div>${options}</div></div>
    <div class="actions"><button class="btn primary" type="button" id="try-send" ${state.mode === "mockup" ? "disabled" : ""}>${ICON.play}Enviar</button>
      <span class="muted" id="try-status"></span></div>
    <div class="answers" id="answers"></div>`, { wide: true });
}

async function sendTry() {
  const models = [...document.querySelectorAll("[data-try-model]:checked")].map((el) => el.dataset.tryModel);
  const prompt = $("#try-prompt").value.trim();
  if (!models.length || models.length > 3) { $("#try-status").textContent = "Elige entre 1 y 3 modelos."; return; }
  $("#try-send").disabled = true;
  $("#try-status").textContent = "Enviando… cada respuesta aparece cuando llega.";
  $("#answers").innerHTML = models.map((m) => `<div class="answer" data-answer="${esc(m)}"><b>${esc(m)}</b><div class="muted">Esperando…</div></div>`).join("");
  let total = 0;
  await Promise.all(models.map(async (m) => {
    const box = document.querySelector(`[data-answer="${CSS.escape(m)}"]`);
    try {
      const a = await api.post("/api/probar", { model: m, prompt });
      total += a.cost_usd || 0;
      box.innerHTML = a.error ? `<b>${esc(m)}</b><p>${esc(a.error)}</p>`
        : `<b>${esc(m)}</b><div class="muted">${fmtNumber(a.seconds, 1)} s · ${a.prompt_tokens ?? "?"} + ${a.completion_tokens ?? "?"} tokens · ${a.cost_usd === null ? "coste sin datos" : fmtUsd(a.cost_usd)}</div><p style="white-space:pre-wrap">${esc(a.content)}</p>`;
    } catch (err) {
      box.innerHTML = `<b>${esc(m)}</b><p>${esc(err.message)}</p>`;
    }
  }));
  $("#try-send").disabled = false;
  $("#try-status").textContent = `Listo. Coste total: ${fmtUsd(total)}`;
}

const MACHINES = { pc: "PC", vps: "VPS", gpu_server: "Servidor GPU" };

function openPc(editing = null) {
  const mock = state.mode === "mockup";
  const rows = state.profiles.map((p) => `<div class="profile${state.profile && p.name === state.profile.name ? " active" : ""}">
      <div class="place-main"><div class="place-title">${esc(p.name)} <span class="badge">${esc(MACHINES[p.machine] || "Equipo")}</span>${p.detected ? ' <span class="badge accent">detectado</span>' : ""}</div>
        <div class="place-sub">${esc(p.summary || "")}</div></div>
      <button class="btn small" type="button" data-use-profile="${esc(p.name)}">Usar</button>
      ${mock || p.detected ? "" : `<button class="btn small" type="button" data-edit-profile="${esc(p.name)}">Editar</button>`}
      ${mock ? "" : `<button class="btn small" type="button" data-delete-profile="${esc(p.name)}" title="Borrar">${ICON.x}</button>`}</div>`).join("");
  const e = editing || {};
  const groups = { vps: "VPS sin GPU", pc: "PC con GPU NVIDIA", gpu_server: "Servidor GPU alquilado" };
  const templates = Object.entries(groups).map(([m, label]) => `<optgroup label="${esc(label)}">${state.templates
    .filter((x) => x.machine === m).map((x) => `<option value="${esc(x.id)}">${esc(x.label)}</option>`).join("")}</optgroup>`).join("");
  modal("Mis equipos", `
    ${mock ? '<div class="notice"><span><b>Maqueta:</b> equipo de ejemplo. En la app se detecta tu ordenador y puedes guardar otros (VPS, servidores con GPU).</span></div>' : ""}
    <p class="muted">Guarda los equipos donde quieras ejecutar modelos: tu ordenador, un VPS o un servidor con GPU. Al elegir «Mis equipos» verás solo los modelos que caben en el equipo seleccionado.</p>
    <div class="places">${rows || '<p class="muted">Todavía no hay equipos guardados.</p>'}</div>
    <div class="actions"><button class="btn" type="button" id="pc-detect" ${mock ? "disabled" : ""}>Detectar este ordenador</button></div>
    <p class="section-title">${editing ? `Editar «${esc(e.name)}»` : "Añadir un equipo"}</p>
    ${editing ? "" : `<div class="field"><label for="eq-template">Partir de una plantilla <span class="muted">(opcional, luego puedes cambiar los datos)</span></label>
      <select class="select" id="eq-template"><option value="">— Escribir los datos a mano —</option>${templates}</select></div>`}
    <div class="form-grid">
      <div class="field"><label for="eq-name">Nombre</label><input class="input" id="eq-name" value="${esc(e.name || "")}" placeholder="VPS de clase"></div>
      <div class="field"><label for="eq-machine">Tipo</label><select class="select" id="eq-machine">${Object.entries(MACHINES).map(([k, v]) =>
        `<option value="${k}" ${(e.machine || "pc") === k ? "selected" : ""}>${v}</option>`).join("")}</select></div>
      <div class="field"><label for="eq-ram">Memoria RAM (GB)</label><input class="input" id="eq-ram" type="number" min="1" step="1" value="${esc(e.ram_gb ?? "")}" placeholder="16"></div>
      <div class="field"><label for="eq-threads">Núcleos de CPU <span class="muted">(opcional)</span></label><input class="input" id="eq-threads" type="number" min="1" step="1" value="${esc(e.cpu_threads ?? "")}" placeholder="8"></div>
      <div class="field"><label for="eq-gpu">Tarjeta gráfica <span class="muted">(vacío si no tiene)</span></label><input class="input" id="eq-gpu" value="${esc(e.gpu || "")}" placeholder="NVIDIA RTX 3060"></div>
      <div class="field"><label for="eq-vram">Memoria de cada GPU (GB)</label><input class="input" id="eq-vram" type="number" min="0" step="1" value="${esc(e.vram_gb ?? "")}" placeholder="12"></div>
      <div class="field"><label for="eq-count">Número de GPUs</label><input class="input" id="eq-count" type="number" min="1" max="16" step="1" value="${esc(e.gpu_count || 1)}"></div>
    </div>
    <p class="muted small-text">¿No sabes qué tiene? En Linux: <span class="code">free -h</span> (RAM), <span class="code">nproc</span> (núcleos) y <span class="code">nvidia-smi</span> (GPU NVIDIA y su memoria). En un VPS, mira el plan contratado.</p>
    <div class="actions"><button class="btn primary" type="button" id="pc-save" data-original="${esc(e.name || "")}" ${mock ? "disabled" : ""}>${editing ? "Guardar cambios" : "Guardar equipo"}</button>
      ${editing ? '<button class="btn" type="button" data-open="pc">Cancelar</button>' : ""}<span class="muted" id="pc-status"></span></div>`);
}

function applyTemplate(id) {
  const tpl = state.templates.find((x) => x.id === id);
  if (!tpl) return;
  $("#eq-name").value = tpl.label; $("#eq-machine").value = tpl.machine; $("#eq-ram").value = tpl.ram_gb;
  $("#eq-gpu").value = tpl.gpu || ""; $("#eq-vram").value = tpl.vram_gb ?? ""; $("#eq-count").value = tpl.gpu_count || 1;
}

async function savePc(original) {
  const num = (id) => ($(id).value === "" ? null : Number($(id).value));
  const body = { name: $("#eq-name").value.trim(), machine: $("#eq-machine").value, ram_gb: num("#eq-ram"),
                 cpu_threads: num("#eq-threads"), gpu: $("#eq-gpu").value.trim(), vram_gb: num("#eq-vram"),
                 gpu_count: num("#eq-count") || 1, original_name: original || null };
  try {
    const res = await api.post("/api/profiles", body);
    state.profiles = res.profiles;
    await useProfile(res.saved);
    openPc(); toast("Equipo guardado.");
  } catch (err) { $("#pc-status").textContent = err.message; }
}

async function deletePc(name) {
  try {
    state.profiles = (await api.post("/api/profiles/delete", { name })).profiles;
    if (state.profile && state.profile.name === name) await useProfile(state.profiles[0] ? state.profiles[0].name : null);
    openPc();
  } catch (err) { toast(err.message); }
}

async function detectPc() {
  $("#pc-status").textContent = "Detectando…";
  try {
    const res = await api.post("/api/profiles/detect", {});
    state.profiles = res.profiles;
    await useProfile(res.saved);
    openPc();
  } catch (err) { $("#pc-status").textContent = err.message; }
}

// Typing a name also searches all of Hugging Face (beyond the catalog), e.g. «abliterated», «dolphin».
let liveTimer = null;
function liveSearch(text) {
  clearTimeout(liveTimer);
  if (state.mode !== "live" || text.length < 3) return;
  liveTimer = setTimeout(async () => {
    state.live = { text, busy: true, added: 0, error: null };
    renderHeader();
    try {
      const prof = state.profile ? `&profile=${encodeURIComponent(state.profile.name)}` : "";
      const res = await api.get(`/api/hfsearch?q=${encodeURIComponent(text)}${prof}`);
      let added = 0;
      for (const r of res.records) {
        if (!state.extra.has(r.id)) { state.extra.set(r.id, { ...r, live: true }); added += 1; }
      }
      state.fit = { ...(state.fit || {}), ...res.fit };
      state.live = { text, busy: false, added: res.records.length, error: null };
    } catch (err) {
      state.live = { text, busy: false, added: 0, error: err.message };
    }
    if (state.query.text === text) update({ keepPage: true });
  }, 450);
}

async function useProfile(name) {
  state.profile = state.profiles.find((p) => p.name === name) || null;
  try { if (name) localStorage.setItem("profile", name); } catch (e) { /* ignore */ }
  await loadFit();
  state.extra.clear();
  $("#pc-label").textContent = state.profile ? state.profile.name : "Mis equipos";
  update();
  if (state.query.text) liveSearch(state.query.text);
}

async function openSettings() {
  let s = { openrouter_key: false, hf_token: false, refresh_hours: 6, version: {} };
  if (state.mode === "live") s = await api.get("/api/settings");
  const mock = state.mode === "mockup";
  const v = s.version || {};
  modal("Ajustes", `
    ${mock ? '<div class="notice"><span><b>Maqueta:</b> los ajustes se guardan en la app, no aquí.</span></div>' : ""}
    <div class="field"><label for="set-or">Clave de OpenRouter <span class="muted">(solo para «Probar»)</span></label>
      <input class="input" id="set-or" type="password" autocomplete="off" placeholder="${s.openrouter_key ? "Guardada en el llavero del sistema" : "sk-or-…"}">
      <span class="hint">Créala en <a href="https://openrouter.ai/keys" target="_blank" rel="noopener">openrouter.ai/keys</a>. Se guarda en el llavero del sistema, nunca en archivos.
      ${s.openrouter_key ? ' · <a href="#" data-delete-key="openrouter_api_key">Borrar la clave</a>' : ""}</span></div>
    <div class="field"><label for="set-hf">Token de Hugging Face <span class="muted">(opcional: más consultas y modelos restringidos)</span></label>
      <input class="input" id="set-hf" type="password" autocomplete="off" placeholder="${s.hf_token ? "Guardado" : "hf_…"}">
      ${s.hf_token ? '<span class="hint"><a href="#" data-delete-key="huggingface_token">Borrar el token</a></span>' : ""}</div>
    <div class="field"><label for="set-hours">Actualizar el catálogo cada (horas)</label>
      <input class="input" id="set-hours" type="number" min="1" max="720" value="${esc(s.refresh_hours)}">
      <span class="hint">También se actualiza al abrir la app si los datos tienen más de ese tiempo.</span></div>
    <div class="actions"><button class="btn primary" type="button" id="set-save" ${mock ? "disabled" : ""}>Guardar</button>
      <button class="btn" type="button" id="set-refresh" ${mock ? "disabled" : ""}>Actualizar catálogo ahora</button>
      <span class="muted" id="set-status"></span></div>
    <p class="muted small-text">Versión de la app: ${v.commit ? `${esc(v.commit)} (${fmtDate(v.date)})` : "sin datos"}${v.source === "desarrollo" ? " · desarrollo" : ""}.
      Se actualiza sola al abrirla si hay una versión nueva.</p>`);
}

async function saveSettings() {
  try {
    const res = await api.post("/api/settings", { openrouter_key: $("#set-or").value.trim() || null,
      hf_token: $("#set-hf").value.trim() || null, refresh_hours: Number($("#set-hours").value) });
    $("#set-status").textContent = res.message || "Guardado.";
    $("#set-or").value = ""; $("#set-hf").value = "";
  } catch (err) { $("#set-status").textContent = err.message; }
}

async function deleteKey(name) {
  try { await api.post("/api/settings", { delete: [name] }); await openSettings(); toast("Borrado."); }
  catch (err) { toast(err.message); }
}

async function refreshCatalog() {
  try {
    const res = await api.post("/api/refresh", {});
    closeModal();
    toast(res.message);
    await pollStatus();
  } catch (err) { toast(err.message); }
}

// Keeps the freshness indicator current and reloads the list when the server finished an update.
async function pollStatus() {
  if (state.mode !== "live") return;
  try {
    const s = await api.get("/api/status");
    const changed = state.status && s.built_at && s.built_at !== state.status.built_at;
    state.status = s;
    if (changed) {
      const data = await api.get("/api/catalog");
      state.records = data.records; state.meta = data; state.checks = {};
      await loadFit();
      update({ keepPage: true });
      toast("Datos actualizados.");
    } else {
      renderFreshness();
    }
  } catch (e) { /* server restarting: try again later */ }
}

// --- misc ---

let toastTimer = null;
function toast(text) {
  const t = $("#toast");
  t.textContent = text; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 2200);
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast(`Copiado: ${text}`); }
  catch (e) { toast("No se ha podido copiar"); }
}

function showFatal(message) {
  $("#count").textContent = "No se ha podido cargar el catálogo";
  $("#method").textContent = message;
}

function applySuggestion() {
  const s = state.result.suggestion; const q = state.query;
  if (!s) return;
  if (s.kind === "cap") q.caps = q.caps.filter((c) => c !== s.key);
  else if (s.kind === "req") q.reqs = q.reqs.filter((r) => r !== s.key);
  else if (s.kind === "price") q.maxPrice = null;
  else if (s.kind === "text") { q.text = ""; $("#search").value = ""; }
  else if (s.kind === "fits") q.onlyFits = false;
  else if (s.kind === "where") q.where = s.key;
  update();
}

function bind() {
  const q = () => state.query;
  let timer = null;
  $("#search").addEventListener("input", (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => { q().text = e.target.value.trim(); update(); liveSearch(q().text); }, 120);
  });
  $("#sort").addEventListener("change", (e) => { q().sort = e.target.value; update(); });
  $("#sidebar").addEventListener("change", async (e) => {
    const id = e.target.dataset.id;
    if (e.target.id === "profile") { await useProfile(e.target.value); return; }
    if (e.target.id === "harness") { q().harness = e.target.value || null; q().harnessStrict = true; update(); return; }
    if (e.target.id === "price") return;
    if (!id) return;
    const on = e.target.checked;
    const toggle = (list, k) => (on ? [...new Set([...list, k])] : list.filter((x) => x !== k));
    if (id.startsWith("cap:")) q().caps = toggle(q().caps, id.slice(4));
    else if (id.startsWith("req:")) q().reqs = toggle(q().reqs, id.slice(4));
    else if (id === "onlyFits") q().onlyFits = on;
    else if (id === "harnessAll") q().harnessStrict = !on;
    else if (id === "includeOld") q().includeOld = on;
    else if (id === "community") q().community = on;
    else if (id === "limitPrice") q().maxPrice = on ? PRICE_STEPS[state.priceStep] : null;
    update();
  });
  $("#sidebar").addEventListener("input", (e) => {
    if (e.target.id !== "price") return;
    state.priceStep = Number(e.target.value);
    q().maxPrice = PRICE_STEPS[state.priceStep];
    update();
    const el = $("#price"); if (el) el.focus();
  });
  $("#sidebar").addEventListener("click", (e) => {
    const b = e.target.closest("[data-where]");
    if (b) { q().where = b.dataset.where; update(); }
    if (e.target.closest("[data-open='pc']")) openPc();
  });
  $("#chips").addEventListener("click", (e) => {
    const key = e.target.closest("[data-chip]")?.dataset.chip;
    if (!key) return;
    if (key === "all") { Object.assign(state.query, { ...emptyQuery(), where: q().where, sort: q().sort }); $("#search").value = ""; }
    else if (key === "text") { q().text = ""; $("#search").value = ""; }
    else if (key.startsWith("cap:")) q().caps = q().caps.filter((c) => c !== key.slice(4));
    else if (key.startsWith("req:")) q().reqs = q().reqs.filter((r) => r !== key.slice(4));
    else if (key === "price") q().maxPrice = null;
    else if (key === "includeOld") q().includeOld = false;
    else if (key === "community") q().community = false;
    else if (key === "harness") { q().harness = null; q().harnessStrict = true; }
    update();
  });
  document.addEventListener("click", (e) => {
    const t = e.target;
    const model = t.closest("[data-open-model]");
    if (model && !t.closest("#drawer")) { openModel(model.dataset.openModel); return; }
    if (t.closest("[data-more]")) { state.shown += PAGE; renderList(); return; }
    if (t.closest("[data-suggest]")) { applySuggestion(); return; }
    if (t.closest("[data-close]") || t === $("#scrim")) { closeDrawer(); return; }
    if (t.closest("[data-close-modal]") || t === $("#modal")) { closeModal(); return; }
    const c = t.closest("[data-copy]"); if (c) { copy(c.dataset.copy); return; }
    const tr = t.closest("[data-try]"); if (tr) { openTry(tr.dataset.try); return; }
    const up = t.closest("[data-use-profile]"); if (up) { useProfile(up.dataset.useProfile).then(() => openPc()); return; }
    const ed = t.closest("[data-edit-profile]");
    if (ed) { openPc(state.profiles.find((p) => p.name === ed.dataset.editProfile)); return; }
    const del = t.closest("[data-delete-profile]");
    if (del) { if (confirm(`¿Borrar el equipo «${del.dataset.deleteProfile}»?`)) deletePc(del.dataset.deleteProfile); return; }
    const dk = t.closest("[data-delete-key]"); if (dk) { e.preventDefault(); deleteKey(dk.dataset.deleteKey); return; }
    if (t.closest("#modal [data-open='pc']")) { openPc(); return; }
    if (t.closest("#freshness")) { refreshCatalog(); return; }
    if (t.closest("#try-send")) { sendTry(); return; }
    if (t.closest("#pc-save")) { savePc(t.closest("#pc-save").dataset.original); return; }
    if (t.closest("#pc-detect")) { detectPc(); return; }
    if (t.closest("#set-save")) { saveSettings(); return; }
    if (t.closest("#set-refresh")) { refreshCatalog(); }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { if (!$("#modal").hidden) closeModal(); else closeDrawer(); }
    if (e.key === "Enter" && e.target.matches("[data-open-model]")) openModel(e.target.dataset.openModel);
    if (e.key === "/" && document.activeElement !== $("#search") && !e.target.matches("input, textarea")) {
      e.preventDefault(); $("#search").focus();
    }
  });
  $("#open-pc").addEventListener("click", () => openPc());
  document.addEventListener("change", (e) => {
    if (e.target.id === "eq-template") applyTemplate(e.target.value);
    if (e.target.id === "drawer-harness") {
      const rec = state.records.find((r) => r.id === state.openId);
      if (rec) renderSteps(e.target.value, rec);
    }
  });
  $("#open-settings").addEventListener("click", openSettings);
  $("#theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
  });
}

async function main() {
  restoreQuery();
  if (state.query.maxPrice !== null) {
    const i = PRICE_STEPS.indexOf(state.query.maxPrice);
    state.priceStep = i >= 0 ? i : state.priceStep;
  }
  $("#search").value = state.query.text;
  bind();
  $("#top").innerHTML = '<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>';
  if (!(await loadCatalog())) return;
  $("#pc-label").textContent = state.profile ? state.profile.name : "Mis equipos";
  update();
  if (state.query.text) liveSearch(state.query.text);
  setInterval(pollStatus, 30000);
  const open = new URLSearchParams(location.search).get("open");     // used for screenshots and links
  if (open === "pc") openPc();
  else if (open === "settings") openSettings();
  else if (open) openModel(open);
}

main();
