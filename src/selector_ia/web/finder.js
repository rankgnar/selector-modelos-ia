// Model search logic: filter, count, sort and explain. Pure functions over the catalog records built by
// selector_ia/finder/catalog.py, so the page reacts instantly and the same choices always give the same list.
// Nothing here invents data: every sentence is built from a field of the record; missing = "sin datos".

export const CAPABILITIES = [
  { key: "chat", label: "Escribir y chatear", help: "Conversar, redactar, resumir o responder preguntas." },
  { key: "code", label: "Programar", help: "Modelos con alguna prueba de que sirven para programar: índice de programación de Artificial Analysis, benchmarks de programación en su ficha, su nombre (coder, devstral…) o la descripción de su proveedor. Se ordenan por el índice de programación." },
  { key: "tools", label: "Usar herramientas", help: "Llamar a funciones y APIs: agentes y automatizaciones (dato de OpenRouter)." },
  { key: "reasoning", label: "Razonar paso a paso", help: "Piensa antes de responder (dato de OpenRouter)." },
  { key: "pdf", label: "Leer PDF y archivos", help: "Acepta archivos directamente (dato de OpenRouter)." },
  { key: "vision", label: "Ver imágenes", help: "Entiende fotos, capturas y gráficos." },
  { key: "audio_in", label: "Escuchar audio y transcribir", help: "Acepta audio: transcripción o preguntas sobre grabaciones." },
  { key: "video_in", label: "Ver vídeo", help: "Acepta vídeo como entrada (dato de OpenRouter)." },
  { key: "image_gen", label: "Generar imágenes", help: "Crea imágenes a partir de un texto." },
  { key: "image_edit", label: "Editar imágenes", help: "Cambia una imagen siguiendo instrucciones." },
  { key: "voice", label: "Generar voz", help: "Convierte texto en voz." },
  { key: "music", label: "Generar música y sonidos", help: "Crea música o efectos a partir de un texto." },
  { key: "video_gen", label: "Generar vídeo", help: "Crea vídeo a partir de un texto o una imagen." },
  { key: "embeddings", label: "Buscar en documentos", help: "Modelos de embeddings para buscar por significado (RAG)." },
  { key: "rerank", label: "Reordenar resultados (RAG)", help: "Modelos «rerank»: ordenan por relevancia los documentos encontrados antes de dárselos al chat." },
];

export const REQUIREMENTS = [
  { key: "spanish", label: "Declara español", help: "Solo los modelos que publican que hablan español. Ojo: la mayoría de modelos de API (Claude, GPT, Gemini…) no publican sus idiomas aunque lo hablen, así que no salen con esta casilla." },
  { key: "uncensored", label: "Sin censura", help: "Modelos que su autor o proveedor declara sin censura («uncensored», «abliterated»…). Casi siempre son versiones de la comunidad: calidad variable y licencia del modelo original." },
  { key: "free", label: "Gratis", help: "Versión gratis en OpenRouter (con límites de uso) o descargable para usarlo sin pagar." },
  { key: "commercial", label: "Uso comercial", help: "Su licencia permite usarlo en un negocio (a veces con condiciones)." },
  { key: "open", label: "Se puede descargar", help: "Pesos abiertos: se puede usar en tu equipo o en un servidor propio." },
  { key: "long", label: "Textos muy largos", help: "Admite al menos 200.000 tokens de una vez (dato de OpenRouter)." },
];

export const WHERE = [
  { key: "openrouter", label: "OpenRouter", help: "Por internet con tu clave de OpenRouter, pagando por uso." },
  { key: "local", label: "Mis equipos", help: "Descargado y ejecutado en tu ordenador, un VPS o un servidor con GPU: sin pagar por uso y con tus datos en casa." },
  { key: "provider", label: "Otro proveedor", help: "Servicios fuera de OpenRouter que lo ofrecen ahora (fal.ai, Replicate, Together…), según Hugging Face." },
  { key: "any", label: "Cualquiera", help: "Todos los modelos, estén donde estén." },
];

export const SORTS = [
  { key: "quality", label: "Mejor calidad" },
  { key: "value", label: "Mejor calidad-precio" },
  { key: "cheap", label: "Más barato" },
  { key: "popular", label: "Más usado" },
  { key: "new", label: "Más nuevo" },
  { key: "light", label: "Más ligero" },
];

export const LONG_CONTEXT = 200000;
const LLM_CAPS = new Set(["chat", "code", "tools", "reasoning", "pdf", "vision", "video_in"]);
const BENCH = {
  intelligence: { label: "índice de inteligencia", source: "Artificial Analysis" },
  coding: { label: "índice de programación", source: "Artificial Analysis" },
  agentic: { label: "índice de agentes", source: "Artificial Analysis" },
  asr_wer: { label: "error de transcripción (WER)", source: "Open ASR Leaderboard", lowerIsBetter: true, unit: " %" },
  arena_image: { label: "ELO de imágenes", source: "Design Arena" },
  arena_edit: { label: "ELO de edición", source: "Design Arena" },
};

export function emptyQuery() {
  return { text: "", caps: [], reqs: [], where: "openrouter", onlyFits: true, maxPrice: null,
           includeOld: false, sort: "quality", harness: null, harnessStrict: true, community: false,
           // how a downloaded model will be used (Mis equipos): drives memory, quantization and ranking
           use: "chat", ctx: 8192, agents: 1, kvq: false };
}

// --- harnesses (coding agents that use the OpenRouter API; data in harnesses.json, from official docs) ---

export function harnessOk(rec, harness, strict = true) {
  if (!harness) return true;
  if (!rec.openrouter || !(rec.capabilities || []).includes("chat")) return false;   // text agents
  const authors = (harness.requires && harness.requires.authors) || [];
  return !strict || !authors.length || authors.includes(rec.openrouter.id.split("/")[0]);
}

// The agent's setup steps with this model filled in.
export function harnessSteps(harness, rec) {
  const id = rec.openrouter ? rec.openrouter.id : rec.id;
  const fill = (s) => s.replaceAll("{model}", id).replaceAll("{name}", rec.name)
    .replaceAll("{tools_capability}", rec.openrouter && rec.openrouter.tools ? "\n    capabilities:\n      - tool_use" : "");
  return harness.steps.map((st) => ({ title: fill(st.title), text: st.text ? fill(st.text) : null,
                                      code: st.code ? fill(st.code) : null }));
}

// --- formatting ---

export function fmtNumber(value, decimals = 2) {
  if (value === null || value === undefined) return "sin datos";
  return value.toLocaleString("es-ES", { maximumFractionDigits: decimals });
}

export function fmtUsd(value) {
  if (value === null || value === undefined) return "sin datos";
  if (value === 0) return "gratis";
  return `${fmtNumber(value, value < 1 ? 3 : 2)} USD`;
}

export function fmtTokens(n) {
  if (!n) return "sin datos";
  if (n >= 1e6) return `${fmtNumber(n / 1e6, 2)} M`;
  return `${fmtNumber(Math.round(n / 1000), 0)} K`;
}

export function fmtParams(n) {
  if (!n) return "sin datos";
  return n >= 1e9 ? `${fmtNumber(n / 1e9, 1)} mil M` : `${fmtNumber(n / 1e6, 0)} M`;
}

export function fmtDate(iso) {
  if (!iso) return "sin datos";
  const d = new Date(iso);
  return d.toLocaleDateString("es-ES", { day: "2-digit", month: "2-digit", year: "numeric" });
}

// --- facts ---

export function capLabel(key) {
  return (CAPABILITIES.find((c) => c.key === key) || { label: key }).label;
}

export function reqLabel(key) {
  return (REQUIREMENTS.find((r) => r.key === key) || { label: key }).label;
}

export function whereLabel(key) {
  return (WHERE.find((w) => w.key === key) || { label: key }).label;
}

// Main API price for what the student asked for, USD per million tokens, with what it measures.
export function price(rec, caps = []) {
  const p = rec.openrouter && rec.openrouter.prices;
  if (rec.openrouter && rec.openrouter.video_usd_per_second != null && (caps.includes("video_gen") || !p || !Object.keys(p).length)) {
    return { value: rec.openrouter.video_usd_per_second, unit: `por segundo de vídeo (${rec.openrouter.video_resolution}, la resolución más barata)` };
  }
  if (!p || !Object.keys(p).length) return { value: null, unit: "" };
  if (caps.includes("audio_in")) {
    if (p.audio !== undefined) return { value: p.audio, unit: "por millón de tokens de audio" };
    if (p.prompt !== undefined) return { value: p.prompt, unit: "por millón de tokens de entrada (sin precio de audio aparte)" };
  }
  if ((caps.includes("image_gen") || caps.includes("image_edit")) && p.image_output !== undefined) {
    return { value: p.image_output, unit: "por millón de tokens de imagen generada" };
  }
  if (p.prompt === undefined || p.completion === undefined) return { value: null, unit: "" };
  return { value: (3 * p.prompt + p.completion) / 4, unit: "por millón de tokens (3 de entrada por 1 de salida)" };
}

export function isFree(rec) {
  return Boolean(rec.openrouter && rec.openrouter.free_id);
}

function requirementOk(rec, key, where) {
  switch (key) {
    case "spanish": return (rec.languages || []).includes("es");
    case "free": return where === "openrouter" ? isFree(rec) : isFree(rec) || Boolean(rec.hf_id);
    case "commercial": return !rec.hf_id || ["permissive", "conditions"].includes(rec.license_class);
    case "open": return Boolean(rec.hf_id);
    case "long": return Boolean(rec.openrouter && rec.openrouter.context >= LONG_CONTEXT);
    case "uncensored": return Boolean(rec.uncensored) && !(rec.openrouter && rec.openrouter.moderated === true);
    default: return true;
  }
}

function availableAt(rec, where, query, fit) {
  if (where === "openrouter") return Boolean(rec.openrouter);
  if (where === "provider") return (rec.providers || []).length > 0;
  if (where === "local") {
    if (!rec.hf_id) return false;
    if (!query.onlyFits) return true;
    const f = fit && fit[rec.id];
    return Boolean(f && (f.verdict === "ok" || f.verdict === "limited"));
  }
  return true;
}

function textOk(rec, text) {
  const words = (text || "").toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const hay = `${rec.id} ${rec.name} ${rec.author} ${rec.openrouter ? rec.openrouter.id : ""}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}

// Requirements that only make sense in some places (hidden elsewhere).
export function requirementsFor(where) {
  return REQUIREMENTS.filter((r) => !(r.key === "free" && (where === "local" || where === "provider")))
    .filter((r) => !(r.key === "open" && where === "local"));
}

export function sortsFor(where) {
  return SORTS.filter((s) => !(where === "local" && (s.key === "cheap" || s.key === "value")));
}

// Community versions (fine-tunes, merges, uncensored builds by third parties) only when asked for.
export function communityShown(query) {
  return query.community || query.reqs.includes("uncensored");
}

export function matches(rec, query, fit, harnesses = null) {
  // Models found by name in all of Hugging Face (rec.live) are shown even if old or community: the student
  // searched for them; their badges say what they are.
  if (!query.includeOld && rec.old && !rec.live) return false;
  if (rec.community && !communityShown(query) && !rec.live) return false;
  if (query.harness && query.where === "openrouter" && harnesses) {
    if (!harnessOk(rec, harnesses[query.harness], query.harnessStrict)) return false;
  }
  if (!textOk(rec, query.text)) return false;
  if (!availableAt(rec, query.where, query, fit)) return false;
  const caps = rec.capabilities || [];
  if (!query.caps.every((c) => caps.includes(c))) return false;
  const shown = new Set(requirementsFor(query.where).map((r) => r.key));
  if (!query.reqs.every((r) => !shown.has(r) || requirementOk(rec, r, query.where))) return false;
  if (query.maxPrice !== null && query.where === "openrouter") {
    const p = price(rec, query.caps).value;
    if (p === null || p > query.maxPrice) return false;
  }
  return true;
}

// --- quality ---

// Which benchmarks measure "quality" for what was ticked.
export function relevantBenchmarks(caps, query = null) {
  const keys = [];
  // Agentic use on own hardware: tool-calling reliability first (the method's criterion number one).
  if (query && query.where === "local" && query.use === "agent") keys.push("agentic");
  if (caps.includes("code")) keys.push("coding");
  if (caps.includes("tools") && !keys.includes("agentic")) keys.push("agentic");
  if (caps.includes("audio_in")) keys.push("asr_wer");
  if (caps.includes("image_gen")) keys.push("arena_image");
  if (caps.includes("image_edit")) keys.push("arena_edit");
  const llm = caps.some((c) => LLM_CAPS.has(c));
  if (!keys.length && (llm || !caps.length)) keys.push("intelligence");
  return keys;
}

function bestValues(records, keys) {
  const best = {};
  for (const k of keys) {
    const vals = records.map((r) => r.benchmarks && r.benchmarks[k]).filter((v) => typeof v === "number" && v > 0);
    if (!vals.length) continue;
    best[k] = BENCH[k].lowerIsBetter ? Math.min(...vals) : Math.max(...vals);
  }
  return best;
}

function logNorm(values, value) {
  const pos = values.filter((v) => v > 0);
  if (!value || value <= 0 || !pos.length) return 0;
  const lo = Math.log10(Math.min(...pos));
  const hi = Math.log10(Math.max(...pos));
  return hi <= lo ? 1 : (Math.log10(value) - lo) / (hi - lo);
}

function scoreAll(records, query) {
  const keys = relevantBenchmarks(query.caps, query);
  const best = bestValues(records, keys);
  const downloads = records.map((r) => r.downloads || 0);
  const trending = records.map((r) => r.trending || 0);
  return records.map((rec) => {
    const parts = [];
    for (const k of keys) {
      const v = rec.benchmarks && rec.benchmarks[k];
      if (typeof v !== "number" || !best[k]) continue;
      const norm = BENCH[k].lowerIsBetter ? best[k] / v : v / best[k];
      parts.push({ key: k, value: v, norm });
    }
    const quality = parts.length ? parts.reduce((s, p) => s + p.norm, 0) / parts.length : null;
    const popularity = 0.5 * logNorm(downloads, rec.downloads) + 0.5 * logNorm(trending, rec.trending);
    return { rec, parts, quality, popularity, price: price(rec, query.caps) };
  });
}

const byId = (a, b) => (a.rec.id < b.rec.id ? -1 : a.rec.id > b.rec.id ? 1 : 0);
const desc = (x, y) => (y ?? -Infinity) - (x ?? -Infinity);
const asc = (x, y) => (x ?? Infinity) - (y ?? Infinity);

function sorter(sort) {
  const newest = (a, b) => desc(Date.parse(a.rec.created || 0), Date.parse(b.rec.created || 0));
  const byQuality = (a, b) => desc(a.quality, b.quality) || desc(a.popularity, b.popularity) || newest(a, b) || byId(a, b);
  switch (sort) {
    case "cheap": return (a, b) => asc(a.price.value, b.price.value) || byQuality(a, b);
    case "value": return (a, b) => desc(valueOf(a), valueOf(b)) || byQuality(a, b);
    case "popular": return (a, b) => desc(a.rec.downloads, b.rec.downloads) || desc(a.popularity, b.popularity) || byId(a, b);
    case "new": return (a, b) => desc(Date.parse(a.rec.created || 0), Date.parse(b.rec.created || 0)) || byId(a, b);
    case "light": return (a, b) => asc(a.rec.params, b.rec.params) || byQuality(a, b);
    default: return byQuality;
  }
}

function valueOf(s) {
  if (s.quality === null || s.price.value === null) return null;
  return s.price.value === 0 ? s.quality * 1e6 : s.quality / s.price.value;
}

// --- search ---

export function search(records, query, fit = null, harnesses = null) {
  const matching = records.filter((r) => matches(r, query, fit, harnesses));
  const scored = scoreAll(matching, query).sort(sorter(query.sort));
  scored.forEach((s, i) => { s.position = i + 1; });
  return {
    results: scored,
    count: scored.length,
    counts: counts(records, query, fit, harnesses),
    suggestion: scored.length ? null : suggestion(records, query, fit, harnesses),
    method: methodText(query),
  };
}

function countWith(records, query, fit, harnesses) {
  let n = 0;
  for (const r of records) if (matches(r, query, fit, harnesses)) n += 1;
  return n;
}

// How many models would be listed if each option were toggled (or chosen, for "where").
export function counts(records, query, fit, harnesses = null) {
  const n = (q) => countWith(records, q, fit, harnesses);
  const toggle = (list, key) => (list.includes(key) ? list.filter((k) => k !== key) : [...list, key]);
  const caps = {};
  for (const c of CAPABILITIES) caps[c.key] = n({ ...query, caps: toggle(query.caps, c.key) });
  const reqs = {};
  for (const r of REQUIREMENTS) reqs[r.key] = n({ ...query, reqs: toggle(query.reqs, r.key) });
  const where = {};
  for (const w of WHERE) where[w.key] = n({ ...query, where: w.key });
  const harness = {};
  for (const id of Object.keys(harnesses || {})) harness[id] = n({ ...query, where: "openrouter", harness: id });
  harness[""] = n({ ...query, harness: null });
  const old = n({ ...query, includeOld: !query.includeOld });
  const community = n({ ...query, community: !query.community });
  const strict = n({ ...query, harnessStrict: !query.harnessStrict });
  return { caps, reqs, where, harness, old, strict, community };
}

// When nothing matches: the single change that brings back the most models.
export function suggestion(records, query, fit, harnesses = null) {
  const countWithH = (rs, q, f) => countWith(rs, q, f, harnesses);
  const options = [];
  for (const c of query.caps) {
    options.push({ kind: "cap", key: c, label: `Quitar «${capLabel(c)}»`,
                   n: countWithH(records, { ...query, caps: query.caps.filter((x) => x !== c) }, fit) });
  }
  for (const r of query.reqs) {
    options.push({ kind: "req", key: r, label: `Quitar «${reqLabel(r)}»`,
                   n: countWithH(records, { ...query, reqs: query.reqs.filter((x) => x !== r) }, fit) });
  }
  if (query.maxPrice !== null && query.where === "openrouter") {
    options.push({ kind: "price", key: "maxPrice", label: "Quitar el precio máximo",
                   n: countWithH(records, { ...query, maxPrice: null }, fit) });
  }
  if (query.text) {
    options.push({ kind: "text", key: "text", label: `Borrar la búsqueda «${query.text}»`,
                   n: countWithH(records, { ...query, text: "" }, fit) });
  }
  if (query.harness && query.where === "openrouter") {
    options.push({ kind: "harness", key: "harness", label: "Quitar el programa elegido",
                   n: countWithH(records, { ...query, harness: null }, fit) });
    if (query.harnessStrict) {
      options.push({ kind: "strict", key: "harnessStrict", label: "Ver también los modelos sin garantía",
                     n: countWithH(records, { ...query, harnessStrict: false }, fit) });
    }
  }
  if (query.where === "local" && query.onlyFits) {
    options.push({ kind: "fits", key: "onlyFits", label: "Ver también los que no caben",
                   n: countWithH(records, { ...query, onlyFits: false }, fit) });
  }
  for (const w of WHERE) {
    if (w.key === query.where) continue;
    options.push({ kind: "where", key: w.key, label: `Ver «${w.label}»`,
                   n: countWithH(records, { ...query, where: w.key }, fit) });
  }
  // What the model must DO is what the student needs: change place, requirements or price first,
  // and only suggest dropping a capability when nothing else helps.
  const pick = (list) => list.filter((o) => o.n > 0).sort((a, b) => b.n - a.n)[0];
  return pick(options.filter((o) => o.kind !== "cap")) || pick(options.filter((o) => o.kind === "cap")) || null;
}

export function methodText(query) {
  const keys = relevantBenchmarks(query.caps, query);
  const names = keys.map((k) => `${BENCH[k].label} (${BENCH[k].source})`).join(" y ");
  switch (query.sort) {
    case "cheap": return "Del más barato al más caro, con el precio publicado en OpenRouter.";
    case "value": return `Calidad dividida por precio. Calidad = ${names || "sin benchmark para esta selección"}.`;
    case "popular": return "Por descargas del último mes en Hugging Face (los modelos solo de API no publican descargas).";
    case "new": return "Del más reciente al más antiguo, por fecha de publicación.";
    case "light": return "Del más pequeño (menos parámetros) al más grande: necesita menos memoria y va más rápido.";
    default:
      return names ? `Por ${names}${keys.length > 1 ? ", combinados" : ""}. Los modelos sin ese dato van detrás, por uso y novedad.`
                   : "No hay benchmark público para esta selección: se ordena por uso actual en Hugging Face y, sin ese dato, del más nuevo al más antiguo.";
  }
}

// --- explanations ---

export function benchText(key, value) {
  const b = BENCH[key];
  return `${b.source}, ${b.label}: ${fmtNumber(value)}${b.unit || ""}`;
}

// One line: why this model is at this position for the chosen order.
export function reason(s, query, total) {
  const of = `${s.position}.º de ${total}`;
  switch (query.sort) {
    case "cheap":
      return s.price.value === null ? `Sin precio publicado (${of})` : `${fmtUsd(s.price.value)} ${s.price.unit} · ${of}`;
    case "value":
      return valueOf(s) === null ? `Sin benchmark o sin precio (${of})`
        : `${s.parts.map((p) => benchText(p.key, p.value)).join(" · ")} por ${fmtUsd(s.price.value)} · ${of}`;
    case "popular":
      return s.rec.downloads ? `${fmtNumber(s.rec.downloads, 0)} descargas el último mes · ${of}` : `Sin datos de descargas (${of})`;
    case "new":
      return `Publicado el ${fmtDate(s.rec.created)} · ${of}`;
    case "light":
      return s.rec.params ? `${fmtParams(s.rec.params)} de parámetros · ${of}` : `Tamaño sin datos (${of})`;
    default:
      return s.parts.length ? `${s.parts.map((p) => benchText(p.key, p.value)).join(" · ")} · ${of}`
        : s.rec.downloads ? `Sin benchmark para esto; ${fmtNumber(s.rec.downloads, 0)} descargas al mes · ${of}`
        : `Sin benchmark ni datos de uso; publicado el ${fmtDate(s.rec.created)} · ${of}`;
  }
}

// Pros and cons, built only from the record. `context`: the other results (for "the cheapest", etc.).
export function prosCons(s, query, results) {
  const rec = s.rec;
  const pros = [];
  const cons = [];
  const days = rec.created ? (Date.now() - Date.parse(rec.created)) / 86400000 : null;
  const priced = results.map((r) => r.price.value).filter((v) => v !== null && v > 0).sort((a, b) => a - b);

  for (const p of s.parts) pros.push(benchText(p.key, p.value));
  if (!s.parts.length && relevantBenchmarks(query.caps, query).length) cons.push("Sin benchmark publicado para lo que has marcado");
  if (rec.mtp) pros.push("Trae MTP (predicción de varios tokens): puede ir más rápido con decodificación especulativa");
  if (isFree(rec) && query.where !== "local") pros.push("Tiene versión gratis en OpenRouter (con límites de uso)");
  else if (s.price.value !== null && priced.length > 1 && s.price.value === priced[0]) pros.push("El más barato de la lista");
  if (rec.hf_id) pros.push("Se puede descargar y usar en tu equipo");
  if (rec.license_class === "permissive") pros.push(`Licencia ${rec.license}: permite uso comercial`);
  if (rec.openrouter && rec.openrouter.context >= LONG_CONTEXT) pros.push(`Textos muy largos: ${fmtTokens(rec.openrouter.context)} tokens`);
  if ((rec.providers || []).length) pros.push(`También en: ${rec.providers.join(", ")}`);
  if (rec.downloads && rec.downloads >= 100000) pros.push(`Muy usado: ${fmtNumber(rec.downloads, 0)} descargas al mes`);
  if (days !== null && days <= 120) pros.push(`Reciente: ${fmtDate(rec.created)}`);
  if (query.reqs.includes("spanish") && (rec.languages || []).includes("es")) pros.push("Declara que habla español");
  if (rec.uncensored && query.reqs.includes("uncensored")) pros.push("Declarado sin censura por su autor o proveedor");

  if (rec.community) {
    cons.unshift(`Versión de la comunidad${rec.base_model ? ` basada en ${rec.base_model}` : ""}: no es oficial y la calidad varía`);
  }
  if (rec.adult) cons.unshift("Su autor lo marca como contenido adulto (no apto para todos los públicos)");
  if (!rec.hf_id) cons.push("No se puede descargar: solo por internet (API)");
  if (s.price.value !== null && priced.length >= 4 && s.price.value >= priced[Math.floor(priced.length * 0.75)]) {
    cons.push(`Entre el 25 % más caro: ${fmtUsd(s.price.value)} ${s.price.unit}`);
  }
  if (rec.hf_id) {
    if (rec.license_class === "non_commercial") cons.push(`Su licencia no permite uso comercial (${rec.license})`);
    else if (rec.license_class === "conditions") cons.push(`Uso comercial con condiciones: revisa la licencia «${rec.license}»`);
    else if (rec.license_class === "unknown") cons.push("Licencia sin datos: revísala antes de usarlo");
  }
  if (rec.gated) cons.push("Hay que aceptar sus condiciones en Hugging Face antes de descargarlo");
  if (rec.params_note) cons.push(rec.params_note);
  if (rec.params && rec.params >= 70e9) cons.push(`Muy grande (${fmtParams(rec.params)} de parámetros): en local necesita un equipo muy potente`);
  if (rec.old) cons.push(`Antiguo: publicado el ${fmtDate(rec.created)}`);
  if (rec.openrouter && rec.openrouter.expires) cons.push(`OpenRouter lo retira el ${fmtDate(rec.openrouter.expires)}`);
  if (rec.openrouter && rec.openrouter.preview) cons.push("Versión preliminar (preview): puede cambiar");
  return { pros, cons };
}
