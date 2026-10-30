// Run with: node --test tests/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { emptyQuery, search, matches, price, relevantBenchmarks, reason, prosCons, requirementsFor }
  from "../../src/selector_ia/web/finder.js";

const NOW = new Date().toISOString();
const rec = (id, extra = {}) => ({
  id, name: id.split("/")[1], author: id.split("/")[0], capabilities: ["chat", "code"], providers: [],
  languages: [], benchmarks: {}, created: NOW, old: false, license_class: "unknown", ...extra,
});
const or = (id, prompt, completion, extra = {}) => ({ id, url: `https://openrouter.ai/${id}`, context: 128000,
  prices: { prompt, completion }, ...extra });

const RECORDS = [
  rec("v/smart", { capabilities: ["chat", "code", "pdf", "tools"], openrouter: or("v/smart", 10, 40),
                   benchmarks: { intelligence: 60, coding: 50, agentic: 40 } }),
  rec("v/coder", { capabilities: ["chat", "code", "tools"], openrouter: or("v/coder", 1, 4),
                   benchmarks: { intelligence: 45, coding: 58, agentic: 30 } }),
  rec("v/agent", { capabilities: ["chat", "code", "pdf", "tools"], openrouter: or("v/agent", 2, 8),
                   benchmarks: { intelligence: 50, coding: 40, agentic: 55 } }),
  rec("o/open", { capabilities: ["chat", "code"], hf_id: "o/open", license: "apache-2.0", license_class: "permissive",
                  openrouter: or("o/open", 0.1, 0.4, { free_id: "o/open:free" }), params: 8e9, downloads: 5e6,
                  languages: ["en"], benchmarks: { intelligence: 30, coding: 25 } }),
  rec("o/tiny", { hf_id: "o/tiny", params: 1e9, downloads: 2e6, languages: ["es", "en"], license_class: "permissive",
                  license: "mit" }),
  rec("a/asr-good", { capabilities: ["audio_in"], hf_id: "a/asr-good", benchmarks: { asr_wer: 5.1 }, downloads: 10 }),
  rec("a/asr-popular", { capabilities: ["audio_in"], hf_id: "a/asr-popular", benchmarks: { asr_wer: 7.4 }, downloads: 1e7 }),
  rec("m/video", { capabilities: ["video_gen"], hf_id: "m/video", providers: ["fal-ai"] }),
  rec("x/legacy", { openrouter: or("x/legacy", 1, 1), old: true, benchmarks: { intelligence: 99 } }),
];

const ids = (r) => r.results.map((s) => s.rec.id);
const q = (over = {}) => ({ ...emptyQuery(), ...over });

test("every ticked capability must be met (AND)", () => {
  const r = search(RECORDS, q({ caps: ["pdf", "tools"] }));
  assert.deepEqual(ids(r).sort(), ["v/agent", "v/smart"]);
  for (const s of r.results) assert.ok(["pdf", "tools"].every((c) => s.rec.capabilities.includes(c)));
});

test("the order follows what was ticked", () => {
  assert.equal(ids(search(RECORDS, q()))[0], "v/smart");                        // intelligence
  assert.equal(ids(search(RECORDS, q({ caps: ["code"] })))[0], "v/coder");      // coding index
  assert.equal(ids(search(RECORDS, q({ caps: ["tools"] })))[0], "v/agent");     // agentic index
  assert.deepEqual(relevantBenchmarks(["code", "tools"]), ["coding", "agentic"]);
});

test("sort options change the order", () => {
  const caps = ["code"];
  assert.equal(ids(search(RECORDS, q({ caps, sort: "cheap" })))[0], "o/open");
  assert.equal(ids(search(RECORDS, q({ caps, sort: "value" })))[0], "o/open");   // free version counts as best value
  assert.equal(ids(search(RECORDS, q({ caps, sort: "quality" })))[0], "v/coder");
});

test("same query, same answer", () => {
  assert.deepEqual(ids(search(RECORDS, q({ caps: ["code"] }))), ids(search(RECORDS, q({ caps: ["code"] }))));
});

test("old models only when asked", () => {
  assert.ok(!ids(search(RECORDS, q())).includes("x/legacy"));
  assert.ok(ids(search(RECORDS, q({ includeOld: true }))).includes("x/legacy"));
});

test("where: openrouter, provider, any and local with fit", () => {
  assert.ok(!ids(search(RECORDS, q())).includes("o/tiny"));
  assert.deepEqual(ids(search(RECORDS, q({ where: "provider" }))), ["m/video"]);
  const fit = { "o/tiny": { verdict: "ok" }, "o/open": { verdict: "no" } };
  assert.deepEqual(ids(search(RECORDS, q({ where: "local" }), fit)).filter((i) => i.startsWith("o/")), ["o/tiny"]);
  assert.ok(ids(search(RECORDS, q({ where: "local", onlyFits: false }), fit)).includes("o/open"));
});

test("requirements: spanish only when declared, free, commercial, open, long", () => {
  const any = q({ where: "any" });
  const es = ids(search(RECORDS, { ...any, reqs: ["spanish"] }));
  assert.deepEqual(es, ["o/tiny"]);                          // declares es; undeclared and en-only are out
  assert.deepEqual(ids(search(RECORDS, q({ reqs: ["free"] }))), ["o/open"]);
  assert.ok(!ids(search(RECORDS, { ...any, reqs: ["commercial"] })).includes("a/asr-good"));   // licence unknown
  assert.ok(ids(search(RECORDS, { ...any, reqs: ["commercial"] })).includes("v/smart"));       // API: provider terms
  assert.deepEqual(ids(search(RECORDS, q({ reqs: ["long"] }))), []);
  assert.ok(!requirementsFor("local").some((r) => r.key === "free"));
});

test("speech: lowest error first", () => {
  assert.deepEqual(ids(search(RECORDS, q({ where: "any", caps: ["audio_in"] }))), ["a/asr-good", "a/asr-popular"]);
});

test("max price only on OpenRouter", () => {
  assert.deepEqual(ids(search(RECORDS, q({ caps: ["tools"], maxPrice: 3 }))), ["v/coder"]);
  assert.equal(price(RECORDS[1]).value, 1.75);
});

test("counts show what each checkbox would leave", () => {
  const r = search(RECORDS, q());
  assert.equal(r.counts.caps.pdf, 2);
  assert.equal(r.counts.caps.video_gen, 0);
  assert.equal(r.counts.where.provider, 1);
});

test("nothing matches: suggest the change that brings back the most", () => {
  const r = search(RECORDS, q({ caps: ["video_gen"] }));
  assert.equal(r.count, 0);
  assert.equal(r.suggestion.kind, "where");
  const r2 = search(RECORDS, q({ caps: ["pdf"], reqs: ["free"] }));
  assert.equal(r2.suggestion.kind, "req");
  assert.equal(r2.suggestion.key, "free");
});

test("reasons and pros/cons come from the data", () => {
  const r = search(RECORDS, q({ caps: ["code"] }));
  assert.match(reason(r.results[0], q({ caps: ["code"] }), r.count), /índice de programación: 58 · 1\.º de 4/);
  const open = r.results.find((s) => s.rec.id === "o/open");
  const pc = prosCons(open, q({ caps: ["code"] }), r.results);
  assert.ok(pc.pros.includes("Tiene versión gratis en OpenRouter (con límites de uso)"));
  assert.ok(pc.pros.includes("Licencia apache-2.0: permite uso comercial"));
  const smart = prosCons(r.results.find((s) => s.rec.id === "v/smart"), q({ caps: ["code"] }), r.results);
  assert.ok(smart.cons.includes("No se puede descargar: solo por internet (API)"));
});

test("matches never passes unknown data as a requirement", () => {
  assert.equal(matches(rec("z/z"), q({ where: "any", reqs: ["long"] })), false);
});

// --- harnesses ---
import { readFileSync } from "node:fs";
import { harnessOk, harnessSteps, counts } from "../../src/selector_ia/web/finder.js";

const HARNESSES = Object.fromEntries(JSON.parse(readFileSync(
  new URL("../../src/selector_ia/web/harnesses.json", import.meta.url))).harnesses.map((h) => [h.id, h]));
const CLAUDE = rec("anthropic/claude-x", { capabilities: ["chat", "code", "tools"],
  openrouter: or("anthropic/claude-x", 3, 15, { tools: true }), benchmarks: { intelligence: 55 } });
const WITH_CLAUDE = [...RECORDS, CLAUDE];

test("every harness has official docs and setup steps", () => {
  for (const h of Object.values(HARNESSES)) {
    assert.match(h.docs, /^https:\/\//, h.id);
    assert.ok(h.steps.length, h.id);
  }
});

test("Claude Code: only Anthropic models unless the student asks for the rest", () => {
  const r = search(WITH_CLAUDE, q({ harness: "claude-code" }), null, HARNESSES);
  assert.deepEqual(ids(r), ["anthropic/claude-x"]);
  const all = search(WITH_CLAUDE, q({ harness: "claude-code", harnessStrict: false }), null, HARNESSES);
  assert.ok(all.count > 1);
  assert.equal(r.counts.strict, all.count);
});

test("other agents accept any text model on OpenRouter", () => {
  assert.equal(search(WITH_CLAUDE, q({ harness: "cline" }), null, HARNESSES).count,
               search(WITH_CLAUDE, q({ caps: ["chat"] }), null, HARNESSES).count);
  assert.equal(harnessOk(rec("m/video", { capabilities: ["video_gen"] }), HARNESSES.cline), false);
});

test("setup steps carry the model id", () => {
  const steps = harnessSteps(HARNESSES.aider, CLAUDE);
  assert.ok(steps.some((s) => s.code === "aider --model openrouter/anthropic/claude-x"));
  const cont = harnessSteps(HARNESSES.continue, CLAUDE).map((s) => s.code).join("\n");
  assert.match(cont, /model: anthropic\/claude-x/);
  assert.match(cont, /- tool_use/);                   // the model supports tools (OpenRouter data)
  const noTools = harnessSteps(HARNESSES.continue, RECORDS[3]).map((s) => s.code).join("\n");
  assert.doesNotMatch(noTools, /tool_use/);
});

test("counts per harness and suggestion to drop it", () => {
  const c = counts(WITH_CLAUDE, q(), null, HARNESSES);
  assert.equal(c.harness["claude-code"], 1);
  const r = search(WITH_CLAUDE, q({ harness: "claude-code", reqs: ["free"] }), null, HARNESSES);
  assert.equal(r.count, 0);
  assert.ok(r.suggestion && r.suggestion.kind !== "cap");
});

// --- community versions and uncensored ---
const COMMUNITY = [
  ...RECORDS,
  rec("fan/open-abliterated", { hf_id: "fan/open-abliterated", community: true, base_model: "o/open",
                                uncensored: true, params: 8e9, downloads: 3e5 }),
  rec("lab/open-medical", { hf_id: "lab/open-medical", community: true, base_model: "o/open", downloads: 1e3 }),
  rec("venice/uncensored", { uncensored: true, openrouter: or("venice/uncensored", 0.5, 2, { moderated: false }) }),
  rec("v/filtered-uncensored", { uncensored: true, openrouter: or("v/filtered-uncensored", 1, 1, { moderated: true }) }),
];

test("community versions are hidden unless asked for", () => {
  const all = q({ where: "any" });
  assert.ok(!ids(search(COMMUNITY, all)).some((i) => i.startsWith("fan/") || i.startsWith("lab/")));
  const com = ids(search(COMMUNITY, { ...all, community: true }));
  assert.ok(com.includes("fan/open-abliterated") && com.includes("lab/open-medical"));
  assert.equal(search(COMMUNITY, all).counts.community, search(COMMUNITY, { ...all, community: true }).count);
});

test("uncensored: declared and not moderated by the provider; brings community builds", () => {
  const r = search(COMMUNITY, q({ where: "any", reqs: ["uncensored"] }));
  assert.deepEqual(ids(r).sort(), ["fan/open-abliterated", "venice/uncensored"]);
  const pc = prosCons(r.results.find((s) => s.rec.id === "fan/open-abliterated"), q({ reqs: ["uncensored"] }), r.results);
  assert.match(pc.cons[0], /comunidad basada en o\/open/);
  assert.ok(pc.pros.includes("Declarado sin censura por su autor o proveedor"));
});

test("models found by name in all of Hugging Face are shown even if old or community", () => {
  const found = rec("dphn/dolphin-x", { hf_id: "dphn/dolphin-x", community: true, old: true, live: true });
  const r = search([...RECORDS, found], q({ where: "any", text: "dolphin" }));
  assert.deepEqual(ids(r), ["dphn/dolphin-x"]);
});

test("video models are priced per second; rerank is its own capability", () => {
  const vid = rec("v/video", { capabilities: ["video_gen"],
    openrouter: { id: "v/video", url: "u", prices: {}, video_usd_per_second: 0.02, video_resolution: "480p" } });
  const p = price(vid, ["video_gen"]);
  assert.equal(p.value, 0.02);
  assert.match(p.unit, /por segundo de vídeo \(480p/);
  const rr = rec("v/rerank", { capabilities: ["rerank"], openrouter: { id: "v/rerank", url: "u", prices: {} } });
  const r = search([vid, rr], q({ caps: ["rerank"] }));
  assert.deepEqual(ids(r), ["v/rerank"]);
  assert.equal(r.results[0].price.value, null);          // no published price -> "sin datos", never "gratis"
});

test("agentic use on own hardware ranks by the agents index", () => {
  const recs = [
    rec("o/smart", { hf_id: "o/smart", benchmarks: { intelligence: 60, agentic: 30 } }),
    rec("o/agent", { hf_id: "o/agent", benchmarks: { intelligence: 40, agentic: 50 }, mtp: true }),
  ];
  const fit = { "o/smart": { verdict: "ok" }, "o/agent": { verdict: "ok" } };
  assert.equal(ids(search(recs, q({ where: "local" }), fit))[0], "o/smart");
  const r = search(recs, q({ where: "local", use: "agent" }), fit);
  assert.equal(ids(r)[0], "o/agent");
  assert.match(r.method, /índice de agentes/);
  assert.ok(prosCons(r.results[0], q({ where: "local", use: "agent" }), r.results).pros.some((p) => p.includes("MTP")));
});
