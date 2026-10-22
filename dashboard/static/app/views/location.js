// Territory view (F21): left — header (name, state badge, reason, Підтвердити), the main camera filling the height,
// the other cameras as thumbnails, "people now" chips; right — ONE panel with tabs Події · Розмова · Газ · Рішення
// GPT, each scrolling inside the panel. Nothing scrolls the page. The skeleton is mounted once per territory; live
// events prepend to the lists (LiveList keeps the user's position), summaries update the left side in place.
// Switching territory opens "Події"; #/loc/<id>/<tab> opens a tab; scroll positions are remembered per territory.
import { bars } from "../charts.js";
import { ackLocation, api, cameras, history, locationOf, personName, renamePerson, store } from "../store.js";
import { LiveList, STATE_LABEL, dialog, h, hms, json, personColour, since } from "../ui.js";
import { describe, eventRow } from "./events.js";
import { tile } from "./grid.js";

export const PANEL_TABS = ["events", "talk", "gas", "gpt"];
const TAB_LABEL = { events: "Події", talk: "Розмова", gas: "Газ", gpt: "Рішення GPT" };
const ARROW = { rising: "↗", falling: "↘", stable: "→" };
const LEVEL_LABEL = { low: "низький", medium: "середній", high: "високий", critical: "критичний" };
const QUIET = new Set(["gas_reading", "motion", "heartbeat", "location_summary", "mic_state"]);  // not in "Події"
const memory = new Map();       // territory → {scroll: {events, talk, gpt}, cam}
let current = null;

const fmt2 = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(2));
const remember = (id) => { if (!memory.has(id)) memory.set(id, { scroll: {}, cam: null }); return memory.get(id); };

function speakerName(ev) {
  const p = ev.payload;
  if (p.speaker_status === "uncertain") return "?";
  return p.person_id ? personName(p.person_id, p.speaker_name || p.speaker) : (p.speaker_name || p.speaker || "?");
}

async function loadData(id) {
  const day = new Date(Date.now() - 24 * 3600e3).toISOString();
  const [speech, gas, events, llm] = await Promise.all([
    history({ location_id: id, types: "speech", since: day, limit: 500 }),
    history({ location_id: id, types: "gas_reading", since: new Date(Date.now() - 900e3).toISOString(), limit: 1000 }),
    history({ location_id: id, since: new Date(Date.now() - 6 * 3600e3).toISOString(), limit: 1500 }),
    history({ location_id: id, types: "llm_decision", since: day, limit: 100 })]);
  return { speech, gas: gas.reverse(), events: events.filter((e) => !QUIET.has(e.type)), llm };   // lists: newest first
}

// --- F17 corridor, F19 source, F16 trust (also used by other views) ------------------------------------------------
export function corridorBand(d) {
  if (d.risk_floor === undefined || d.risk_final === undefined) return null;
  const pct = (v) => `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
  return h("div", { class: "corridor", role: "img",
                    "aria-label": `коридор ${fmt2(d.risk_floor)}–${fmt2(d.risk_ceil)}, GPT ${fmt2(d.risk_llm)}, підсумок ${fmt2(d.risk_final)}` },
    h("div", { class: "band", style: { left: pct(d.risk_floor), width: pct(Math.max(0.01, d.risk_ceil - d.risk_floor)) } }),
    d.risk_llm !== null && d.risk_llm !== undefined ? h("div", { class: "mark llm", style: { left: pct(d.risk_llm) } }) : null,
    h("div", { class: "mark final", style: { left: pct(d.risk_final) } }));
}

// F24: which uncertain inputs widen the IT2 corridor (envelope_reason.widened_by)
const VAR_UA = { gas_concentration: "газ", gas_trend: "динаміка газу", motion_level: "рух", people_count: "люди",
                 unknown_share: "незнайомці", action_risk: "дії", time_of_day_risk: "час" };
function widthNote(d) {
  const by = (d.envelope_reason || {}).widened_by || [];
  if (!(d.risk_width > 0.005) || !by.length) return null;
  return h("div", { class: "muted small width-note", title: "ширина коридору: невизначеність доказів (IT2)" },
    `ширина ${fmt2(d.risk_width)} — невизначені: ${by.map((w) => `${VAR_UA[w.variable] || w.variable} +${fmt2(w.width)}`).join(", ")}`);
}

function envelopeReasons(d) {
  const r = d.envelope_reason || {};
  const items = [...(r.fired || []).map((x) => `правило: ${x}`), ...(r.confirmation || []).map((x) => `підтвердження: ${x}`),
                 ...(r.ignored_claims || []).map((c) => `проігноровано (τ ${c.trust ?? "?"}): ${c.provenance} «${c.text}»`)];
  return items.length ? h("ul", { class: "env-reasons" }, items.map((x) => h("li", {}, x))) : null;
}

export function sourceBadge(p) {
  const fuzzy = p.source === "fuzzy_only";
  return h("span", { class: `src ${fuzzy ? "fuzzy" : "llm"}`, title: fuzzy ? "рішення без GPT: нечіткі правила" : "оцінка GPT" },
    fuzzy ? "без GPT" : "LLM");
}

function decisionCard(p, ts) {
  const final = p.risk_level_final || p.risk_level;
  return h("div", { class: `decision ${final}` },
    h("div", { class: "row" }, sourceBadge(p), h("b", { class: `lvl ${final}` }, (LEVEL_LABEL[final] || final || "").toUpperCase()),
      final !== p.risk_level && p.source !== "fuzzy_only" ? h("span", { class: "muted" }, `GPT: ${LEVEL_LABEL[p.risk_level] || p.risk_level}`) : null,
      h("span", { class: "muted mono" }, hms(ts))),
    h("div", {}, p.summary),
    corridorBand(p),
    p.risk_floor !== undefined ? h("div", { class: "muted small" },
      p.source === "fuzzy_only" ? `коридор ${fmt2(p.risk_floor)}–${fmt2(p.risk_ceil)} · ризик = floor ${fmt2(p.risk_final)}`
        : `коридор ${fmt2(p.risk_floor)}–${fmt2(p.risk_ceil)} · GPT ${fmt2(p.risk_llm)} → ${fmt2(p.risk_final)} · c = ${fmt2(p.confirmation)}`,
      p.over_reaction ? " · GPT, можливо, перебільшує" : "") : null,
    widthNote(p),
    p.envelope_applied ? h("details", { class: "rejected" },
      h("summary", {}, h("span", { class: "badge-rejected" }, "зниження відхилено")), envelopeReasons(p)) : null,
    p.snapshot_id ? h("button", { class: "link", onclick: () => snapshotDialog(p.snapshot_id) }, "знімок, надісланий у GPT") : null);
}

async function snapshotDialog(id) {
  try {
    const j = await api("GET", `/api/snapshot/${encodeURIComponent(id)}`);
    const text = j.digest_sent_to_gpt || "—";
    let pretty = text;
    try { pretty = text.startsWith("{") ? JSON.stringify(JSON.parse(text), null, 1) : text; } catch (e) { /* text */ }
    dialog(`Знімок ${id}`, h("p", { class: "muted" }, "Надіслано в GPT:"), h("pre", {}, pretty), json(j.snapshot));
  } catch (e) { dialog("Знімок", h("p", {}, e.message)); }
}

export function trustBadge(p) {
  if (p.trust === undefined || p.trust === null) return null;
  const level = p.trust >= 0.6 ? "high" : p.trust >= 0.3 ? "medium" : "low";
  const claim = { deescalation: " ⚠ заспокоює", instruction_to_system: " ⚠ команда системі", escalation: " ‼" }[p.claim_type] || "";
  return h("span", { class: `trust t-${level}`, title: (p.trust_reasons || []).join("\n") }, `τ ${Number(p.trust).toFixed(2)}${claim}`);
}

const speechRow = (ev) => h("div", { class: "talk-row" },
  h("span", { class: "mono muted" }, hms(ev.payload.start || ev.timestamp)),
  h("b", { style: { color: personColour(ev.payload.person_id) } }, speakerName(ev)),
  h("span", { class: "talk-text" }, ev.payload.text), trustBadge(ev.payload));
const gptRow = (ev) => decisionCard(ev.payload, ev.timestamp);

// --- skeleton ---------------------------------------------------------------------------------------------------
function mount(root, id, tab, go) {
  const mem = remember(id);
  const lists = {
    events: new LiveList({ empty: "подій немає", label: "події території", onScroll: (y) => { mem.scroll.events = y; } }),
    talk: new LiveList({ empty: "тиша", label: "розмова", onScroll: (y) => { mem.scroll.talk = y; } }),
    gpt: new LiveList({ empty: "рішень ще немає", label: "рішення GPT", onScroll: (y) => { mem.scroll.gpt = y; } }),
  };
  const refs = {
    head: h("header", { class: "loc-head" }), cam: h("div", { class: "main-cam" }), thumbs: h("div", { class: "thumbs" }),
    people: h("div", { class: "people-chips", "aria-label": "люди зараз" }),
    gasNow: h("div", { class: "gas-now" }), gasChart: h("canvas", { class: "gas-bars", dataset: { chart: "bars" } }),
    gptNow: h("div", { class: "gpt-now" }),
  };
  const tabs = h("div", { class: "panel-tabs", role: "tablist", "aria-label": "розділи території" },
    PANEL_TABS.map((t) => h("a", { role: "tab", href: `#/loc/${encodeURIComponent(id)}/${t}`, dataset: { tab: t },
                                   id: `tab-${t}`, "aria-controls": `pane-${t}` }, TAB_LABEL[t])));
  const panes = {
    events: h("div", { class: "pane", role: "tabpanel", id: "pane-events" }, lists.events.el),
    talk: h("div", { class: "pane", role: "tabpanel", id: "pane-talk" }, lists.talk.el),
    gas: h("div", { class: "pane gas-pane", role: "tabpanel", id: "pane-gas" }, refs.gasNow,
           h("div", { class: "muted small" }, "останні 10 хв"), refs.gasChart),
    gpt: h("div", { class: "pane gpt-pane", role: "tabpanel", id: "pane-gpt" }, refs.gptNow,
           h("h2", {}, "Історія"), lists.gpt.el),
  };
  const page = h("div", { class: "loc-view", dataset: { view: "loc", loc: id } },
    h("section", { class: "loc-left" }, refs.head, refs.cam, refs.thumbs, refs.people),
    h("aside", { class: "panel loc-right" }, tabs, h("div", { class: "panes" }, Object.values(panes))));
  root.replaceChildren(page);
  current = { id, root, page, lists, refs, tabs, panes, tab: null, data: null, go };
  selectTab(tab || "events");
  loadData(id).then((data) => {
    if (!current || current.id !== id) return;
    current.data = data;
    lists.events.set(data.events.map((ev) => eventRow(ev)));
    lists.talk.set(data.speech.map(speechRow));
    lists.gpt.set(data.llm.map(gptRow));
    for (const [key, list] of Object.entries(lists)) list.list.scrollTop = mem.scroll[key] || 0;   // inside the panel
    drawGas();
  }).catch(() => {});
}

function selectTab(tab) {
  current.tab = tab;
  current.tabs.querySelectorAll("[role=tab]").forEach((a) => {
    const on = a.dataset.tab === tab;
    a.classList.toggle("on", on); a.setAttribute("aria-selected", on ? "true" : "false"); a.tabIndex = on ? 0 : -1;
  });
  for (const [name, pane] of Object.entries(current.panes)) pane.hidden = name !== tab;
  if (tab === "gas") requestAnimationFrame(drawGas);
}

// --- live parts -------------------------------------------------------------------------------------------------
function gasThreshold(location, gasName) {
  const rule = ((location.rules || {}).gas_thresholds || {})[gasName];
  return rule ?? (store.status.gas_thresholds || {})[gasName] ?? null;
}

function drawGas() {
  if (!current || current.tab !== "gas" || !current.data) return;
  const location = locationOf(current.id) || {};
  const last = current.data.gas.at(-1);
  const name = last ? last.payload.gas : null;
  const points = current.data.gas.map((e) => ({ t: Date.parse(e.timestamp), v: Number(e.payload.concentration_pct) || 0 }));
  bars(current.refs.gasChart, points, { threshold: gasThreshold(location, name) });
}

function update() {
  const { id, refs, go } = current;
  const location = locationOf(id) || { location_id: id, name: id, icon: "" };
  const s = store.summaries[id] || { state: "safe", reason: "", people: [], people_count: 0, active_alerts: 0 };
  refs.head.className = `loc-head st-${s.state}`;
  refs.head.replaceChildren(
    h("span", { class: "title" }, `${location.icon || ""} ${location.name}`.trim()),
    h("span", { class: `badge st-${s.state}` }, h("span", { class: `dot st-${s.state}`, "aria-hidden": "true" }),
      ` ${(STATE_LABEL[s.state] || s.state).toUpperCase()}`),
    h("span", { class: "reason", title: s.reason || "" }, s.reason || ""),
    h("button", { class: "ack", disabled: !s.active_alerts, "aria-label": "підтвердити тривоги території",
                  onclick: () => ackLocation(id) }, s.active_alerts ? `Підтвердити (${s.active_alerts})` : "Підтвердити"));
  const cams = cameras().filter((d) => d.location_id === id);
  const mem = remember(id);
  const main = cams.find((d) => d.device_id === mem.cam) || cams[0];
  refs.cam.replaceChildren(main ? tile(main, "full") : h("div", { class: "grid-empty muted" }, "у цій території немає камер"));
  refs.thumbs.hidden = cams.length < 2;
  refs.thumbs.replaceChildren(...cams.filter((d) => d !== main).map((d) => tile(d, "small", () => { mem.cam = d.device_id; go(null); })));
  refs.people.replaceChildren(...(s.people.length ? s.people.map(personChip) : [h("span", { class: "muted" }, "нікого")]));
  const g = s.gas;
  refs.gasNow.replaceChildren(g ? h("div", { class: "gas-big" }, h("b", {}, `${g.concentration_pct ?? "?"} %`), ` ${g.main || ""} `,
      h("span", { class: `trend ${g.trend || ""}` }, `${ARROW[g.trend] || ""} ${g.trend || ""}`),
      gasThreshold(location, g.main) !== null ? h("span", { class: "muted" }, ` · поріг ${gasThreshold(location, g.main)} %`) : null)
    : h("span", { class: "muted" }, "немає газового датчика"));
  refs.gptNow.replaceChildren(s.last_llm ? decisionCard(s.last_llm, s.last_llm.ts) : h("div", { class: "muted" }, "ще немає оцінки"));
}

function personChip(p) {
  const name = personName(p.person_id, p.name);
  const unknown = !p.person_id || String(name).startsWith("unknown_") || name === "?";
  const deviceName = (store.devices.find((d) => d.device_id === p.device_id) || {}).name || "";
  const label = h("span", { class: "pname", style: { color: personColour(p.person_id) } }, name);
  return h("span", { class: `person-chip${unknown ? " unknown" : ""}`, title: deviceName },
    label, ` · ${p.action || "—"}`, p.since ? h("span", { class: "muted" }, ` · ${since(p.since)}`) : null,
    p.person_id ? h("button", { class: "icon", "aria-label": `перейменувати ${name}`, onclick: () => {
      const input = h("input", { value: name, "aria-label": "нове ім'я" });
      input.onkeydown = async (e) => {
        e.stopPropagation();
        if (e.key === "Enter" && input.value.trim()) { try { await renamePerson(p.person_id, input.value.trim()); } catch (err) { alert(err.message); } }
        if (e.key === "Enter" || e.key === "Escape") input.replaceWith(label);
      };
      label.replaceWith(input); input.focus(); input.select();
    } }, "✎") : null);
}

export function renderLocation(root, { go, id, tab }) {
  if (!current || current.id !== id || !root.contains(current.page)) mount(root, id, tab, go);
  else if (tab && tab !== current.tab) selectTab(tab);
  current.go = go;
  update();
}

export function onLocationEvent(ev) {
  if (!current || !current.data || ev.location_id !== current.id) return;
  if (ev.type === "gas_reading") {
    current.data.gas.push(ev);
    if (current.data.gas.length > 1500) current.data.gas.splice(0, 500);
    drawGas();
    return;
  }
  if (QUIET.has(ev.type)) return;
  current.lists.events.prepend([eventRow(ev)]);
  if (ev.type === "speech") current.lists.talk.prepend([speechRow(ev)]);
  if (ev.type === "llm_decision") current.lists.gpt.prepend([gptRow(ev)]);
}

export function redrawCharts() { drawGas(); }
export function leaveLocation() { current = null; }
export { describe };
