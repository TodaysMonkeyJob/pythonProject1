// Location page: a one-viewport summary (cameras, people, transcripts, gas, GPT) and the details below it
// (transcript log, event table, GPT history, gas chart, state timeline, device health). CSS scroll-snap between them.
import { lineChart, sparkline, timeline } from "../charts.js";
import { ackLocation, api, cameras, history, locationOf, personName, renamePerson, store } from "../store.js";
import { STATE_LABEL, dialog, downloadCsv, h, hms, json, personColour, since } from "../ui.js";
import { cameraGrid } from "./grid.js";

const ARROW = { rising: "↗", falling: "↘", stable: "→" };
let current = null;         // {id, focus, data: {speech, gas, events, llm, states, restarts}, filters}

function speakerName(ev) {
  const p = ev.payload;
  if (p.speaker_status === "uncertain") return "?";
  return p.person_id ? personName(p.person_id, p.speaker_name || p.speaker) : (p.speaker_name || p.speaker || "?");
}

export async function loadLocationData(id) {
  const day = new Date(Date.now() - 24 * 3600e3).toISOString();
  const hour = new Date(Date.now() - 3600e3).toISOString();
  const [speech, gas, events, llm, states, alerts] = await Promise.all([
    history({ location_id: id, types: "speech", since: day, limit: 2000 }),
    history({ location_id: id, types: "gas_reading", since: hour, limit: 4000 }),
    history({ location_id: id, since: new Date(Date.now() - 6 * 3600e3).toISOString(), limit: 3000 }),
    history({ location_id: id, types: "llm_decision", since: day, limit: 200 }),
    history({ location_id: id, types: "location_state_changed", since: day, limit: 500 }),
    history({ types: "alert", since: day, limit: 1000 })]);
  return { speech: speech.reverse(), gas: gas.reverse(), events: events.reverse(), llm, states: states.reverse(),
           restarts: alerts.filter((e) => e.payload.reason === "module_restarted") };
}

// live events of this location are appended while the page is open
export function onLocationEvent(ev) {
  if (!current || ev.location_id !== current.id || !current.data) return false;
  const d = current.data;
  if (ev.type === "speech") d.speech.push(ev);
  if (ev.type === "gas_reading") d.gas.push(ev);
  if (ev.type === "llm_decision") d.llm.unshift(ev);
  if (ev.type === "location_state_changed") d.states.push(ev);
  d.events.push(ev);
  return ["speech", "gas_reading", "llm_decision", "location_state_changed"].includes(ev.type);
}

export function renderLocation(root, { id, details, focus, go, neighbours }) {
  if (!current || current.id !== id) {
    current = { id, focus: focus || null, data: null, filters: { speaker: "", range: 24, search: "", source: "", type: "",
                                                                 severity: "", device: "" } };
    loadLocationData(id).then((data) => { if (current && current.id === id) { current.data = data; renderLocation(root, { id, go, neighbours }); } })
      .catch(() => {});
  } else if (focus !== undefined && focus !== null) current.focus = focus;
  const location = locationOf(id) || { location_id: id, name: id, icon: "" };
  const s = store.summaries[id] || { state: "safe", reason: "", people: [], people_count: 0, devices: { online: 0, total: 0 } };
  const existing = root.querySelector(".loc-page");
  if (existing && existing.dataset.loc === id) {
    // live update: only the summary; the details keep their filters and scroll position
    existing.querySelector(".summary").replaceWith(summarySection(location, s, go, neighbours));
    if (current.data && existing.querySelector(".details").dataset.loaded !== "1")
      existing.querySelector(".details").replaceWith(detailsSection(location));
    if (details) existing.querySelector(".details").scrollIntoView();
    drawCharts(existing, location);
    return;
  }
  const page = h("div", { class: "loc-page", dataset: { loc: id } },
    summarySection(location, s, go, neighbours), detailsSection(location));
  root.replaceChildren(page);
  if (details) requestAnimationFrame(() => page.querySelector(".details").scrollIntoView());
  drawCharts(page, location);
}

function summarySection(location, s, go, neighbours) {
  const cams = cameras().filter((d) => d.location_id === location.location_id);
  const grid = cameraGrid(cams, { focus: current.focus, onTile: (d) => {
    current.focus = current.focus === d.device_id ? null : d.device_id; go(null); } });
  return h("section", { class: "summary" },
    h("header", { class: `loc-head st-${s.state}` },
      h("button", { class: "nav", title: "←", onclick: () => go(neighbours.prev) }, "◀"),
      h("span", { class: "title" }, `${location.icon || ""} ${location.name}`.trim()),
      h("span", { class: `badge st-${s.state}` }, `● ${STATE_LABEL[s.state] || s.state}`),
      h("span", { class: "reason" }, s.reason ? `— ${s.reason}` : ""),
      s.active_alerts ? h("button", { class: "ack", onclick: () => ackLocation(location.location_id) }, `Підтвердити (${s.active_alerts})`) : null,
      h("button", { class: "nav", title: "→", onclick: () => go(neighbours.next) }, "▶")),
    h("div", { class: "loc-body" },
      h("section", { class: "panel cams" }, grid),
      h("aside", { class: "panel side" },
        h("h2", {}, `Люди зараз (${s.people_count})`), peopleNow(s),
        h("h2", {}, "Останні фрази"), lastSpeech(),
        h("h2", {}, "Газ"), gasNow(s),
        h("h2", {}, "GPT"), s.last_llm ? h("div", {}, h("b", {}, s.last_llm.risk_level), ` — ${s.last_llm.summary} `,
                                           h("span", { class: "muted" }, `(${hms(s.last_llm.ts)})`))
                                     : h("div", { class: "muted" }, "ще немає оцінки"))),
    h("button", { class: "to-details", onclick: (e) => e.target.closest(".loc-page").querySelector(".details").scrollIntoView({ behavior: "smooth" }) },
      "▼ Деталі"));
}

function peopleNow(s) {
  if (!s.people.length) return h("div", { class: "muted" }, "нікого");
  const deviceName = (id) => (store.devices.find((d) => d.device_id === id) || {}).name || "";
  return h("div", { class: "people-now" }, s.people.map((p) => {
    const name = h("span", { class: "pname", style: { color: personColour(p.person_id) } }, personName(p.person_id, p.name));
    const edit = p.person_id ? h("button", { class: "icon", title: "перейменувати", onclick: () => {
      const input = h("input", { value: personName(p.person_id, p.name) });
      input.onkeydown = async (e) => {
        if (e.key === "Enter" && input.value.trim()) { try { await renamePerson(p.person_id, input.value.trim()); } catch (err) { alert(err.message); } }
        if (e.key === "Enter" || e.key === "Escape") input.replaceWith(name);
        e.stopPropagation();
      };
      name.replaceWith(input); input.focus(); input.select();
    } }, "✎") : null;
    return h("div", { class: "row" }, name, edit, h("span", {}, p.action || "—"),
             h("span", { class: "muted" }, p.since ? since(p.since) : ""), h("span", { class: "muted" }, deviceName(p.device_id)));
  }));
}

function lastSpeech() {
  const items = current.data ? current.data.speech.slice(-5).reverse() : [];
  if (!items.length) return h("div", { class: "muted" }, current.data ? "тиша" : "завантажую…");
  return h("div", { class: "speech" }, items.map((ev) => h("div", {},
    h("span", { class: "muted" }, hms(ev.payload.start || ev.timestamp) + " "),
    h("b", { style: { color: personColour(ev.payload.person_id) } }, speakerName(ev)), `: "${ev.payload.text}"`)));
}

function gasNow(s) {
  const g = s.gas;
  return h("div", { class: "gas-now" },
    g ? h("span", {}, `${g.main} · ${g.concentration_pct ?? "?"}% · ${ARROW[g.trend] || ""} ${g.trend || ""}`) : h("span", { class: "muted" }, "немає датчика"),
    h("canvas", { class: "spark", dataset: { chart: "spark" } }));
}

function gasThreshold(location, gasName) {
  const rule = ((location.rules || {}).gas_thresholds || {})[gasName];
  return rule ?? (store.status.gas_thresholds || {})[gasName] ?? null;
}

function drawCharts(page, location) {
  if (!current.data) return;
  const points = (since) => current.data.gas.filter((e) => Date.parse(e.timestamp) >= since)
    .map((e) => ({ t: Date.parse(e.timestamp), v: Number(e.payload.concentration_pct) || 0 }));
  const gasName = (current.data.gas.at(-1) || { payload: {} }).payload.gas;
  const spark = page.querySelector('[data-chart="spark"]');
  if (spark) sparkline(spark, points(Date.now() - 600e3), { threshold: gasThreshold(location, gasName) });
  const chart = page.querySelector('[data-chart="gas"]');
  if (chart) lineChart(chart, points(Date.now() - 3600e3), { threshold: gasThreshold(location, gasName), label: gasName ? `${gasName}, %` : "" });
  const bar = page.querySelector('[data-chart="timeline"]');
  if (bar) {
    const to = Date.now(), from = to - 24 * 3600e3, segments = [];
    let state = current.data.states.length ? current.data.states[0].payload.from : "safe", start = from;
    for (const ev of current.data.states) { const t = Date.parse(ev.timestamp); segments.push({ from: start, to: t, state }); state = ev.payload.to; start = t; }
    segments.push({ from: start, to, state });
    timeline(bar, segments, from, to);
  }
}

export function redrawCharts(root) {
  const page = root.querySelector(".loc-page");
  if (page && current) drawCharts(page, locationOf(current.id) || {});
}

// --- details --------------------------------------------------------------------------------------------------------

function detailsSection(location) {
  if (!current.data) return h("section", { class: "details" }, h("div", { class: "muted" }, "завантажую історію…"));
  return h("section", { class: "details", dataset: { loaded: "1" } },
    h("div", { class: "filters" }, h("button", { onclick: async (e) => {
      current.data = await loadLocationData(current.id);
      e.target.closest(".details").replaceWith(detailsSection(location));
      drawCharts(document.querySelector(".loc-page"), location);
    } }, "↻ оновити історію")),
    h("div", { class: "details-grid" },
      h("div", { class: "panel wide" }, h("h2", {}, "Транскрипти"), transcriptLog(location)),
      h("div", { class: "panel wide" }, h("h2", {}, "Події"), eventTable()),
      h("div", { class: "panel" }, h("h2", {}, "Рішення GPT"), gptHistory()),
      h("div", { class: "panel" }, h("h2", {}, "Газ, 1 год"), h("canvas", { class: "chart", dataset: { chart: "gas" } })),
      h("div", { class: "panel" }, h("h2", {}, "Стан, 24 год"), h("canvas", { class: "bar", dataset: { chart: "timeline" } }),
        h("div", { class: "state-list" }, current.data.states.slice(-8).reverse().map((ev) =>
          h("div", {}, h("span", { class: "muted" }, hms(ev.timestamp) + " "), `${ev.payload.from} → ${ev.payload.to}: ${ev.payload.reason}`)))),
      h("div", { class: "panel" }, h("h2", {}, "Пристрої"), deviceHealth(location))));
}

function transcriptLog(location) {
  const f = current.filters, speakers = [...new Set(current.data.speech.map(speakerName))].sort();
  const box = h("div", { class: "scroll-list" });
  const fill = () => {
    const from = Date.now() - f.range * 3600e3, q = f.search.toLowerCase();
    const rows = current.data.speech.filter((ev) => Date.parse(ev.timestamp) >= from && (!f.speaker || speakerName(ev) === f.speaker)
                                             && (!q || ev.payload.text.toLowerCase().includes(q))).reverse();
    box.replaceChildren(...(rows.length ? rows.slice(0, 500).map((ev) => h("div", { class: "row" },
      h("span", { class: "muted" }, hms(ev.payload.start || ev.timestamp)),
      h("b", { style: { color: personColour(ev.payload.person_id) } }, speakerName(ev)),
      h("span", {}, ev.payload.text))) : [h("div", { class: "muted" }, "нічого")]));
    return rows;
  };
  const select = (key, options, label) => h("select", { onchange: (e) => { f[key] = e.target.value; fill(); } },
    options.map(([v, text]) => h("option", { value: v, selected: String(f[key]) === String(v) }, text ?? label)));
  const search = h("input", { placeholder: "пошук…", value: f.search, oninput: (e) => { f.search = e.target.value; fill(); } });
  const csv = h("button", { onclick: () => downloadCsv(`transcripts_${location.location_id}.csv`,
    [["time", "speaker", "person_id", "text"], ...fill().map((ev) => [ev.payload.start || ev.timestamp, speakerName(ev), ev.payload.person_id, ev.payload.text])]) }, "CSV");
  fill();
  return h("div", {}, h("div", { class: "filters" }, select("speaker", [["", "усі мовці"], ...speakers.map((n) => [n, n])]),
    select("range", [[1, "1 год"], [6, "6 год"], [24, "24 год"]]), search, csv), box);
}

const ROW = 24;
function eventTable() {
  const f = current.filters, all = current.data.events;
  const opts = (key) => [...new Set(all.map((e) => (key === "device" ? e.device_id : e[key])))].sort();
  const viewport = h("div", { class: "vtable" }), inner = h("div", { class: "vinner" });
  viewport.append(inner);
  let rows = [];
  const draw = () => {
    const start = Math.max(0, Math.floor(viewport.scrollTop / ROW) - 5), end = Math.min(rows.length, start + Math.ceil((viewport.clientHeight || 360) / ROW) + 10);
    inner.style.height = `${rows.length * ROW}px`;
    inner.replaceChildren(...rows.slice(start, end).map((ev, i) => h("div", { class: `vrow sev-${ev.severity}`, style: { top: `${(start + i) * ROW}px` },
      onclick: () => dialog(`${ev.source} ${ev.type}`, json(ev)) },
      h("span", {}, hms(ev.timestamp)), h("span", {}, ev.source), h("span", {}, ev.type), h("span", { class: "muted" }, ev.device_id),
      h("span", {}, JSON.stringify(ev.payload).slice(0, 120)))));
  };
  const apply = () => {
    rows = all.filter((e) => (!f.source || e.source === f.source) && (!f.type || e.type === f.type)
                         && (!f.severity || e.severity === f.severity) && (!f.device || e.device_id === f.device)).reverse();
    draw();
  };
  viewport.addEventListener("scroll", draw);
  const select = (key, label) => h("select", { onchange: (e) => { f[key] = e.target.value; apply(); } },
    h("option", { value: "" }, label), opts(key).map((v) => h("option", { value: v, selected: f[key] === v }, v)));
  requestAnimationFrame(apply);
  return h("div", {}, h("div", { class: "filters" }, select("source", "усі джерела"), select("type", "усі типи"),
    select("severity", "усі рівні"), select("device", "усі пристрої")), viewport);
}

function gptHistory() {
  if (!current.data.llm.length) return h("div", { class: "muted" }, "ще немає");
  return h("div", { class: "scroll-list" }, current.data.llm.slice(0, 50).map((ev) => h("div", { class: `decision ${ev.payload.risk_level}` },
    h("div", {}, h("b", {}, ev.payload.risk_level), ` · ${ev.payload.recommended_action} · `, h("span", { class: "muted" }, hms(ev.timestamp))),
    h("div", {}, ev.payload.summary),
    h("button", { onclick: async () => {
      try { const j = await api("GET", `/api/snapshot/${encodeURIComponent(ev.payload.snapshot_id)}`);
        dialog(`Знімок ${ev.payload.snapshot_id}`, h("p", { class: "muted" }, "Надіслано в GPT:"), h("pre", {}, j.digest_sent_to_gpt || "—"), json(j.snapshot));
      } catch (e) { dialog("Знімок", h("p", {}, e.message)); } } }, "знімок"))));
}

function deviceHealth(location) {
  const devices = store.devices.filter((d) => d.location_id === location.location_id);
  if (!devices.length) return h("div", { class: "muted" }, "пристроїв немає");
  const modules = (store.status || {}).modules || {};
  return h("div", { class: "health" }, devices.map((d) => {
    const name = `${d.kind === "camera" ? "vision" : d.kind === "mic" ? "audio" : "gas"}@${d.device_id}`;
    const m = modules[name] || {}, restarts = current.data.restarts.filter((e) => e.payload.module === name).length;
    return h("div", { class: "row" }, h("span", { class: `dot ${d.status === "online" ? "on" : ""}` }), h("b", {}, d.name),
      h("span", { class: "muted" }, `${d.kind} · ${d.transport}`), h("span", {}, m.frame_fps ? `${m.frame_fps} FPS` : d.status),
      h("span", { class: "muted" }, d.last_seen ? `${hms(d.last_seen)}` : ""), h("span", { class: restarts ? "warn" : "muted" }, `перезапусків: ${restarts}`));
  }));
}

export function leaveLocation() { current = null; }
