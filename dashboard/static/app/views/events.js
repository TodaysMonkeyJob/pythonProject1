// Event rows shared by the Overview's global log and the territory "Події" tab (F21): one line, newest first.
import { locationOf, personName } from "../store.js";
import { STATE_LABEL, dialog, h, hms, json } from "../ui.js";
import { feedbackButtons } from "./trust.js";
import { reviewPanel } from "./review.js";

const LLM_PROBLEM = { auth_error: "ключ не прийнято", not_found: "немає деплойменту", rate_limited: "ліміт запитів",
                      unreachable: "немає з'єднання", disabled: "немає ключів" };
const LEVEL = { low: "низький", medium: "середній", high: "високий", critical: "критичний" };
// What the global log shows (routine detections stay in the territory tab)
const GLOBAL = new Set(["llm_status", "alert", "location_state_changed", "llm_decision", "deescalation_rejected", "notification",
                        "escalation_review", "escalation_resolved",
                        "device_offline", "device_online", "person_identified", "action"]);

export const forGlobalLog = (ev) => GLOBAL.has(ev.type) && !(ev.type === "person_identified"
  && ["not_visible", "unidentified"].includes(ev.payload.identity)) && !(ev.type === "action"
  && !["lying", "possible_smoking"].includes(ev.payload.action));

// G1 (TOD-53): raw value in the sensor's unit + «% від критичного рівня» (graphene vol % or driver readings)
const UNIT = { vol_pct: "об.%", ppm: "ppm", mg_m3: "мг/м³", ug_m3: "мкг/м³" };
function gasText(p) {
  const share = (x) => (x !== undefined && x !== null ? ` (${x} % від критичного рівня)` : "");
  const items = (p.readings || []).length && p.concentration_pct === undefined
    ? p.readings.map((r) => `${r.gas} ${r.concentration} ${UNIT[r.unit] || r.unit || ""}${share(r.pct_of_crit)}`)
    : [`${p.gas} ${p.concentration_pct ?? "?"} об.%${share(p.pct_of_crit)}${p.below_model_range ? " — нижче діапазону моделі" : ""}`];
  return `${items.join(", ")} ${p.trend || ""}`.trim();
}

export function describe(ev) {
  const p = ev.payload || {};
  switch (ev.type) {
    case "llm_status": return p.status === "ok" ? "GPT знову доступний"
      : `GPT недоступний: ${LLM_PROBLEM[p.status] || p.status}${p.http_status ? ` (HTTP ${p.http_status})` : ""}`;
    case "alert": return `тривога: ${p.reason || ""} ${p.gas || p.module || ""}`.trim();
    case "location_state_changed": return `${STATE_LABEL[p.from] || p.from} → ${STATE_LABEL[p.to] || p.to}: ${p.reason || ""}`;
    case "llm_decision": return `${p.source === "fuzzy_only" ? "без GPT" : "GPT"}: ${LEVEL[p.risk_level_final || p.risk_level] || ""} — ${p.summary || ""}`;
    case "escalation_review": return `GPT бачить вищий ризик (${LEVEL[p.risk_level_llm] || p.risk_level_llm}) — потрібне рішення оператора`;
    case "escalation_resolved": return `оператор ${p.by || ""}: ескалацію ${p.accepted ? "прийнято" : "відхилено"}`;
    case "deescalation_rejected": return `зниження відхилено: GPT ${LEVEL[p.risk_level_llm] || p.risk_level_llm} → ${LEVEL[p.risk_level_final] || p.risk_level_final}`;
    case "hfs_risk": return `нечітка система: ${LEVEL[p.level] || p.level} ${Number(p.overall).toFixed(2)} (газ ${p.risk1 ?? "—"} · камера ${p.risk2 ?? "—"} · звук ${p.risk3 ?? "—"})${p.partial ? ", часткова" : ""}`;
    case "notification": return `${p.title || ""}: ${p.body || ""}`;
    case "person_identified": return `#${p.track_id} ${personName(p.person_id, p.identity)}${p.known ? "" : " (незнайомець)"}`;
    case "object_detected": return `${p.class} #${p.track_id} ${p.state === "disappeared" ? "зник" : "з'явився"}`;
    case "action": return `#${p.track_id} ${p.action}`;
    case "speech": return `«${p.text || ""}»`;
    case "gas_reading": return gasText(p);
    case "motion": return `рух ${p.state === "end" ? "припинився" : "почався"}`;
    case "device_offline": return `пристрій офлайн: ${p.name || p.device_id}`;
    case "device_online": return `пристрій онлайн: ${p.name || p.device_id}`;
    default: return JSON.stringify(p).slice(0, 120);
  }
}

export function eventRow(ev, { withLocation = false } = {}) {
  const location = withLocation ? locationOf(ev.location_id) : null;
  return h("button", { class: `ev-row sev-${ev.severity}`, title: "деталі події", onclick: () => dialog(`${ev.source} · ${ev.type}`,
    ev.type === "escalation_review" ? reviewPanel(ev.payload.review_id, ev.payload) : null, feedbackButtons(ev), json(ev)) },
    h("span", { class: "mono muted" }, hms(ev.timestamp)),
    withLocation ? h("span", { class: "ev-loc" }, location ? location.name : ev.location_id) : null,
    h("span", { class: "ev-text" }, describe(ev)));
}
