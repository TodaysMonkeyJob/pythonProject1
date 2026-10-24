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
    case "notification": return `${p.title || ""}: ${p.body || ""}`;
    case "person_identified": return `#${p.track_id} ${personName(p.person_id, p.identity)}${p.known ? "" : " (незнайомець)"}`;
    case "object_detected": return `${p.class} #${p.track_id} ${p.state === "disappeared" ? "зник" : "з'явився"}`;
    case "action": return `#${p.track_id} ${p.action}`;
    case "speech": return `«${p.text || ""}»`;
    case "gas_reading": return `${p.gas} ${p.concentration_pct}% ${p.trend || ""}`;
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
