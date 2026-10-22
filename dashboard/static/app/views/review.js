// F25 (TOD-36): GPT sees a higher risk than the corridor allows → the operator accepts GPT's level (the location
// takes it, the GPT trust channel is confirmed, the outcome goes to the rule-suggestion store) or rejects it (the
// corridor's upper bound stays). Used by the decision cards and the event dialog.
import { escalationResolution, resolveEscalation } from "../store.js";
import { h, toast } from "../ui.js";

const LEVEL_LABEL = { low: "низький", medium: "середній", high: "високий", critical: "критичний" };

export function reviewPanel(reviewId, p) {
  const done = escalationResolution(reviewId);
  if (done) return h("div", { class: "muted small" },
    `оператор ${done.payload.by || ""} ${done.payload.accepted ? "прийняв рівень GPT" : "залишив межу коридору"}`);
  const level = p.risk_level_llm || p.risk_level;
  const act = (accepted) => async (e) => {
    e.stopPropagation();
    try {
      await resolveEscalation(reviewId, accepted, { location_id: p.location_id, risk_level_llm: level });
      toast("Ескалація", accepted ? "рівень GPT прийнято" : "відхилено: діє межа коридору", "safe");
    } catch (err) { toast("Помилка", err.message, "danger"); }
  };
  return h("div", { class: "review" },
    h("span", { class: "badge-review" }, "GPT бачить вищий ризик — потрібне рішення оператора"),
    (p.reasons || []).length ? h("ul", { class: "env-reasons" }, p.reasons.slice(0, 3).map((r) => h("li", {}, r))) : null,
    h("div", { class: "row-actions" },
      h("button", { onclick: act(true) }, `✓ Прийняти «${LEVEL_LABEL[level] || level}»`),
      h("button", { onclick: act(false) }, "✗ Відхилити")));
}
