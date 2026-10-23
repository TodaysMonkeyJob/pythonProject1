// F27 (TOD-38) page «Пропозиції правил»: GPT's rule suggestions from the separate review store (never in the live
// feed). Filters by status / location / confidence / support; a suggestion shows its rule, rationale, supporting
// cases, the validator report (live while editing) and the gate report. Reject, edit (re-validated) or send to the
// TOD-37 approval flow (checks + regression gate in the background) — approval itself stays `python -m tools.rules approve`.
import { api } from "../store.js";
import { clear, h, hms, toast } from "../ui.js";

const STATUS = { new: "нова", rejected: "відхилена", edited: "відредагована", sent_to_approval: "на схваленні",
                 approved: "схвалена" };
const CONFIDENCE = { low: "низька", medium: "середня", high: "висока" };
const TRIGGER = { accepted_escalation: "прийнята ескалація", repeated_escalation: "повторні ескалації",
                  coverage_gap: "прогалина покриття", manual: "вручну" };
const filters = { status: "", location: "", confidence: "", min_support: 1 };
let selected = null;
let poll = null;

export async function renderSuggestions(view) {
  clearInterval(poll);
  const page = h("section", { class: "admin rules-page suggestions-page" });
  clear(view, page);
  const list = h("div", { class: "suggestion-list" });
  const detail = h("div", { class: "suggestion-detail" });
  const select = (name, options) => h("select", { "aria-label": name, onchange: (e) => { filters[name] = e.target.value; load(); } },
    h("option", { value: "" }, "усі"),
    Object.entries(options).map(([k, v]) => h("option", { value: k, selected: filters[name] === k }, v)));
  clear(page,
    h("div", { class: "admin-head" },
      h("h2", {}, "Пропозиції правил"),
      h("span", { class: "muted small" }, "від GPT; нічого не змінюється без схвалення (python -m tools.rules approve)"),
      h("label", {}, "статус ", select("status", STATUS)),
      h("label", {}, "впевненість ", select("confidence", CONFIDENCE)),
      h("input", { type: "search", placeholder: "локація", value: filters.location, "aria-label": "локація",
                   onchange: (e) => { filters.location = e.target.value.trim(); load(); } }),
      h("label", {}, "підтримка ≥ ", h("input", { type: "number", min: 1, value: filters.min_support, style: { width: "4em" },
                                                   onchange: (e) => { filters.min_support = Number(e.target.value) || 1; load(); } }))),
    h("div", { class: "suggestion-columns" }, list, detail));

  async function load() {
    const q = new URLSearchParams(Object.entries(filters).filter(([, v]) => v !== "" && v !== null));
    let data;
    try { data = await api("GET", `/api/suggestions?${q}`); }
    catch (e) { clear(list, h("p", { class: "muted" }, `Пропозиції недоступні: ${e.message}`)); return; }
    clear(list, data.suggestions.length ? h("table", { class: "rules-table" },
      h("thead", {}, h("tr", {}, ["#", "статус", "×", "впевн.", "локація", "правило"].map((t) => h("th", {}, t)))),
      h("tbody", {}, data.suggestions.map((s) => h("tr", { class: s.id === selected ? "selected" : "", style: { cursor: "pointer" },
                                                         onclick: () => { selected = s.id; load(); } },
        h("td", { class: "mono" }, s.id), h("td", {}, STATUS[s.status] || s.status),
        h("td", { class: "mono" }, s.support_count), h("td", {}, CONFIDENCE[s.confidence] || "—"),
        h("td", { class: "mono small" }, s.location),
        h("td", { class: `mono small ${s.validator_report.valid ? "" : "warn-text"}` }, s.rule || `(потрібна змінна: ${s.new_variable_needed || "?"})`)))))
      : h("p", { class: "muted" }, "Пропозицій немає."));
    if (selected !== null) showDetail(selected, data.sending.includes(selected));
  }

  async function showDetail(id, sending) {
    let s;
    try { s = await api("GET", `/api/suggestions/${id}`); }
    catch (e) { clear(detail, h("p", { class: "muted" }, e.message)); return; }
    const report = h("div", { class: "small" });
    const drawReport = (r) => clear(report,
      h("b", { class: r.valid ? "" : "warn-text" }, r.valid ? "валідне" : "невалідне"),
      (r.errors || []).map((x) => h("div", { class: "warn-text" }, x)),
      (r.warnings || []).map((x) => h("div", { class: "muted" }, x)));
    drawReport(s.validator_report);
    const locked = ["rejected", "sent_to_approval", "approved"].includes(s.status);
    const rule = h("textarea", { rows: 3, class: "mono", style: { width: "100%" }, disabled: locked,
                                 "aria-label": "правило" }, s.rule);
    let timer = null;
    rule.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(async () => drawReport(await api("POST", `/api/suggestions/${id}/validate`, { rule: rule.value })), 300);
    });
    const reason = h("input", { type: "text", placeholder: "причина відхилення", "aria-label": "причина" });
    const act = async (method, url, body, message) => {
      try { await api(method, url, body); toast("Пропозиції правил", message, "safe"); }
      catch (e) { toast("Пропозиції правил", e.message, "danger"); }
      load();
    };
    const gate = s.gate_report;
    clear(detail,
      h("h3", {}, `#${s.id} · ${STATUS[s.status] || s.status}`),
      h("p", { class: "small muted" }, `${TRIGGER[s.trigger] || s.trigger || ""} · ${hms(s.created_at)} · підтримка ×${s.support_count}`,
        ` · впевненість ${CONFIDENCE[s.confidence] || "—"} · локація ${s.location}`,
        s.reviewer ? ` · ${s.reviewer}` : "", s.review_reason ? ` (${s.review_reason})` : ""),
      rule,
      s.original_rule && s.original_rule !== s.rule ? h("p", { class: "small muted mono" }, `від GPT: ${s.original_rule}`) : null,
      s.new_variable_needed ? h("p", { class: "small" }, `Потрібна нова змінна: ${s.new_variable_needed}`) : null,
      report,
      h("p", {}, s.rationale || ""),
      locked ? null : h("div", { class: "row-actions" },
        h("button", { onclick: () => act("POST", `/api/suggestions/${id}/edit`, { rule: rule.value }, "збережено") }, "Зберегти правку"),
        h("button", { disabled: sending || !s.validator_report.valid, title: "перевірки + регресійний гейт → пропозиція TOD-37",
                      onclick: () => act("POST", `/api/suggestions/${id}/send`, undefined, "гейт запущено") },
          sending ? "гейт виконується…" : "На схвалення"),
        reason,
        h("button", { onclick: () => act("POST", `/api/suggestions/${id}/reject`, { reason: reason.value }, "відхилено") }, "Відхилити")),
      gate ? h("div", { class: "small" }, h("h4", {}, `Гейт: ${gate.passed ? "пройдено" : "заблоковано"}`),
        gate.proposal ? h("div", { class: "mono" }, `пропозиція ${gate.proposal}` + (gate.passed ? ` → python -m tools.rules approve ${gate.proposal} --by … --role … --comment …` : "")) : null,
        [...(gate.errors || []), ...(gate.reasons || [])].map((x) => h("div", { class: "warn-text" }, x))) : null,
      h("h4", {}, `Випадки (${s.cases.length})`),
      h("ol", { class: "changelog" }, s.cases.map((c) => h("li", {},
        h("span", { class: "mono muted" }, `${hms(c.at)} #${c.id} `), TRIGGER[c.kind] || c.kind, ` · ${c.location_id || "—"}`,
        c.risk_level_llm ? ` · GPT ${c.risk_level_llm} → ${c.risk_level_final || "—"}` : "",
        c.accepted ? " · підтверджено оператором" : "", c.reason ? ` · ${c.reason}` : "",
        c.context && c.context.summary ? h("div", { class: "small muted" }, c.context.summary) : null))));
    clearInterval(poll);
    if (sending) poll = setInterval(() => { if (!document.body.contains(detail)) clearInterval(poll); else load(); }, 3000);
  }
  load();
}
