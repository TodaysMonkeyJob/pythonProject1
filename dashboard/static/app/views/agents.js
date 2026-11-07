// B2 (TOD-49) page «Агенти»: every discovered agent (role, version, backend, status, latency, last claims) with an
// on / off switch (→ POST /api/agents/{name} → agent_control on the bus), invalid manifests, the latest decisions.
// Refreshed every 2 s while the page is open.
import { api } from "../store.js";
import { clear, h, hms, toast } from "../ui.js";

const ROLE = { fuzzy: "нечітка логіка", video: "відео", audio: "аудіо", gas: "газ", validation: "валідація",
               explanation: "пояснення", rule_suggestion: "пропозиції правил", other: "інше" };
const BACKEND = { rules: "правила", local_model: "локальна модель", cloud_model: "хмарна модель" };
const BRANCH = { fuzzy_only: "лише нечітка система", agree: "узгоджено", deescalation_rejected: "зниження відхилено",
                 deescalation_confirmed: "зниження підтверджено", escalation_review: "до оператора" };
let timer = null;

const fmtMs = (x) => (x == null ? "—" : `${x.toFixed(1).replace(".", ",")} мс`);
const claimText = (c) => `${c.claim} ${JSON.stringify(c.value).slice(0, 80)}`;

export async function renderAgents(view) {
  clearInterval(timer);
  const draw = async () => {
    if (!location.hash.startsWith("#/agents")) { clearInterval(timer); return; }
    let data;
    try { data = await api("GET", "/api/agents"); }
    catch (e) { clear(view, h("section", { class: "admin" }, h("h2", {}, "Агенти"), h("p", { class: "muted" }, `Агенти недоступні: ${e.message}`))); return; }
    const on = data.agents.filter((a) => a.enabled).length;
    clear(view, h("section", { class: "admin agents-page" },
      h("div", { class: "admin-head" },
        h("h2", {}, "Агенти"),
        h("span", { class: "muted" }, `${data.agents.length} знайдено · ${on} увімкнено`)),
      h("table", { class: "rules-table agents-table" },
        h("thead", {}, h("tr", {}, ["", "агент", "роль", "версія", "бекенд", "стан", "затримка", "тайм-аут",
                                    "заяв", "помилки / пізно", "точність", "останні заяви"].map((t) => h("th", {}, t)))),
        h("tbody", {}, data.agents.map((a) => h("tr", { class: a.enabled ? "" : "muted" },
          h("td", {}, h("input", { type: "checkbox", checked: a.enabled, disabled: a.required || (!a.enabled && a.blocked),
            title: a.required ? "обов'язковий агент" : a.blocked || "",
            "aria-label": `${a.enabled ? "вимкнути" : "увімкнути"} ${a.name}`,
            onchange: async (e) => {
              try { await api("POST", `/api/agents/${encodeURIComponent(a.name)}`, { enabled: e.target.checked });
                    toast("Агенти", `${a.name}: ${e.target.checked ? "вмикається" : "вимикається"}`, "safe"); }
              catch (err) { toast("Агенти", err.message); e.target.checked = !e.target.checked; }
            } })),
          h("td", {}, h("b", {}, a.name), h("div", { class: "small muted" }, a.description)),
          h("td", {}, ROLE[a.role] || a.role), h("td", { class: "mono" }, a.version),
          h("td", {}, BACKEND[a.backend] || a.backend),
          h("td", {}, a.problem ? h("span", { class: "warn-text" }, a.problem) : a.health?.status || "—"),
          h("td", { class: "mono" }, fmtMs(a.latency_ms)), h("td", { class: "mono" }, `${a.timeout_s} с`),
          h("td", { class: "mono" }, a.claims), h("td", { class: "mono" }, `${a.errors} / ${a.late}`),
          h("td", { class: "mono" }, a.evaluation?.status === "measured" ? a.evaluation.accuracy.toFixed(2) : "не виміряно"),
          h("td", { class: "small mono" }, (a.last_claims || []).slice(0, 2).map((c) => h("div", {}, `${hms(c.ts)} ${claimText(c)}`))))))),
      Object.keys(data.errors || {}).length ? h("div", {}, h("h3", {}, "Не завантажені"),
        h("ul", {}, Object.entries(data.errors).map(([n, e]) => h("li", { class: "small" }, h("b", {}, n), ` ${e}`)))) : null,
      h("h3", {}, "Останні рішення"),
      h("ul", { class: "changelog" }, (data.decisions || []).slice().reverse().map((d) => h("li", { class: "small" },
        `${d.location_id}: ${d.level} ${(d.risk ?? 0).toFixed(2)} · ${BRANCH[d.branch] || d.branch} · ${d.decided_by}`,
        d.missing?.length ? ` · без відповіді: ${d.missing.join(", ")}` : "")))));
  };
  await draw();
  timer = setInterval(draw, 2000);
}
