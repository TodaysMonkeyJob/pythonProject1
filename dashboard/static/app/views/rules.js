// F26 (TOD-37) page «Правила»: the rule base version in force, every rule with its metadata and firing statistics
// for that version (fired / bound kept / GPT accepted by the operator), and the append-only changelog. Read-only:
// changes go through `python -m tools.rules propose` → gate → `approve` (an approval record is required).
import { api } from "../store.js";
import { clear, h, hms } from "../ui.js";

const SECTION = { risk: "ризик", "trust.source": "довіра: мовець", "trust.asr": "довіра: ASR",
                  "trust.identity": "довіра: обличчя", "envelope.confirmation": "підтвердження безпеки",
                  gating: "виклик GPT" };
const ACTION = { baseline: "базова версія", approve: "схвалено", rollback: "відкат" };
let filter = "";

export async function renderRules(view) {
  clear(view, h("section", { class: "admin rules-page" }, h("p", { class: "muted" }, "завантаження…")));
  let data;
  try { data = await api("GET", "/api/rules"); }
  catch (e) { clear(view, h("section", { class: "admin" }, h("p", { class: "muted" }, `Правила недоступні: ${e.message}`))); return; }
  const draw = () => {
    const q = filter.toLowerCase();
    const rows = data.rules.filter((r) => !q || `${r.id} ${r.rule} ${r.rationale} ${r.source}`.toLowerCase().includes(q));
    clear(view, h("section", { class: "admin rules-page" },
      h("div", { class: "admin-head" },
        h("h2", {}, "Правила"),
        h("span", { class: "mono" }, `версія ${data.short}`),
        h("span", { class: data.registered ? "muted" : "warn-text" },
          data.registered ? `${data.rules.length} правил · зареєстрована` : "НЕ зареєстрована — зміна без схвалення?"),
        h("input", { type: "search", placeholder: "пошук", value: filter, "aria-label": "пошук правил",
                     oninput: (e) => { filter = e.target.value; draw(); } })),
      h("p", { class: "muted small" }, "Зміни: python -m tools.rules propose <файл> → перевірка й регресійний гейт → approve --by --role --comment."),
      h("table", { class: "rules-table" },
        h("thead", {}, h("tr", {}, ["id", "розділ", "правило", "вага", "джерело", "обґрунтування", "додано",
                                    "спрацювало", "межа втримана", "GPT прийнято"].map((t) => h("th", {}, t)))),
        h("tbody", {}, rows.map((r) => h("tr", {},
          h("td", { class: "mono" }, r.id), h("td", {}, SECTION[r.section] || r.section),
          h("td", { class: "mono small" }, r.rule), h("td", { class: "mono" }, r.weight),
          h("td", { class: "small" }, r.source), h("td", { class: "small" }, r.rationale),
          h("td", { class: "mono small" }, r.added), h("td", { class: "mono" }, r.fired),
          h("td", { class: "mono" }, r.confirmed), h("td", { class: "mono" }, r.rejected))))),
      h("h3", {}, "Журнал змін"),
      h("ol", { class: "changelog" }, data.changelog.map((e) => h("li", {},
        h("span", { class: "mono muted" }, `${hms(e.at)} `), h("b", {}, ACTION[e.action] || e.action), " ",
        h("span", { class: "mono" }, `${(e.from || "—").slice(0, 12)} → ${(e.to || "").slice(0, 12)}`),
        ` · ${e.by} (${e.role}) — ${e.comment || ""}`)))));
  };
  draw();
}
