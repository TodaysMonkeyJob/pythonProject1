// TOD-52: the mode switch — Автономний (стаття 1) / Мультиагентний (стаття 2). The header chip shows the global mode
// (admin: opens the switch dialog, a confirmation before every change), the banner shows locations whose multi-agent
// decision fell back to the fuzzy system, with the reason. Per-location overrides and the audit log in the dialog.
import { orderedLocations, store, switchMode } from "../store.js";
import { $, clear, dialog, h, hms, toast } from "../ui.js";

const SHORT = { autonomous: "автономний", multiagent: "мультиагентний" };

export function drawMode() {
  const m = store.mode, chip = $("#mode"), banner = $("#mode-banner");
  if (!m) { chip.textContent = "режим…"; banner.hidden = true; return; }
  chip.textContent = `режим: ${m.labels[m.mode]}`;
  chip.classList.toggle("multi", m.mode === "multiagent");
  chip.disabled = false;
  chip.title = m.role === "admin" ? "змінити режим" : "режим (змінює лише адміністратор)";
  banner.hidden = !m.fallback.length;
  clear(banner, m.fallback.length
    ? ["Працює автономний режим: ", m.fallback.map((f) => `${f.name} — ${f.reason || "причина невідома"}`).join("; ")]
    : []);
}

async function confirmSwitch(mode, locationId, name) {
  const labels = store.mode.labels;
  const where = locationId ? `для локації «${name}»` : "для всієї системи";
  if (!confirm(`Перемкнути ${where} на «${labels[mode]}»?\nЗміна діє з наступного циклу рішень і зберігається після перезапуску.`)) return;
  try { await switchMode(mode, locationId); toast("Режим", `${labels[mode]} — ${where}`, "safe"); $("#dlg").close(); }
  catch (e) { toast("Режим", e.message); }
}

export function openModeDialog() {
  const m = store.mode;
  if (!m) return;
  const admin = m.role === "admin";
  const choice = (mode, locationId = null, name = "") => h("button", {
    class: mode === (locationId ? (m.locations[locationId] || m.mode) : m.mode) ? "on" : "",
    disabled: !admin, onclick: () => confirmSwitch(mode, locationId, name) }, locationId ? SHORT[mode] : m.labels[mode]);
  dialog("Режим роботи",
    h("p", { class: "muted small" }, admin ? "Автономний: рішення приймає нечітка система (стаття 1), без агентів і мережі. "
      + "Мультиагентний: агенти додають заяви, рішення — агент валідації в межах нечіткого коридору (стаття 2)."
      : "Змінювати режим може лише адміністратор."),
    h("div", { class: "filters" }, choice("autonomous"), choice("multiagent")),
    h("h3", {}, "Окремі локації"),
    h("table", { class: "rules-table" }, h("tbody", {}, orderedLocations().map((l) => {
      const e = m.effective[l.location_id];
      return h("tr", {}, h("td", {}, `${l.icon || ""} ${l.name}`.trim()),
        h("td", {}, choice("autonomous", l.location_id, l.name), " ", choice("multiagent", l.location_id, l.name)),
        h("td", { class: "small muted" }, e ? `зараз: ${SHORT[e.effective] || "—"}${e.reason ? ` (${e.reason})` : ""}` : ""));
    }))),
    h("h3", {}, "Журнал перемикань"),
    m.history.length ? h("ul", { class: "changelog small" }, m.history.slice().reverse().map((x) => h("li", {},
      `${hms(x.at)} ${x.by}: ${x.scope === "location" ? `${x.location_id}: ` : ""}${SHORT[x.from] || x.from} → ${SHORT[x.to] || x.to}`)))
      : h("p", { class: "muted small" }, "перемикань ще не було"));
}
