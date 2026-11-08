// C1 (TOD-54): persistent sensor calibration on the Devices page — status per device (source stored / factory / new,
// date, age, last check), «Перекалібрувати» per device and per location (confirmation with instructions, progress,
// cancel), history with «Відновити». Admin only for the actions (the API answers 403 otherwise).
import { api, store } from "../store.js";
import { dialog, h, toast } from "../ui.js";

const SOURCE = { stored: "збережена", factory: "заводська", new: "нова" };
const DOT = { ready: "on", factory: "on", warming_up: "st-attention", checking: "st-attention", needs_check: "st-danger",
              calibrating: "st-attention", uncalibrated: "st-danger" };
const guard = (fn) => async (...args) => { try { await fn(...args); } catch (e) { alert(e.message); } };

function age(days) {
  if (days === null || days === undefined) return "";
  return days < 1 ? "сьогодні" : `${Math.round(days)} дн. тому`;
}

async function recalibrate(target, label, instructions) {
  const text = `Перекалібрувати ${label}?\n\n${instructions || "Забезпечте звичайні умови без руху, звуків і газу."}` +
    "\n\nДо завершення використовується збережена калібровка з ширшою невизначеністю.";
  if (!confirm(text)) return;
  await api("POST", "/api/calibration/recalibrate", { ...target, confirm: true });
  toast("Калібрування", `${label}: калібрування розпочато`, "attention");
}

function history(device, record) {
  const rows = (record.history || []).map((v, i) => h("tr", {},
    h("td", {}, (v.date || "").slice(0, 16).replace("T", " ")), h("td", {}, v.method || ""), h("td", {}, v.by || ""),
    h("td", {}, h("button", { onclick: guard(async () => {
      if (!confirm(`Відновити калібровку від ${(v.date || "").slice(0, 10)} для ${device.name}?`)) return;
      await api("POST", "/api/calibration/restore", { uid: record.uid, index: i, confirm: true });
      toast("Калібрування", `${device.name}: відновлено калібровку від ${(v.date || "").slice(0, 10)}`, "safe");
    }) }, "Відновити"))));
  dialog(`Історія калібровки: ${device.name}`,
    h("p", {}, `Поточна: ${(record.date || "—").slice(0, 16).replace("T", " ")} — ${record.method || ""}`),
    rows.length ? h("table", { class: "table" }, h("tr", {}, h("th", {}, "дата"), h("th", {}, "метод"), h("th", {}, "хто"),
      h("th", {}, "")), ...rows) : h("p", { class: "muted" }, "попередніх версій немає"));
}

function row(device, record) {
  const check = device.last_check;
  const busy = device.status === "calibrating";
  return h("div", { class: "devrow" }, h("span", { class: `dot ${DOT[device.status] || ""}` }),
    h("span", { title: `${device.device_id} · ${device.uid}` }, device.name),
    h("span", { class: device.status === "needs_check" ? "gas-crit" : "" }, device.text || device.status),
    h("span", { class: "muted" }, [SOURCE[device.source] || device.source, device.date ? device.date.slice(0, 10) : null,
      age(device.age_days), device.ready_after_sec !== undefined ? `готово за ${device.ready_after_sec} с` : null]
      .filter(Boolean).join(" · ")),
    h("span", { class: "muted small", title: check ? check.detail : "" },
      check ? `перевірка ${check.ok ? "✓" : "✗"} ${(check.at || "").slice(11, 16)}` : ""),
    device.status === "calibrating" && device.progress !== undefined
      ? h("progress", { max: 1, value: device.progress, title: `${Math.round(device.progress * 100)} %` }) : null,
    device.source === "factory" ? null : busy
      ? h("button", { onclick: guard(async () => { await api("POST", "/api/calibration/cancel", { device_id: device.device_id });
        }) }, "Скасувати")
      : h("button", { onclick: guard(() => recalibrate({ device_id: device.device_id }, device.name, device.instructions)) },
          "Перекалібрувати"),
    record && (record.history || []).length ? h("button", { onclick: () => history(device, record) }, "Історія") : null);
}

export function calibrationSection() {
  const box = h("div", { class: "devlist" }, h("div", { class: "muted" }, "завантаження…"));
  const draw = async () => {
    if (!box.isConnected && box.dataset.drawn) return;
    let j;
    try { j = await api("GET", "/api/calibration"); } catch (e) {
      box.replaceChildren(h("div", { class: "muted" }, `немає даних: ${e.message}`)); return;
    }
    box.dataset.drawn = "1";
    const records = Object.fromEntries((j.records || []).map((r) => [r.uid, r]));
    const locations = [...new Set(j.devices.map((d) => d.location_id).filter(Boolean))];
    const name = (id) => (store.locations.find((l) => l.location_id === id) || {}).name || id;
    box.replaceChildren(
      ...(j.devices.length ? j.devices.map((d) => row(d, records[d.uid])) : [h("div", { class: "muted" },
        "пристрої ще не повідомили про калібровку (sensor_calibration у config.yaml)")]),
      ...(locations.length ? [h("div", { class: "devrow" }, h("span", { class: "muted" }, "Уся локація:"),
        ...locations.map((loc) => h("button", { onclick: guard(() => recalibrate({ location_id: loc }, `усі пристрої «${name(loc)}»`,
          "Забезпечте чисте повітря, тишу й порожній кадр у цій локації.")) }, `Перекалібрувати «${name(loc)}»`)))] : []));
    if (box.isConnected || !box.dataset.drawn) setTimeout(draw, 3000);      // live: statuses / progress
  };
  draw();
  return box;
}
