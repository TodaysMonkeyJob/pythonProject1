// Camera grid that never scrolls: 1 → 1×1, 2 → 2×1, 3–4 → 2×2, 5–6 → 3×2, 7–9 → 3×3, more → pages of 9.
// F21 tile: territory chip with its state dot, people count, device name; the border shows the territory state
// (none when safe); an offline camera says how long it has been silent. Tiles are focusable (Enter opens).
import { locationOf, stateOf, store } from "../store.js";
import { STATE_LABEL, h } from "../ui.js";

export function layout(count) {
  if (count <= 1) return [1, 1];
  if (count === 2) return [2, 1];
  if (count <= 4) return [2, 2];
  if (count <= 6) return [3, 2];
  return [3, 3];
}

export const PAGE_SIZE = 9;

// options: {page, onTile(device), onPage(n), size: "small" | "full"}
export function cameraGrid(devices, { page = 0, onTile = null, onPage = null, size = "small" } = {}) {
  if (!devices.length) return h("div", { class: "grid-empty muted" }, "камер немає — підключіть на сторінці «Пристрої»");
  const pages = Math.ceil(devices.length / PAGE_SIZE), current = Math.min(page, pages - 1);
  const shown = devices.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);
  const [cols, rows] = layout(shown.length);
  const grid = h("div", { class: "cam-grid", style: { gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
                                                      gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))` } },
                 shown.map((d) => tile(d, size, onTile)));
  if (pages <= 1) return grid;
  return h("div", { class: "cam-paged" }, grid,
           h("div", { class: "pager" }, Array.from({ length: pages }, (_, i) =>
             h("button", { class: i === current ? "on" : "", "aria-label": `сторінка камер ${i + 1}`,
                           onclick: () => onPage && onPage(i) }, String(i + 1)))));
}

function silentFor(device) {
  if (!device.last_seen) return "Немає сигналу";
  const minutes = Math.max(1, Math.round((Date.now() - new Date(device.last_seen).getTime()) / 60000));
  if (!Number.isFinite(minutes) || minutes > 525600) return "Немає сигналу";   // unknown / bogus last_seen
  if (minutes < 60) return `Немає сигналу ${minutes} хв`;
  if (minutes < 48 * 60) return `Немає сигналу ${Math.round(minutes / 60)} год`;
  return `Немає сигналу ${Math.round(minutes / 1440)} дн`;
}

export function tile(device, size = "small", onTile = null) {
  const location = locationOf(device.location_id), state = stateOf(device.location_id);
  const summary = store.summaries[device.location_id];
  const online = device.status === "online";
  const people = summary ? summary.people.filter((p) => !p.device_id || p.device_id === device.device_id).length : 0;
  const territory = location ? `${location.icon || ""} ${location.name}`.trim() : device.location_id;
  const shownState = online ? state : "offline";
  const open = () => onTile && onTile(device);
  return h("div", { class: `tile st-${shownState}`, role: onTile ? "button" : null, tabindex: onTile ? "0" : null,
                    "aria-label": `${device.name}, ${territory}: ${STATE_LABEL[shownState] || shownState}${people ? `, людей ${people}` : ""}`,
                    onclick: open, onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { open(); e.preventDefault(); } } },
    online ? h("img", { alt: "", dataset: { video: device.device_id, size } })
           : h("div", { class: "placeholder" }, silentFor(device)),
    h("div", { class: "chip" }, h("span", { class: `dot st-${state}`, "aria-hidden": "true" }), territory,
      people ? h("span", { class: "people" }, ` · 👤 ${people}`) : null),
    h("div", { class: "overlay" }, h("span", { class: "name" }, device.name)));
}
