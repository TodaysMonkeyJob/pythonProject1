// Camera grid that never scrolls: 1 → 1×1, 2 → 2×1, 3–4 → 2×2, 5–6 → 3×2, 7–9 → 3×3, more → pages of 9.
import { store, locationOf, stateOf } from "../store.js";
import { h, hms } from "../ui.js";

export function layout(count) {
  if (count <= 1) return [1, 1];
  if (count === 2) return [2, 1];
  if (count <= 4) return [2, 2];
  if (count <= 6) return [3, 2];
  return [3, 3];
}

export const PAGE_SIZE = 9;

// options: {page, focus, onTile(device), onPage(n)}
export function cameraGrid(devices, { page = 0, focus = null, onTile = null, onPage = null } = {}) {
  if (!devices.length) return h("div", { class: "grid-empty muted" }, "камер немає — підключіть на сторінці «Пристрої»");
  if (focus && devices.some((d) => d.device_id === focus)) {
    const device = devices.find((d) => d.device_id === focus);
    return h("div", { class: "cam-grid", style: { gridTemplateColumns: "1fr", gridTemplateRows: "1fr" } },
             tile(device, "full", onTile));
  }
  const pages = Math.ceil(devices.length / PAGE_SIZE), current = Math.min(page, pages - 1);
  const shown = devices.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);
  const [cols, rows] = layout(shown.length);
  const grid = h("div", { class: "cam-grid", style: { gridTemplateColumns: `repeat(${cols}, 1fr)`,
                                                      gridTemplateRows: `repeat(${rows}, 1fr)` } },
                 shown.map((d) => tile(d, "small", onTile)));
  if (pages <= 1) return grid;
  return h("div", { class: "cam-paged" }, grid,
           h("div", { class: "pager" }, Array.from({ length: pages }, (_, i) =>
             h("button", { class: i === current ? "on" : "", onclick: () => onPage && onPage(i) }, String(i + 1)))));
}

function tile(device, size, onTile) {
  const location = locationOf(device.location_id), state = stateOf(device.location_id);
  const summary = store.summaries[device.location_id];
  const online = device.status === "online";
  const people = summary ? summary.people.filter((p) => !p.device_id || p.device_id === device.device_id).length : 0;
  return h("div", { class: `tile st-${online ? state : "offline"}`, title: "клік — збільшити / зменшити",
                    onclick: () => onTile && onTile(device) },
    online ? h("img", { alt: device.name, dataset: { video: device.device_id, size } })
           : h("div", { class: "placeholder" }, "немає сигналу",
               h("small", {}, device.last_seen ? `востаннє ${hms(device.last_seen)}` : "")),
    h("div", { class: "overlay" },
      h("span", { class: `dot st-${online ? state : "offline"}` }),
      h("b", {}, device.name), h("span", { class: "muted" }, ` · ${location ? location.name : device.location_id}`),
      people ? h("span", { class: "people" }, `👤 ${people}`) : null));
}
