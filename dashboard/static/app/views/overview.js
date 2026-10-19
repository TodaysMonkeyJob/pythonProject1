// Overview (F21): left ~65 % every camera (click → its territory), right ~35 % the territory cards sorted by severity
// (danger → attention → offline → safe) and a short global log, newest on top, scrolling inside its own box.
// The skeleton is mounted once; live updates replace the grid / cards and prepend to the log (never re-created, so
// its scroll position is kept).
import { ackNotification, cameras, orderedLocations, personName, store } from "../store.js";
import { LiveList, STATE_LABEL, STATE_RANK, h, hms } from "../ui.js";
import { forGlobalLog, eventRow } from "./events.js";
import { cameraGrid } from "./grid.js";

let mounted = null;           // {root, grid, cards, log, page}

export function locationCard(location) {
  const s = store.summaries[location.location_id], state = s ? s.state : "safe";
  const people = s ? s.people.map((p) => personName(p.person_id, p.name)).slice(0, 3).join(", ") : "";
  return h("a", { class: `loc-card st-${state}`, href: `#/loc/${encodeURIComponent(location.location_id)}`,
                  "aria-label": `${location.name}: ${STATE_LABEL[state] || state}` },
    h("div", { class: "row" }, h("span", { class: "title" }, `${location.icon || ""} ${location.name}`.trim()),
      h("span", { class: `badge st-${state}` }, h("span", { class: `dot st-${state}`, "aria-hidden": "true" }),
        ` ${(STATE_LABEL[state] || state).toUpperCase()}`)),
    h("div", { class: "reason" }, s && s.reason ? s.reason : state === "safe" ? "без подій" : "—"),
    h("div", { class: "muted small" }, `людей: ${s ? s.people_count : 0}${people ? ` — ${people}` : ""}`,
      s ? ` · пристрої ${s.devices.online}/${s.devices.total}` : ""));
}

export function notificationFeed(limit = 5) {
  const list = store.notifications.slice(0, limit);
  if (!list.length) return h("div", { class: "muted" }, "сповіщень немає");
  return list.map((n) => h("div", { class: `note ${n.level}${n.acknowledged_at ? " acked" : ""}` },
    h("div", { class: "row" }, h("b", {}, n.title), h("span", { class: "muted" }, hms(n.ts))),
    h("div", {}, n.body),
    n.acknowledged_at ? h("div", { class: "muted" }, `підтвердив ${n.acknowledged_by}`)
                      : h("button", { class: "ack", onclick: () => ackNotification(n.id) }, "Підтвердити")));
}

export function sortedLocations() {
  return orderedLocations().slice().sort((a, b) =>
    (STATE_RANK[(store.summaries[a.location_id] || {}).state || "safe"] - STATE_RANK[(store.summaries[b.location_id] || {}).state || "safe"])
    || a.order - b.order);
}

function mount(root) {
  const log = new LiveList({ cls: "global-log", empty: "подій ще немає", label: "загальний журнал" });
  const grid = h("section", { class: "panel cams", "aria-label": "камери" });
  const cards = h("div", { class: "loc-list" });
  root.replaceChildren(h("div", { class: "overview", dataset: { view: "overview" } }, grid,
    h("aside", { class: "panel side" }, h("h2", {}, "Території"), cards, h("h2", {}, "Журнал"), log.el)));
  log.set(store.events.filter(forGlobalLog).slice(-150).reverse().map((ev) => eventRow(ev, { withLocation: true })));
  mounted = { root, grid, cards, log, page: mounted ? mounted.page : 0 };
}

export function renderOverview(root, { go }) {
  if (!mounted || mounted.root !== root || !root.querySelector('[data-view="overview"]')) mount(root);
  const order = Object.fromEntries(orderedLocations().map((l, i) => [l.location_id, i]));
  const cams = cameras().slice().sort((a, b) => (order[a.location_id] ?? 99) - (order[b.location_id] ?? 99)
                                               || a.name.localeCompare(b.name));
  mounted.grid.replaceChildren(cameraGrid(cams, {
    page: mounted.page, onPage: (n) => { mounted.page = n; renderOverview(root, { go }); },
    onTile: (d) => go(`#/loc/${encodeURIComponent(d.location_id)}`) }));
  mounted.cards.replaceChildren(...sortedLocations().map(locationCard));
}

export function onOverviewEvent(ev) {
  if (mounted && forGlobalLog(ev)) mounted.log.prepend([eventRow(ev, { withLocation: true })]);
}

export function leaveOverview() { mounted = null; }
