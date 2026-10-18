// Overview (carousel page 0): every camera + every location card + the latest notifications, without scrolling.
import { ackLocation, ackNotification, cameras, orderedLocations, personName, store, unacked } from "../store.js";
import { STATE_LABEL, STATE_RANK, h, hms } from "../ui.js";
import { cameraGrid } from "./grid.js";

let page = 0;

export function locationCard(location, go) {
  const s = store.summaries[location.location_id], state = s ? s.state : "safe";
  const people = s ? s.people.map((p) => personName(p.person_id, p.name) + (p.action ? ` ${p.action}` : "")).slice(0, 3).join(", ") : "";
  return h("div", { class: `loc-card st-${state}`, onclick: () => go(`#/loc/${encodeURIComponent(location.location_id)}`) },
    h("div", { class: "row" }, h("span", { class: "title" }, `${location.icon || ""} ${location.name}`.trim()),
      h("span", { class: `badge st-${state}` }, `● ${STATE_LABEL[state] || state}`)),
    s && s.reason ? h("div", { class: "reason" }, s.reason) : null,
    h("div", { class: "muted" }, `людей: ${s ? s.people_count : 0}${people ? " — " + people : ""}`,
      s ? ` · пристрої ${s.devices.online}/${s.devices.total}` : ""),
    s && s.active_alerts ? h("button", { class: "ack", onclick: (e) => { e.stopPropagation(); ackLocation(location.location_id); } },
                             `Підтвердити (${s.active_alerts})`) : null);
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

export function renderOverview(root, { go }) {
  const order = Object.fromEntries(orderedLocations().map((l, i) => [l.location_id, i]));
  const cams = cameras().slice().sort((a, b) => (order[a.location_id] ?? 99) - (order[b.location_id] ?? 99)
                                               || a.name.localeCompare(b.name));
  const grid = cameraGrid(cams, { page, onPage: (n) => { page = n; renderOverview(root, { go }); },
                                   onTile: (d) => go(`#/loc/${encodeURIComponent(d.location_id)}?focus=${encodeURIComponent(d.device_id)}`) });
  root.replaceChildren(h("div", { class: "overview" },
    h("section", { class: "panel cams" }, grid),
    h("aside", { class: "panel side" },
      h("h2", {}, "Локації"),
      h("div", { class: "loc-list" }, sortedLocations().map((l) => locationCard(l, go))),
      h("h2", {}, `Сповіщення${unacked().length ? ` (${unacked().length})` : ""}`),
      h("div", { class: "note-feed" }, notificationFeed(5)))));
}

