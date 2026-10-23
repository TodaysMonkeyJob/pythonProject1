// Control room (F13, single-screen redesign F21): one screen, the page never scrolls on desktop — only panels scroll
// inside themselves. Views: #/overview · #/loc/<id> · #/loc/<id>/<tab> (events | talk | gas | gpt) · #/people ·
// #/devices · #/locations · #/trust (F23). Territories switch with the ◀ ▶ buttons around the tabs, ← → (wrap around), Home
// (Overview) and 1–9 (territory by index); back / forward follow the hash.
import { cameras, on, orderedLocations, start, stateOf, store, unacked } from "./store.js";
import { STATE_LABEL, $, clear, h, plural, toast } from "./ui.js";
import { connectVideo, syncSubscriptions } from "./video.js";
import { renderDevices, renderLocations, renderPeople } from "./views/admin.js";
import { renderTrust } from "./views/trust.js";
import { renderRules } from "./views/rules.js";
import { PANEL_TABS, leaveLocation, onLocationEvent, redrawCharts, renderLocation } from "./views/location.js";
import { leaveOverview, notificationFeed, onOverviewEvent, renderOverview } from "./views/overview.js";

const view = $("#view");
let route = { page: "overview" };

function parse(hash) {
  const parts = (hash || "#/overview").replace(/^#\/?/, "").split("?")[0].split("/").map(decodeURIComponent);
  if (parts[0] === "loc" && parts[1]) return { page: "loc", id: parts[1], tab: PANEL_TABS.includes(parts[2]) ? parts[2] : null };
  if (["people", "devices", "locations", "trust", "rules"].includes(parts[0])) return { page: parts[0] };
  return { page: "overview" };
}

// The territory cycle: Overview, then every location in its order
const cycle = () => ["#/overview", ...orderedLocations().map((l) => `#/loc/${encodeURIComponent(l.location_id)}`)];
function cycleIndex() {
  if (route.page !== "loc") return route.page === "overview" ? 0 : -1;
  return Math.max(0, cycle().indexOf(`#/loc/${encodeURIComponent(route.id)}`));
}
function step(delta) {
  const pages = cycle(), i = cycleIndex();
  go(pages[(Math.max(0, i) + delta + pages.length) % pages.length]);
}

// go(null) re-renders; go(hash) navigates (a history entry: back / forward work)
function go(hash) {
  if (hash === null) { render(); return; }
  if (hash === location.hash) render(); else location.hash = hash;
}

let scheduled = false;
function render() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    const active = document.activeElement;
    if (active && view.contains(active) && /^(INPUT|SELECT|TEXTAREA)$/.test(active.tagName)) {
      setTimeout(render, 1000);          // the user is typing / choosing: don't re-render under the cursor
      return;
    }
    draw();
  });
}

function draw() {
  if (route.page !== "loc") leaveLocation();
  if (route.page !== "overview") leaveOverview();
  if (route.page === "overview") renderOverview(view, { go });
  else if (route.page === "loc") renderLocation(view, { go, id: route.id, tab: route.tab });
  else if (route.page === "people") renderPeople(view);
  else if (route.page === "devices") renderDevices(view);
  else if (route.page === "locations") renderLocations(view);
  else if (route.page === "trust") renderTrust(view);
  else if (route.page === "rules") renderRules(view);
  drawTopbar();
  syncSubscriptions();
}

function drawTopbar() {
  const pages = cycle(), current = cycleIndex();
  const focused = $("#tabs").contains(document.activeElement);
  clear($("#tabs"), pages.map((hash, i) => {
    const id = i ? decodeURIComponent(hash.slice(6)) : null;
    const location = id ? orderedLocations().find((l) => l.location_id === id) : null;
    const state = id ? stateOf(id) : null;
    return h("a", { href: hash, role: "tab", class: `tab${i === current ? " on" : ""}`,
                    tabindex: i === Math.max(0, current) ? "0" : "-1", "aria-selected": i === current ? "true" : "false",
                    title: id ? `${i}. ${location ? location.name : id} — ${STATE_LABEL[state] || state}` : "Огляд (Home)" },
      state ? h("span", { class: `dot st-${state}`, "aria-hidden": "true" }) : null,
      id ? (location ? `${location.icon || ""} ${location.name}`.trim() : id) : "Огляд",
      state ? h("span", { class: "sr-only" }, ` — ${STATE_LABEL[state] || state}`) : null);
  }));
  if (focused) $("#tabs").querySelector('[tabindex="0"]')?.focus();
  const counts = { danger: 0, attention: 0, offline: 0 };
  orderedLocations().forEach((l) => { const s = stateOf(l.location_id); if (s in counts) counts[s] += 1; });
  const parts = [["danger", "небезпека"], ["attention", "увага"], ["offline", "офлайн"]].filter(([k]) => counts[k])
    .map(([k, label]) => h("span", { class: `chip-part st-${k}` }, h("span", { class: `dot st-${k}` }), ` ${counts[k]} ${label}`));
  const cams = cameras(), online = cams.filter((c) => c.status === "online").length;
  clear($("#site"), parts.length ? parts.flatMap((p, i) => (i ? [" · ", p] : [p]))
                                 : [h("span", { class: "dot st-safe" }), " усе безпечно"],
        h("span", { class: "muted" }, ` · ${online}/${cams.length} камер`));
  const dups = store.duplicates.length;
  $("#dups").hidden = !dups;
  $("#dups").textContent = `${dups} ${plural(dups, "можливий дублікат", "можливі дублікати", "можливих дублікатів")}`;
  const open = unacked().length;
  $("#bell-count").hidden = !open; $("#bell-count").textContent = open;
  $("#bell").setAttribute("aria-label", `сповіщення${open ? `: ${open} непідтверджених` : ""}`);
  $("#ws-dot").classList.toggle("on", store.connected);
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("on", a.getAttribute("href") === `#/${route.page}`));
  if ($("#notes").open) clear($("#note-list"), notificationFeed(30));
}

// --- input ---------------------------------------------------------------------------------------------------------
window.addEventListener("hashchange", () => { route = parse(location.hash); draw(); });
$("#prev").onclick = () => step(-1);
$("#next").onclick = () => step(1);
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea, [contenteditable]") || e.metaKey || e.ctrlKey || e.altKey) return;
  if ($("#dlg").open) return;
  if (e.key === "ArrowRight") step(1);
  else if (e.key === "ArrowLeft") step(-1);
  else if (e.key === "Home") go("#/overview");
  else if (/^[1-9]$/.test(e.key) && Number(e.key) < cycle().length) go(cycle()[Number(e.key)]);
  else return;
  e.preventDefault();
  if ($("#tabs").contains(document.activeElement)) requestAnimationFrame(() => $("#tabs").querySelector('[tabindex="0"]')?.focus());
});
let swipe = null;                                   // phones: swipe between territories
view.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse") swipe = { x: e.clientX, y: e.clientY }; });
view.addEventListener("pointerup", (e) => {
  if (!swipe || cycleIndex() < 0) { swipe = null; return; }
  const dx = e.clientX - swipe.x, dy = e.clientY - swipe.y; swipe = null;
  if (Math.abs(dx) > 60 && Math.abs(dx) > 1.5 * Math.abs(dy)) step(dx < 0 ? 1 : -1);
});

$("#bell").onclick = () => { const d = $("#notes"); if (d.open) d.close(); else { clear($("#note-list"), notificationFeed(30)); d.show(); } };
$("#theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("iotguard_theme", next); } catch (e) { /* private mode */ }
  redrawCharts(view);
};
try { const saved = localStorage.getItem("iotguard_theme"); if (saved) document.documentElement.dataset.theme = saved; } catch (e) { /* ignore */ }

// Browser notifications for danger: permission asked once, on the first click (browsers require a gesture)
document.addEventListener("click", () => {
  try { if (window.Notification && Notification.permission === "default" && !localStorage.getItem("iotguard_asked")) {
    localStorage.setItem("iotguard_asked", "1"); Notification.requestPermission(); } } catch (e) { /* ignore */ }
}, { once: true });

on("notify", (n) => {
  toast(n.title, n.body, n.level);
  try { if (n.level === "danger" && window.Notification && Notification.permission === "granted")
    new Notification(n.title, { body: n.body, tag: n.id }); } catch (e) { /* not supported */ }
});
for (const topic of ["summary", "locations", "devices", "notifications", "persons", "duplicates", "connection", "trust"]) on(topic, render);
on("status", drawTopbar);
on("event", (ev) => { onOverviewEvent(ev); onLocationEvent(ev); });
setInterval(() => clear($("#clock"), new Date().toLocaleTimeString("uk-UA", { hour12: false })), 1000);

route = parse(location.hash);
connectVideo();
start().then(draw);
draw();
