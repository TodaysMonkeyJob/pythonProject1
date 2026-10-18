// Control room (F13): hash router + horizontal carousel (Overview, one page per location) + top bar pages.
//   #/overview   #/loc/<id>   #/loc/<id>/details   #/people   #/devices   #/locations
// ← / → (or swipe, or ◀ ▶) move through the carousel, ↓ / ↑ between a location's summary and details.
import { cameras, on, orderedLocations, start, stateOf, store, unacked } from "./store.js";
import { STATE_LABEL, STATE_RANK, $, clear, h, plural, toast } from "./ui.js";
import { connectVideo, syncSubscriptions } from "./video.js";
import { renderDevices, renderLocations, renderPeople } from "./views/admin.js";
import { leaveLocation, onLocationEvent, redrawCharts, renderLocation } from "./views/location.js";
import { notificationFeed, renderOverview } from "./views/overview.js";

const view = $("#view");
let route = { page: "overview" };

function parse(hash) {
  const [path, query] = (hash || "#/overview").replace(/^#\/?/, "").split("?");
  const parts = path.split("/").map(decodeURIComponent), params = new URLSearchParams(query || "");
  if (parts[0] === "loc" && parts[1]) return { page: "loc", id: parts[1], details: parts[2] === "details", focus: params.get("focus") };
  if (["people", "devices", "locations"].includes(parts[0])) return { page: parts[0] };
  return { page: "overview" };
}

const carousel = () => ["#/overview", ...orderedLocations().map((l) => `#/loc/${encodeURIComponent(l.location_id)}`)];
function carouselIndex() {
  if (route.page === "overview") return 0;
  if (route.page !== "loc") return -1;
  return Math.max(0, carousel().indexOf(`#/loc/${encodeURIComponent(route.id)}`));
}
function neighbour(step) {
  const pages = carousel(), i = carouselIndex();
  return i < 0 ? pages[0] : pages[(i + step + pages.length) % pages.length];
}

// go(null) re-renders the current page; go(hash) navigates (history entry → back / forward work)
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
  const ctx = { go, neighbours: { prev: neighbour(-1), next: neighbour(1) } };
  if (route.page === "overview") renderOverview(view, ctx);
  else if (route.page === "loc") renderLocation(view, { ...ctx, id: route.id, details: route.details, focus: route.focus });
  else if (route.page === "people") renderPeople(view);
  else if (route.page === "devices") renderDevices(view);
  else if (route.page === "locations") renderLocations(view);
  route.details = false; route.focus = null;           // only on arrival
  drawTopbar(); drawDots(); syncSubscriptions();
}

function drawTopbar() {
  const states = orderedLocations().map((l) => stateOf(l.location_id));
  const worst = states.sort((a, b) => STATE_RANK[a] - STATE_RANK[b])[0] || "safe";
  const cams = cameras(), online = cams.filter((c) => c.status === "online").length;
  clear($("#site"), h("span", { class: `dot st-${worst}` }), ` ${STATE_LABEL[worst] || worst}`);
  clear($("#cams-online"), `${online}/${cams.length} камер онлайн`);
  const dups = store.duplicates.length;
  $("#dups").hidden = !dups; $("#dups").textContent = `${dups} ${plural(dups, "можливий дублікат", "можливі дублікати", "можливих дублікатів")}`;
  const open = unacked().length;
  $("#bell-count").hidden = !open; $("#bell-count").textContent = open;
  $("#ws-dot").classList.toggle("on", store.connected);
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("on", a.getAttribute("href") === `#/${route.page}`
                                                                        || (route.page === "loc" && a.getAttribute("href") === "#/overview")));
  if ($("#notes").open) clear($("#note-list"), notificationFeed(30));
}

function drawDots() {
  const box = $("#dots"), i = carouselIndex(), pages = carousel();
  box.hidden = i < 0;
  clear(box, pages.map((hash, n) => {
    const id = hash.startsWith("#/loc/") ? decodeURIComponent(hash.slice(6)) : null;
    return h("button", { class: `${n === i ? "on" : ""} st-${id ? stateOf(id) : "overview"}`, title: id || "Overview",
                         onclick: () => go(hash) });
  }));
}

// --- input ---------------------------------------------------------------------------------------------------------
window.addEventListener("hashchange", () => { route = parse(location.hash); draw(); });
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea") || e.metaKey || e.ctrlKey || $("#dlg").open) return;
  if (e.key === "ArrowRight" && carouselIndex() >= 0) { go(neighbour(1)); e.preventDefault(); }
  if (e.key === "ArrowLeft" && carouselIndex() >= 0) { go(neighbour(-1)); e.preventDefault(); }
  const page = view.querySelector(".loc-page");
  if (page && e.key === "ArrowDown") { page.querySelector(".details").scrollIntoView({ behavior: "smooth" }); e.preventDefault(); }
  if (page && e.key === "ArrowUp") { page.querySelector(".summary").scrollIntoView({ behavior: "smooth" }); e.preventDefault(); }
});
let swipe = null;
view.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse") swipe = { x: e.clientX, y: e.clientY }; });
view.addEventListener("pointerup", (e) => {
  if (!swipe || carouselIndex() < 0) { swipe = null; return; }
  const dx = e.clientX - swipe.x, dy = e.clientY - swipe.y; swipe = null;
  if (Math.abs(dx) > 60 && Math.abs(dx) > 1.5 * Math.abs(dy)) go(neighbour(dx < 0 ? 1 : -1));
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
for (const topic of ["summary", "locations", "devices", "notifications", "persons", "duplicates", "connection"]) on(topic, render);
on("status", drawTopbar);
on("event", (ev) => { if (onLocationEvent(ev)) render(); });
setInterval(() => clear($("#clock"), new Date().toLocaleTimeString("uk-UA", { hour12: false })), 1000);

route = parse(location.hash);
connectVideo();
start().then(draw);
draw();
