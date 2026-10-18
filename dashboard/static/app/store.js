// Client state: locations + live summaries, devices, people (names by person_id), notifications, recent events.
// Fed by REST on start and by the /ws event stream; views subscribe with on(topic, fn).

export const store = {
  locations: [],          // [{location_id, name, order, icon, rules}]
  summaries: {},          // location_id → location_summary
  devices: [],            // /api/devices
  nodeUrl: null,
  persons: {}, aliases: {},
  notifications: [],
  events: [],             // newest last, capped
  status: {},
  connected: false,
};
const MAX_EVENTS = 3000;
const listeners = {};

export function on(topic, fn) { (listeners[topic] ||= new Set()).add(fn); return () => listeners[topic].delete(fn); }
export function emit(topic, payload) { (listeners[topic] || []).forEach((fn) => fn(payload)); }

export async function api(method, url, body) {
  const r = await fetch(url, { method, headers: body === undefined ? {} : { "Content-Type": "application/json" },
                               body: body === undefined ? undefined : JSON.stringify(body) });
  if (!r.ok) {
    let detail = r.status + "";
    try { detail = (await r.json()).detail || detail; } catch (e) { /* not JSON */ }
    throw new Error(detail);
  }
  return r.json();
}
const get = (url) => api("GET", url);

// --- names: always resolved from person_id (a rename applies to old lines too) -------------------------------------
export const livePid = (pid) => store.aliases[pid] || pid;
export const personName = (pid, fallback) => (store.persons[livePid(pid)] || {}).display_name || fallback || "?";

// --- loading -------------------------------------------------------------------------------------------------------
export async function loadLocations() {
  const j = await get("/api/locations");
  store.locations = j.locations;
  j.locations.forEach((l) => { if (l.summary) store.summaries[l.location_id] = l.summary; });
  emit("locations");
}
export async function loadDevices() {
  const j = await get("/api/devices");
  store.devices = j.devices; store.nodeUrl = j.node_url;
  emit("devices");
}
export async function loadPersons() {
  const j = await get("/api/persons");
  store.persons = {}; store.aliases = j.aliases || {};
  j.persons.forEach((p) => { store.persons[p.person_id] = p; });
  emit("persons");
}
export async function loadNotifications() {
  store.notifications = (await get("/api/notifications?limit=50")).notifications;
  emit("notifications");
}
export async function loadStatus() { store.status = await get("/api/status"); emit("status"); }
export const history = (params) => get("/api/history?" + new URLSearchParams(params)).then((j) => j.events);

export const unacked = () => store.notifications.filter((n) => !n.acknowledged_at);
export const cameras = () => store.devices.filter((d) => d.kind === "camera");
export const locationOf = (id) => store.locations.find((l) => l.location_id === id);
export const stateOf = (id) => (store.summaries[id] || {}).state || "safe";

export function orderedLocations() {          // the carousel: by order; "unassigned" only when it has devices
  return store.locations.filter((l) => l.location_id !== "unassigned"
                                || store.devices.some((d) => d.location_id === "unassigned"));
}

export async function ackLocation(id) { await api("POST", `/api/locations/${encodeURIComponent(id)}/ack`); await loadNotifications(); }
export async function ackNotification(id) { await api("POST", `/api/notifications/${encodeURIComponent(id)}/ack`); await loadNotifications(); }
export async function renamePerson(pid, name) {
  await api("PATCH", `/api/persons/${encodeURIComponent(pid)}`, { display_name: name });
  if (store.persons[pid]) store.persons[pid].display_name = name;
  emit("persons");
}

// --- live events ---------------------------------------------------------------------------------------------------
let delay = 1000;
export function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { store.connected = true; delay = 1000; store.events = []; emit("connection"); };
  ws.onclose = () => { store.connected = false; emit("connection"); setTimeout(connect, delay); delay = Math.min(delay * 2, 15000); };
  ws.onmessage = (m) => { try { handle(JSON.parse(m.data)); } catch (e) { console.warn(e); } };
}

function handle(ev) {
  const p = ev.payload;
  if (ev.type === "location_summary") { store.summaries[p.location_id] = p; emit("summary", p); return; }
  store.events.push(ev);
  if (store.events.length > MAX_EVENTS) store.events.splice(0, store.events.length - MAX_EVENTS);
  emit("event", ev);
  if (ev.type === "notification") { store.notifications.unshift(p); emit("notifications"); emit("notify", p); }
  else if (ev.type === "notification_acked") loadNotifications();
  else if (ev.type.startsWith("device_")) loadDevices();
  else if (ev.type.startsWith("identity_")) {
    if (ev.type === "identity_renamed" && store.persons[p.person_id]) store.persons[p.person_id].display_name = p.display_name;
    loadPersons();
    emit("persons");
  }
}

export async function start() {
  await Promise.allSettled([loadLocations(), loadDevices(), loadPersons(), loadNotifications(), loadStatus()]);
  connect();
  setInterval(() => loadStatus().catch(() => {}), 3000);
  setInterval(() => loadDevices().catch(() => {}), 10000);
  setInterval(() => loadLocations().catch(() => {}), 15000);
}
