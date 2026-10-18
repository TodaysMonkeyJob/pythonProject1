// Separate pages from the top bar: People (rename / merge / delete), Devices (locations, on/off, pairing, RTSP,
// ONVIF), Locations (create / rename / reorder / rules / delete).
import { api, loadDevices, loadLocations, loadPersons, store } from "../store.js";
import { dialog, h, hms } from "../ui.js";

const KIND = { camera: "камера", mic: "мікрофон", gas: "газ" };
const guard = (fn) => async (...args) => { try { await fn(...args); } catch (e) { alert(e.message); } };

export function renderPeople(root) {
  const list = Object.values(store.persons).sort((a, b) => (b.is_known - a.is_known) || String(b.last_seen).localeCompare(String(a.last_seen)));
  root.replaceChildren(h("div", { class: "admin" }, h("h1", {}, "Люди"),
    h("p", { class: "muted" }, "Клікніть ім'я, щоб перейменувати — зміниться всюди одразу."),
    h("div", { class: "cards" }, list.length ? list.map((p) => personCard(p, list, root)) : h("div", { class: "muted" }, "реєстр порожній"))));
}

function personCard(p, list, root) {
  const name = h("div", { class: "name", title: "перейменувати" }, p.display_name);
  name.onclick = () => {
    const input = h("input", { value: p.display_name });
    input.onkeydown = guard(async (e) => {
      e.stopPropagation();
      if (e.key === "Escape") return input.replaceWith(name);
      if (e.key !== "Enter" || !input.value.trim()) return;
      await api("PATCH", `/api/persons/${encodeURIComponent(p.person_id)}`, { display_name: input.value.trim() });
      await loadPersons(); renderPeople(root);
    });
    name.replaceWith(input); input.focus(); input.select();
  };
  const merge = h("select", { onchange: guard(async (e) => {
    const target = e.target.value; if (!target) return;
    if (confirm(`${p.display_name} → ${store.persons[target].display_name}: це одна людина?`)) {
      await api("POST", `/api/persons/${encodeURIComponent(p.person_id)}/merge`, { target_id: target }); await loadPersons(); renderPeople(root);
    } else e.target.value = "";
  }) }, h("option", { value: "" }, "Об'єднати з…"), list.filter((o) => o.person_id !== p.person_id).map((o) => h("option", { value: o.person_id }, o.display_name)));
  return h("div", { class: `card${p.is_known ? "" : " unknown"}` },
    h("img", { alt: p.display_name, loading: "lazy", src: `/api/persons/${encodeURIComponent(p.person_id)}/thumbnail?v=${p.face_samples}` }),
    name, h("div", { class: "muted" }, `обличчя: ${p.face_samples} · голос: ${p.voice_samples}`),
    h("div", { class: "muted" }, p.last_seen ? `востаннє: ${new Date(p.last_seen).toLocaleString("uk-UA")} · ${p.last_location || ""}` : "—"),
    h("div", { class: "row" }, merge, h("button", { onclick: guard(async () => {
      if (!confirm(`Видалити ${p.display_name} разом з усіма зразками?`)) return;
      await api("DELETE", `/api/persons/${encodeURIComponent(p.person_id)}`); await loadPersons(); renderPeople(root);
    }) }, "Видалити")));
}

export function renderDevices(root) {
  const options = (d) => [...store.locations.map((l) => h("option", { value: l.location_id, selected: l.location_id === d.location_id }, l.name)),
                          h("option", { value: "__new" }, "+ нова локація…")];
  const rows = store.devices.map((d) => h("div", { class: `devrow${d.status === "online" ? "" : " offline"}` },
    h("span", { class: `dot ${d.running ? "on" : ""}`, title: d.running ? "працює" : d.status }),
    h("span", { title: d.device_id }, d.name), h("span", { class: "muted" }, KIND[d.kind] || d.kind), h("span", { class: "muted" }, d.transport),
    h("select", { onchange: guard(async (e) => {
      let id = e.target.value;
      if (id === "__new") { const n = prompt("Назва локації"); if (!n) { e.target.value = d.location_id; return; }
        id = (await api("POST", "/api/locations", { name: n })).location_id; await loadLocations(); }
      await api("PATCH", `/api/devices/${encodeURIComponent(d.device_id)}`, { location_id: id }); await loadDevices(); renderDevices(root);
    }) }, options(d)),
    h("label", { class: "muted" }, h("input", { type: "checkbox", checked: d.enabled, onchange: guard(async (e) => {
      await api("PATCH", `/api/devices/${encodeURIComponent(d.device_id)}`, { enabled: e.target.checked }); await loadDevices(); renderDevices(root); }) }), " увімк."),
    h("span", { class: "muted" }, d.status !== "online" ? `${d.status}${d.last_seen ? " · " + hms(d.last_seen) : ""}`
                                  : d.kind === "camera" ? (d.fps ? `${d.fps} FPS` : "старт…") : d.kind === "mic" ? (d.mic_state || "старт…") : "online")));
  const st = store.status || {}, fmt = (v, unit = "") => (v === null || v === undefined ? "—" : v + unit);
  const system = h("div", { class: "status-kv" }, [
    ["CPU", fmt(st.cpu_percent, "%")], ["температура CPU", fmt(st.cpu_temp_c, " °C")],
    ["RAM", st.ram_total_mb ? `${st.ram_used_mb} / ${st.ram_total_mb} MB` : "—"], ["аптайм", fmt(st.uptime_sec, " с")]
  ].map(([k, v]) => [h("span", { class: "muted" }, k), h("span", {}, v)]));
  const modules = h("div", { class: "health" }, Object.entries(st.modules || {}).map(([name, m]) =>
    h("div", { class: "row" }, h("span", { class: `dot ${m.alive ? "on" : ""}` }), h("b", {}, name),
      h("span", { class: "muted" }, m.alive ? `${m.age_sec} с тому` : `немає heartbeat ${m.age_sec} с`),
      h("span", {}, m.frame_fps ? `${m.frame_fps} FPS` : m.mic_state || ""))));
  root.replaceChildren(h("div", { class: "admin" }, h("h1", {}, "Пристрої"),
    h("div", { class: "filters" },
      h("button", { onclick: guard(pairPhone) }, "Підключити телефон"),
      h("button", { onclick: guard(async () => {
        const url = prompt("RTSP-адреса, напр. rtsp://user:pass@192.168.0.50:554/stream1"); if (!url) return;
        await api("POST", "/api/devices/rtsp", { name: prompt("Назва камери", "IP камера") || "IP камера", url }); await loadDevices(); renderDevices(root);
      }) }, "IP-камера (RTSP)"),
      h("button", { onclick: guard(onvif) }, "Пошук ONVIF")),
    h("div", { class: "devlist" }, rows.length ? rows : h("div", { class: "muted" }, "пристроїв ще немає")),
    h("h2", {}, "Система"), system, h("h2", {}, "Модулі"), modules));
}

async function pairPhone() {
  const j = await api("POST", "/api/pairing-code");
  dialog("Підключити телефон або ноутбук",
    h("p", {}, "1. На телефоні в тій самій Wi-Fi мережі відкрийте:"), h("pre", {}, store.nodeUrl || "https://<ip-хаба>:8443/node"),
    h("p", { class: "muted" }, "Браузер попередить про самопідписаний сертифікат хаба — підтвердіть."),
    h("p", {}, "2. Введіть код (10 хв):"), h("div", { class: "code" }, j.code), h("p", {}, "3. Оберіть локацію й почніть трансляцію."));
}

async function onvif() {
  const body = dialog("Пошук IP-камер (ONVIF)", h("p", { class: "muted" }, "шукаю 3 с…"));
  const j = await api("POST", "/api/devices/onvif-scan");
  body.replaceChildren(...(j.found.length ? j.found.map((f) => h("p", {}, `${f.name || "камера"} ${f.hardware || ""} — ${f.host}`))
                                          : [h("p", {}, "Нічого не знайдено.")]));
}

export function renderLocations(root) {
  const list = store.locations;
  const rows = list.map((l, i) => {
    const icon = h("input", { value: l.icon || "", placeholder: "📦", maxlength: 4, class: "icon-input" });
    const name = h("input", { value: l.name });
    const after = h("input", { value: (l.rules || {}).unknown_person_after || "", placeholder: "20:00", title: "невідома людина підозріла після" });
    const gas = h("input", { value: Object.entries((l.rules || {}).gas_thresholds || {}).map(([k, v]) => `${k}=${v}`).join(", "),
                             placeholder: "acetone=3.0", title: "газові пороги, %" });
    const fixed = l.location_id === "unassigned";
    return h("div", { class: "locrow" }, icon, name, after, gas,
      h("button", { onclick: guard(async () => {
        const thresholds = Object.fromEntries(gas.value.split(",").map((x) => x.split("=").map((s) => s.trim())).filter((x) => x[0] && x[1]));
        await api("PATCH", `/api/locations/${encodeURIComponent(l.location_id)}`, { name: name.value, icon: icon.value,
          rules: { ...(l.rules || {}), unknown_person_after: after.value || null, gas_thresholds: thresholds } });
        await loadLocations(); renderLocations(root);
      }) }, "Зберегти"),
      h("button", { disabled: i === 0 || fixed, title: "вище", onclick: guard(async () => {
        const ids = list.map((x) => x.location_id); [ids[i - 1], ids[i]] = [ids[i], ids[i - 1]];
        await api("POST", "/api/locations/reorder", { order: ids }); await loadLocations(); renderLocations(root);
      }) }, "↑"),
      h("button", { disabled: fixed, title: "видалити", onclick: guard(async () => {
        if (!confirm(`Видалити ${l.name}? Пристрої перейдуть у «Unassigned».`)) return;
        await api("DELETE", `/api/locations/${encodeURIComponent(l.location_id)}`); await loadLocations(); await loadDevices(); renderLocations(root);
      }) }, "✕"));
  });
  root.replaceChildren(h("div", { class: "admin" }, h("h1", {}, "Локації"),
    h("div", { class: "locrow head muted" }, "іконка", "назва", "невідомий після", "газові пороги, %", "", "", ""),
    rows, h("button", { onclick: guard(async () => { const n = prompt("Назва локації"); if (!n) return;
      await api("POST", "/api/locations", { name: n }); await loadLocations(); renderLocations(root); }) }, "+ локація")));
}
