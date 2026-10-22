// F23 (TOD-34) page «Довіра»: trust per device × channel, grouped location → device → channel. Columns τ, u, h,
// prior status, last change, explanation; "only degraded" filter; a channel opens its τ / h history (health events
// marked) with operator feedback buttons (confirm / refute → only that channel's counters).
import { locationOf, sendTrustFeedback, store, trustHistory } from "../store.js";
import { clear, dialog, h, hms, toast } from "../ui.js";
import { trustChart } from "../charts.js";

const KIND = { camera: "камера", mic: "мікрофон", gas: "сенсор газу", identity: "реєстр людей" };
const CHANNEL = { objects: "об'єкти", motion: "рух", pose_action: "пози / дії", smoking: "куріння",
                  face_identity: "обличчя", scene_text: "текст у кадрі", vad: "наявність мови", asr: "транскрипт",
                  speaker_identity: "голос", face_voice_link: "голос ↔ обличчя", gas_class: "тип газу",
                  gas_concentration: "концентрація", gas_trend: "динаміка", person_registry: "реєстр людей" };
let onlyDegraded = false;

export const isDegraded = (c) => c.tau < 0.4 || c.h < 0.999 || (c.reasons || []).length > 0;
const fmt = (x) => (x ?? 0).toFixed(2).replace(".", ",");
const level = (tau) => (tau < 0.4 ? "bad" : tau < 0.6 ? "warn" : "ok");

export function renderTrust(view) {
  const channels = store.trust.filter((c) => !onlyDegraded || isDegraded(c));
  const byLocation = new Map();
  for (const c of channels) {
    const loc = c.location_id || "—";
    if (!byLocation.has(loc)) byLocation.set(loc, new Map());
    const devices = byLocation.get(loc);
    if (!devices.has(c.device_id)) devices.set(c.device_id, []);
    devices.get(c.device_id).push(c);
  }
  const degraded = store.trust.filter(isDegraded).length;
  clear(view, h("section", { class: "admin trust-page" },
    h("div", { class: "admin-head" },
      h("h2", {}, "Довіра"),
      h("span", { class: "muted" }, `${store.trust.length} каналів · ${degraded} з проблемами`),
      h("label", { class: "check" }, h("input", { type: "checkbox", checked: onlyDegraded,
        onchange: (e) => { onlyDegraded = e.target.checked; renderTrust(view); } }), " лише деградовані")),
    store.trustError ? h("p", { class: "muted" }, store.trustError) : null,
    h("div", { class: "trust-scroll" },
      [...byLocation.entries()].sort().map(([loc, devices]) => {
        const location = locationOf(loc);
        return h("div", { class: "trust-loc" },
          h("h3", {}, location ? `${location.icon || ""} ${location.name}`.trim() : loc),
          [...devices.entries()].sort().map(([device, rows]) => h("table", { class: "trust-table" },
            h("caption", {}, `${KIND[rows[0].device_kind] || rows[0].device_kind} ${device}`),
            h("thead", {}, h("tr", {}, ["канал", "τ", "u", "h", "апріорі", "зміна", "пояснення"].map((t) => h("th", {}, t)))),
            h("tbody", {}, rows.map((c) => h("tr", { class: `trust-row ${level(c.tau)}`, tabindex: "0",
                onclick: () => openChannel(c), onkeydown: (e) => { if (e.key === "Enter") openChannel(c); } },
              h("td", {}, CHANNEL[c.channel_kind] || c.channel_kind),
              h("td", { class: "mono" }, h("span", { class: `trust-pill ${level(c.tau)}` }, fmt(c.tau))),
              h("td", { class: "mono" }, fmt(c.u)),
              h("td", { class: "mono" }, fmt(c.h), c.h < (c.h_target ?? c.h) - 1e-6 ? " ↗" : ""),
              h("td", {}, c.prior_status === "measured" ? "виміряно" : "не виміряно"),
              h("td", { class: "mono muted" }, c.updated_at ? hms(c.updated_at) : "—"),
              h("td", { class: "trust-why" }, c.explanation)))))));
      }),
      channels.length ? null : h("p", { class: "muted" }, onlyDegraded ? "усі канали в нормі" : "каналів ще немає"))));
}

async function openChannel(c) {
  const canvas = h("canvas", { class: "trust-chart", "aria-label": `історія довіри ${c.channel_id}` });
  const log = h("ol", { class: "trust-log" });
  const feedback = (verdict) => async () => {
    const reason = verdict === "confirm" ? "оператор підтвердив" : "оператор спростував";
    try { await sendTrustFeedback({ channel_id: c.channel_id, verdict, reason }); toast("Довіра", `${c.channel_id}: ${reason}`, "safe"); }
    catch (e) { toast("Помилка", e.message, "danger"); }
  };
  dialog(`${KIND[c.device_kind] || c.device_kind} ${c.device_id} · ${CHANNEL[c.channel_kind] || c.channel_kind}`,
    h("p", {}, c.explanation),
    h("p", { class: "muted" }, `r ${fmt(c.r)} · s ${fmt(c.s)} · γ ${c.gamma}/добу · апріорі: `,
      c.prior_status === "measured" ? `${c.prior.explanation || ""} (${c.prior.source || ""})` : "не виміряно"),
    canvas, log,
    h("div", { class: "row-actions" },
      h("button", { onclick: feedback("confirm") }, "✓ Підтвердити"),
      h("button", { onclick: feedback("refute") }, "✗ Спростувати")));
  try {
    const rows = await trustHistory(c.channel_id);
    requestAnimationFrame(() => trustChart(canvas, rows));
    clear(log, rows.slice(-12).reverse().map((r) => h("li", {}, h("span", { class: "mono muted" }, hms(r.at)), " ",
      `${fmt(r.tau)} — ${r.cause}`)));
  } catch (e) { clear(log, h("li", { class: "muted" }, e.message)); }
}

// Event-dialog feedback (F23): buttons for the event's channel; a transcript can be corrected
export function feedbackButtons(ev) {
  const p = ev.payload || {};
  if (!p.channel_id) return null;
  const send = (body, label) => async () => {
    try { await sendTrustFeedback({ channel_id: p.channel_id, event_id: ev.event_id, ...body }); toast("Довіра", label, "safe"); }
    catch (e) { toast("Помилка", e.message, "danger"); }
  };
  const asr = ev.type === "speech" ? `${ev.device_id}/asr` : null;
  const text = asr ? h("input", { type: "text", value: p.text || "", "aria-label": "виправлений транскрипт" }) : null;
  return h("div", { class: "row-actions" },
    h("span", { class: "muted" }, `канал ${p.channel_id}: `),
    h("button", { onclick: send({ verdict: "confirm", reason: `вірно: ${ev.type}` }, "підтверджено") }, "✓ Вірно"),
    h("button", { onclick: send({ verdict: "refute", reason: `хибно: ${ev.type}` }, "спростовано") }, "✗ Хибно"),
    asr ? [text, h("button", { onclick: async () => {
      try { await sendTrustFeedback({ channel_id: asr, original: p.text || "", corrected: text.value, event_id: ev.event_id });
            toast("Довіра", "транскрипт виправлено", "safe"); }
      catch (e) { toast("Помилка", e.message, "danger"); } } }, "Виправити транскрипт")] : null);
}
