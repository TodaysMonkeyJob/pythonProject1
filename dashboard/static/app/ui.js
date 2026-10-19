// Small DOM helpers: every piece of event / user text goes through textContent, never innerHTML.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export const $ = (selector, root = document) => root.querySelector(selector);

export function clear(el, ...children) { el.replaceChildren(); append(el, children); return el; }

export const STATE_LABEL = { safe: "безпечно", attention: "увага", danger: "небезпека", offline: "офлайн" };
export const STATE_RANK = { danger: 0, attention: 1, offline: 2, safe: 3 };     // F21: severity order of the cards

// Ukrainian plural: 1 дублікат, 2–4 дублікати, 5+ / 11–14 дублікатів
export function plural(n, one, few, many) {
  const d = n % 10, dd = n % 100;
  return d === 1 && dd !== 11 ? one : d >= 2 && d <= 4 && (dd < 12 || dd > 14) ? few : many;
}

export function hms(ts) {
  const date = typeof ts === "number" ? new Date(ts < 1e12 ? ts * 1000 : ts) : new Date(ts);
  return date.toLocaleTimeString("uk-UA", { hour12: false });
}

export function since(ts) {
  const seconds = Math.max(0, (Date.now() - new Date(ts).getTime()) / 1000);
  if (seconds < 60) return `${Math.round(seconds)} с`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} хв`;
  return `${Math.round(seconds / 3600)} год`;
}

// A stable colour per person (transcripts colour-coded by speaker)
export function personColour(key) {
  let hash = 0x811c9dc5;                                   // FNV-1a, then golden-angle hues: neighbours differ
  for (const ch of String(key || "?")) hash = Math.imul(hash ^ ch.charCodeAt(0), 0x01000193) >>> 0;
  return `hsl(${Math.round((hash % 1000) * 137.508) % 360} 70% 62%)`;
}

export function toast(title, body, level = "attention") {
  const box = $("#toasts");
  const node = h("div", { class: `toast ${level}` }, h("b", {}, title), h("div", {}, body));
  box.append(node);
  setTimeout(() => node.remove(), 8000);
}

export function dialog(title, ...content) {
  const dlg = $("#dlg");
  clear($("#dlg-title"), title);
  clear($("#dlg-body"), ...content);
  dlg.showModal();
  return $("#dlg-body");
}

export function json(value) { return h("pre", { class: "json" }, JSON.stringify(value, null, 1)); }

export function downloadCsv(name, rows) {
  const text = rows.map((r) => r.map((v) => `"${String(v ?? "").replaceAll('"', '""')}"`).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob(["﻿" + text], { type: "text/csv;charset=utf-8" }));
  h("a", { href: url, download: name }).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// F21: a newest-first log that never jumps. Live items are prepended; when the user has scrolled down inside the
// list (> 40 px) the visible rows stay where they are (scrollTop grows by the inserted height) and a "N нових ↑" pill
// appears — clicking it scrolls THIS list back to its top. CSS sets overflow-anchor: none so the browser's own scroll
// anchoring doesn't adjust twice. The page itself is never scrolled.
export class LiveList {
  constructor({ cls = "", max = 400, empty = "нічого", label = "журнал", onScroll = null } = {}) {
    this.max = max;
    this.pending = 0;
    this.emptyText = empty;
    this.list = h("div", { class: `live-list ${cls}`, role: "log", tabindex: "0", "aria-label": label });
    this.pill = h("button", { class: "new-pill", hidden: true, "aria-label": "показати нові записи",
                              onclick: () => this.top() });
    this.el = h("div", { class: "live" }, this.pill, this.list);
    this.list.addEventListener("scroll", () => {
      if (this.list.scrollTop <= 40) this.clearPill();
      if (onScroll) onScroll(this.list.scrollTop);
    });
  }

  set(nodes) {
    this.list.replaceChildren(...(nodes.length ? nodes : [h("div", { class: "muted empty" }, this.emptyText)]));
    this.clearPill();
  }

  prepend(nodes) {
    if (!nodes.length) return;
    this.list.querySelector(":scope > .empty")?.remove();
    const scrolled = this.list.scrollTop > 40, before = this.list.scrollHeight;
    this.list.prepend(...nodes);
    while (this.list.childElementCount > this.max) this.list.lastElementChild.remove();
    if (scrolled) {
      this.list.scrollTop += this.list.scrollHeight - before;
      this.pending += nodes.length;
      this.pill.textContent = `${this.pending} нових ↑`;
      this.pill.hidden = false;
    }
  }

  top() { this.list.scrollTop = 0; this.clearPill(); }
  clearPill() { this.pending = 0; this.pill.hidden = true; }
}
