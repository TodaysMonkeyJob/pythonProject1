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
export const STATE_RANK = { danger: 0, attention: 1, safe: 2, offline: 3 };

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
