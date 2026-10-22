// Tiny canvas charts (no CDN, works offline): a sparkline and a time-series line chart with a threshold.

function prepare(canvas) {
  const ratio = window.devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, rect.width * ratio); canvas.height = Math.max(1, rect.height * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  return { ctx, width: rect.width, height: rect.height };
}
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// points: [{t: ms, v: number}]
export function sparkline(canvas, points, { threshold = null } = {}) {
  const { ctx, width, height } = prepare(canvas);
  ctx.clearRect(0, 0, width, height);
  if (points.length < 2) return;
  const t0 = points[0].t, t1 = points[points.length - 1].t || t0 + 1;
  const max = Math.max(threshold ?? 0, ...points.map((p) => p.v)) * 1.1 || 1;
  const x = (t) => ((t - t0) / Math.max(1, t1 - t0)) * width, y = (v) => height - (v / max) * (height - 2) - 1;
  if (threshold !== null) {
    ctx.strokeStyle = css("--critical"); ctx.setLineDash([4, 3]); ctx.beginPath();
    ctx.moveTo(0, y(threshold)); ctx.lineTo(width, y(threshold)); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = css("--accent"); ctx.lineWidth = 1.5; ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(p.t), y(p.v)) : ctx.moveTo(x(p.t), y(p.v))));
  ctx.stroke();
}

export function lineChart(canvas, points, { threshold = null, label = "" } = {}) {
  const { ctx, width, height } = prepare(canvas);
  const pad = { left: 40, right: 8, top: 14, bottom: 18 }, w = width - pad.left - pad.right, hgt = height - pad.top - pad.bottom;
  ctx.clearRect(0, 0, width, height);
  ctx.font = "11px system-ui"; ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--line");
  if (points.length < 2) { ctx.fillText("немає даних", pad.left, pad.top + 12); return; }
  const t0 = points[0].t, t1 = points[points.length - 1].t;
  const max = Math.max(threshold ?? 0, ...points.map((p) => p.v)) * 1.15 || 1;
  const x = (t) => pad.left + ((t - t0) / Math.max(1, t1 - t0)) * w, y = (v) => pad.top + hgt - (v / max) * hgt;
  for (let i = 0; i <= 4; i++) {                       // grid + y labels
    const v = (max * i) / 4; ctx.beginPath(); ctx.moveTo(pad.left, y(v)); ctx.lineTo(width - pad.right, y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(v < 10 ? 1 : 0), 4, y(v) + 4);
  }
  for (let i = 0; i <= 4; i++) {                       // x labels
    const t = t0 + ((t1 - t0) * i) / 4;
    ctx.fillText(new Date(t).toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" }), x(t) - 14, height - 4);
  }
  if (threshold !== null) {
    ctx.strokeStyle = css("--critical"); ctx.setLineDash([6, 4]); ctx.beginPath();
    ctx.moveTo(pad.left, y(threshold)); ctx.lineTo(width - pad.right, y(threshold)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css("--critical"); ctx.fillText(`поріг ${threshold}`, width - pad.right - 60, y(threshold) - 4);
  }
  ctx.strokeStyle = css("--accent"); ctx.lineWidth = 1.5; ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(p.t), y(p.v)) : ctx.moveTo(x(p.t), y(p.v))));
  ctx.stroke();
  if (label) { ctx.fillStyle = css("--text"); ctx.fillText(label, pad.left + 4, 11); }
}

// segments: [{from: ms, to: ms, state}] → coloured bar (state timeline)
export function timeline(canvas, segments, from, to) {
  const { ctx, width, height } = prepare(canvas);
  ctx.clearRect(0, 0, width, height);
  const colour = { safe: css("--low"), attention: css("--medium"), danger: css("--critical"), offline: css("--muted") };
  const x = (t) => ((t - from) / Math.max(1, to - from)) * width;
  for (const s of segments) {
    ctx.fillStyle = colour[s.state] || css("--muted");
    ctx.fillRect(x(Math.max(s.from, from)), 0, Math.max(1, x(Math.min(s.to, to)) - x(Math.max(s.from, from))), height);
  }
}

// F21 gas tab: 10-minute bar chart (one bar per 30 s, the bucket's max), coloured against the threshold
export function bars(canvas, points, { threshold = null, minutes = 10, buckets = 20 } = {}) {
  const { ctx, width, height } = prepare(canvas);
  ctx.clearRect(0, 0, width, height);
  const to = Date.now(), from = to - minutes * 60e3, step = (to - from) / buckets;
  const values = Array.from({ length: buckets }, (_, i) => {
    const inside = points.filter((p) => p.t >= from + i * step && p.t < from + (i + 1) * step);
    return inside.length ? Math.max(...inside.map((p) => p.v)) : null;
  });
  const max = Math.max(threshold ?? 0, ...values.filter((v) => v !== null)) * 1.15 || 1;
  const w = width / buckets;
  values.forEach((v, i) => {
    if (v === null) return;
    const level = threshold === null ? "--safe" : v >= threshold ? "--danger" : v >= 0.6 * threshold ? "--attention" : "--safe";
    ctx.fillStyle = css(level);
    const barHeight = Math.max(1, (v / max) * (height - 14));
    ctx.fillRect(i * w + 1, height - barHeight, w - 2, barHeight);
  });
  if (threshold !== null) {
    const y = height - (threshold / max) * (height - 14);
    ctx.strokeStyle = css("--danger"); ctx.setLineDash([4, 3]); ctx.beginPath();
    ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css("--muted"); ctx.font = "11px sans-serif"; ctx.fillText(`поріг ${threshold}%`, 4, Math.max(11, y - 3));
  }
}

// F23: τ and h of one trust channel over time; health changes marked as ticks. rows: /api/trust/history
export function trustChart(canvas, rows) {
  const { ctx, width, height } = prepare(canvas);
  const pad = { left: 30, right: 8, top: 14, bottom: 18 }, w = width - pad.left - pad.right, hgt = height - pad.top - pad.bottom;
  ctx.clearRect(0, 0, width, height);
  ctx.font = "11px system-ui"; ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--line");
  if (!rows.length) { ctx.fillText("немає історії", pad.left, pad.top + 12); return; }
  const t0 = rows[0].at, t1 = Math.max(rows[rows.length - 1].at, t0 + 1);
  const x = (t) => pad.left + ((t - t0) / (t1 - t0)) * w, y = (v) => pad.top + hgt - v * hgt;
  for (const v of [0, 0.4, 0.6, 1]) {
    ctx.beginPath(); ctx.moveTo(pad.left, y(v)); ctx.lineTo(width - pad.right, y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(1), 4, y(v) + 4);
  }
  for (let i = 0; i <= 3; i++) {
    const t = t0 + ((t1 - t0) * i) / 3;
    ctx.fillText(new Date(t * 1000).toLocaleTimeString("uk-UA", { hour: "2-digit", minute: "2-digit" }), x(t) - 14, height - 4);
  }
  const line = (key, colour, dash) => {
    ctx.strokeStyle = colour; ctx.setLineDash(dash); ctx.lineWidth = 1.5; ctx.beginPath();
    rows.forEach((r, i) => { const px = x(r.at), py = y(r[key]);
      if (i) { ctx.lineTo(px, y(rows[i - 1][key])); ctx.lineTo(px, py); } else ctx.moveTo(px, py); });
    ctx.lineTo(width - pad.right, y(rows[rows.length - 1][key])); ctx.stroke(); ctx.setLineDash([]);
  };
  line("h", css("--muted"), [4, 3]);
  line("tau", css("--accent"), []);
  ctx.fillStyle = css("--critical");
  rows.filter((r) => r.cause_type === "health").forEach((r) => ctx.fillRect(x(r.at) - 1, pad.top, 2, 6));
  ctx.fillStyle = css("--text"); ctx.fillText("τ —   h - -   ▮ подія справності", pad.left + 4, 11);
}
