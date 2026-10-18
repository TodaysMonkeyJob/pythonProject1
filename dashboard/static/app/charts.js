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
