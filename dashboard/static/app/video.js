// One WebSocket per page for every camera tile (browsers allow only ~6 HTTP/1.1 connections per host, so one MJPEG
// stream per tile stalls beyond ~5 cameras). The server sends [4-byte header length][header JSON][JPEG]; the page
// subscribes only to the tiles that are in the DOM right now, and the enlarged one at full size.

let ws = null, wanted = [], focus = null, delay = 1000;
const urls = new Map();          // `${device}|${size}` → object URL currently shown (revoked when replaced)
const lastFrame = new Map();     // device → time of its last frame (tiles show "no signal" when stale)

function send() {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ subscribe: wanted, focus }));
}

function onFrame(data) {
  const view = new DataView(data), size = view.getUint32(0);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, size)));
  const url = URL.createObjectURL(new Blob([new Uint8Array(data, 4 + size)], { type: "image/jpeg" }));
  const key = `${header.device_id}|${header.size}`;
  const previous = urls.get(key);
  urls.set(key, url);
  lastFrame.set(header.device_id, Date.now());
  document.querySelectorAll(`img[data-video="${CSS.escape(header.device_id)}"]`).forEach((img) => {
    if ((img.dataset.size || "small") === header.size || (header.size === "full" && img.dataset.size === "small"
        && !urls.has(`${header.device_id}|small`))) img.src = url;
    img.closest(".tile")?.classList.remove("stale");
  });
  if (previous) setTimeout(() => URL.revokeObjectURL(previous), 1000);   // after the <img> switched
}

export function connectVideo() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/video");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { delay = 1000; send(); };
  ws.onmessage = (m) => { if (m.data instanceof ArrayBuffer) onFrame(m.data); };
  ws.onclose = () => { setTimeout(connectVideo, delay); delay = Math.min(delay * 2, 15000); };
}

// Called after every render: subscribes to exactly the tiles on screen
export function syncSubscriptions() {
  const tiles = [...document.querySelectorAll("img[data-video]")];
  const next = [...new Set(tiles.map((img) => img.dataset.video))];
  const big = tiles.find((img) => img.dataset.size === "full");
  const nextFocus = big ? big.dataset.video : null;
  tiles.forEach((img) => {            // a re-rendered tile shows the last frame at once
    const url = urls.get(`${img.dataset.video}|${img.dataset.size || "small"}`) || urls.get(`${img.dataset.video}|small`);
    if (url && !img.src) img.src = url;
  });
  if (next.join() !== wanted.join() || nextFocus !== focus) { wanted = next; focus = nextFocus; send(); }
}

export const frameAge = (device) => (lastFrame.has(device) ? Date.now() - lastFrame.get(device) : Infinity);
