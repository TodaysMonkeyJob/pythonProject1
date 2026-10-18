# Control-room dashboard (F13 / TOD-24)

`python main.py` → `http://127.0.0.1:8000` (on the LAN with `DASHBOARD_USER` / `DASHBOARD_PASS` in `.env`).
Plain ES modules in `static/app/` — no build step, no CDN, works without internet.

```
 [ Overview ]  ⇄  [ Warehouse ]  ⇄  [ Office ]  ⇄  [ Entrance ]  ⇄ …      ← / →, swipe, ◀ ▶, dots
                        │  ↓ / scroll / ▼ Деталі
                 [ Warehouse — details ]
```

| Screen | What it shows |
| -- | -- |
| **Overview** `#/overview` | every camera (grid 1×1 … 3×3, then pages; border = location state, grey = offline), location cards sorted danger → attention → safe → offline (click → that location), latest notifications with Acknowledge |
| **Location** `#/loc/<id>` | one-screen summary: state + reason + Acknowledge, its cameras (click a tile → full size), people now (✎ rename inline), last 5 transcripts (colour per person), gas + 10-min sparkline, last GPT decision |
| **Details** `#/loc/<id>/details` | transcript log (speaker / time filter, search, CSV), event table (filters, click → JSON), GPT history with the snapshot sent, gas chart 1 h with threshold, 24 h state timeline, device health (FPS, last seen, restarts) |
| **People / Devices / Locations** | F14: "Можливі дублікати" (pair cards: score, evidence, Об'єднати / Це різні люди, Перевірити зараз; top-bar counter), merge history with Скасувати; rename / merge / delete people; locations / on-off / phone pairing / RTSP / ONVIF + system status; create / rename / reorder / rules |

Screenshots: _overview.png_, _location.png_, _details.png_ (placeholders — take them on your setup).

## Keyboard

| Key | Action |
| -- | -- |
| ← / → | previous / next page of the carousel (Overview, then locations in their order) |
| ↓ / ↑ | location details / back to the summary |
| click a location card / camera tile | jump to that location (a tile on a location page: enlarge / back) |
| browser back / forward | previous pages (every page has its own `#/…` address) |

## Video

One WebSocket per page (`/ws/video`) carries every visible camera: small 320 px frames at `dashboard.grid_fps` (4),
the enlarged camera at full size at `dashboard.focus_fps` (10). Browsers allow only ~6 HTTP connections per host, so
one MJPEG stream per tile would stall beyond ~5 cameras; measured here: 8 cameras, each a steady ~4.6 FPS. Small frames
are made once per camera frame and shared by all open pages.

Browser notifications for `danger` are asked for once (on the first click). Theme: ◐ in the top bar.
