# Control-room dashboard (F13 / TOD-24 → single screen F21 / TOD-32)

`python main.py` → `http://127.0.0.1:8000` (on the LAN with `DASHBOARD_USER` / `DASHBOARD_PASS` in `.env`).
Plain ES modules in `static/app/` — no build step, no CDN, works without internet.

**One screen, the page never scrolls on desktop (≥ 1280×720)** — only panels scroll inside themselves.

```
[logo] ◀ [Огляд] [● Склад] [● Офіс] [● Вхід] … ▶   (1 небезпека · 1 увага · 5/6 камер)  Люди Пристрої Локації 🔔
┌ Overview ──────────────────────────────┬───────────────────────┐   ┌ Territory ─────────────────┬──────────────────────────┐
│ all cameras (click → territory)        │ territory cards       │   │ name · БАДЖ · reason · Підтв│ Події Розмова Газ GPT    │
│ chip: territory + state, people count  │ danger→attention→     │   │ main camera (fills height)  │ (each tab scrolls inside)│
│ border = state (none when safe)        │ offline→safe          │   │ other cameras · people chips│ newest first, "N нових ↑"│
│                                        │ global log, newest ↑  │   └─────────────────────────────┴──────────────────────────┘
└────────────────────────────────────────┴───────────────────────┘
```

| Input | Action |
| -- | -- |
| ← / → or ◀ ▶ | previous / next territory (Огляд → territories → Огляд, wraps) |
| Home · 1–9 | Огляд · territory by index |
| click a camera tile / territory card | open that territory |
| `#/loc/<id>/<events|talk|gas|gpt>` | a territory with that tab (switching territory opens Події; scroll positions are kept per territory) |

* Lists are newest first (`ui.LiveList`): live items are prepended; when you have scrolled down (> 40 px) your rows
  stay put and a "N нових ↑" pill appears instead (`overflow-anchor: none`, no page `scrollTo`, no scroll-snap).
* Розмова shows the speaker-attributed transcript with the trust badge (F16); Газ the current level, trend and a
  10-minute bar chart against the threshold; Рішення GPT the decision, "зниження відхилено" and the F17 corridor bar
  (floor–ceil band, dashed LLM marker, white final marker), LLM / без GPT (F19).
* One `/ws/video` socket; only visible tiles are subscribed (overview: small; territory: the main camera full size).
* State colours always come with a text label; phones (< 900 px) stack the panels and may scroll.
* Fonts: "IBM Plex Sans / Mono" when installed, otherwise the system font (not vendored — no extra downloads).

## Довіра (F23 / TOD-34)

`#/trust` lists the trust registry grouped location → device → channel: τ, u, h (↗ = recovering), prior status
(measured / unmeasured), last change and the Ukrainian explanation; "лише деградовані" keeps channels with τ < 0.4,
h < 1 or a failing check. A row opens its τ / h history (health changes marked) with ✓ / ✗ operator feedback.
Camera tiles show the weakest channel's τ and get a red frame when degraded. Event dialogs carry "✓ Вірно / ✗ Хибно"
for the event's channel and a transcript correction box for speech (`POST /api/trust/feedback` → `trust_feedback`
event → the router's registry; the dashboard only reads `trust.registry_db`).

