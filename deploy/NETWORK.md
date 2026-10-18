# Device network (F11 / TOD-22)

A **hub** (the MacBook now, a Raspberry Pi later) runs the event bus, identity registry, aggregator, GPT client and
dashboard. **Nodes** produce data. Nothing extra has to be installed: no MQTT broker, no mkcert, no ffmpeg — only
Python, the packages in `requirements.txt` and the system `openssl` (part of macOS and Raspberry Pi OS).

| Node | Example | How it connects | Processing |
| -- | -- | -- | -- |
| `local` | Mac built-in camera / mic, USB camera | found automatically (`core/discovery.py`) | hub |
| `continuity` | iPhone as a Mac camera (near the Mac) | appears like a local camera | hub |
| `browser` | any phone / laptop on the LAN | `https://<hub-ip>:8443/node` + pairing code | hub |
| `rtsp` | IP camera | RTSP URL on the Devices page (ONVIF scan helps to find it) | hub |
| `edge` | Raspberry Pi with camera / mic / gas | `python -m node_agent.agent` | on the Pi, hub aggregates |

## Hub

```bash
python main.py            # finds cameras + microphones, one pipeline per device; dashboard on :8000, nodes on :8443
```

* **Devices page** (dashboard): every device with status, location (pick one or create “Room B”), on / off.
  New devices start as `unassigned`. Unplugging a USB camera marks it offline within ~15 s; plugging it back starts it
  again. Location seeds for first-time devices: `network.seed` in `config.yaml`.
* Several cameras run in parallel: one capture + motion + YOLO process per camera (YOLO only on motion), pose and face
  recognition pooled in one process each with fair round-robin scheduling (idle cameras: 1 FPS). Every module process
  gets small CPU thread pools (`orchestrator.child_threads`, `vision.threads`): measured on an M2 Pro, 4 cameras ran at
  ~14 FPS each with YOLO 35–45 ms per frame (570 ms without the limits).
* A microphone wakes up only when a camera in **its location** sees someone (`scene_active` is per location).
* Every event carries `device_id`, `node_id`, `location_id` (schema v2; v1 logs stay readable).

## Phone in another room (browser node)

1. Dashboard → Devices → **Підключити телефон**: a 6-digit code (valid 10 min) and the address
   `https://<hub-ip>:8443/node`.
2. Open the address on the phone (same Wi-Fi). The certificate is self-signed (made by the hub with `openssl`, stored
   in `data/tls/`): confirm the browser warning once.
3. Enter the code and a name, choose the location, tap **Почати трансляцію**. The phone streams JPEG frames
   (2–10 FPS) and 16 kHz audio over WSS; the screen is kept on (Screen Wake Lock). Closing the page marks its camera
   and microphone offline.

Lost phone: `DELETE /api/nodes/<node_id>` (or remove it from `data/devices.db`) revokes its token.

## Raspberry Pi (edge node)

```bash
python -m node_agent.agent --pair 123456 --name "Pi kitchen"   # code from the hub dashboard
python -m node_agent.agent                                     # run (put it into a systemd unit)
```

The agent finds the hub by its UDP announcement on the LAN (port 47800; or `--hub https://<hub-ip>:8443`), pins the
hub's certificate, stores the token in `data/node_agent/` (git-ignored) and runs the full pipeline locally (cameras,
microphones, gas sensor, actions). Events (with the Pi's timestamp; the hub adds `received_at`) and a camera preview
every 2 s go to the hub over WSS; offline it buffers events and reconnects. Location changes made on the hub dashboard
are sent back to the Pi. Face identification uses the hub's registry and is off on edge nodes. Keep the Pi's clock in
sync (NTP is on by default in Raspberry Pi OS).

## IP cameras

Devices → **IP-камера (RTSP)**: e.g. `rtsp://user:pass@192.168.0.50:554/stream1` (OpenCV's bundled FFmpeg opens it).
**Пошук ONVIF** sends one WS-Discovery multicast probe and lists cameras that answer (name, address); the RTSP URL is
in the camera's manual / app. The URL (with credentials) stays in the local `data/devices.db`.

## Security notes

* The dashboard on `:8000` without `DASHBOARD_USER` / `DASHBOARD_PASS` listens on 127.0.0.1 only; on `:8443` (LAN) the
  dashboard pages always require those credentials. Node endpoints require the pairing token.
* Pairing codes are one-time, expire after 10 minutes, and wrong codes are slowed down; only token hashes are stored.
