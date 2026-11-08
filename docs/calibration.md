# Persistent sensor calibration — C1 / TOD-54

**Calibration is done once per device and stored.** On start the system loads the stored calibration and runs only a
short check. A full recalibration happens only in these cases:
- a new device or a changed device (new uid);
- repeated failed checks;
- the calibration is older than the maximum age for its type;
- the operator presses the button on the dashboard.

## What is stored

There is one file per device: `sensor_calibration.dir` → `data/calibration/devices/<uid>.json`. It is git-ignored and
survives restarts and updates.
- **uid** = driver + serial / hardware id + bus address + firmware (the TOD-48 descriptor, `core.sensor_calibration.device_uid`).
  Cameras and microphones use their stable registry `device_id` (kind + node + hardware key).
- **Record:** `{uid, kind, device_id, values, method, date, environment, validation (last check), by, history}`. The
  history holds up to 10 previous versions, newest first.
- **Writes are atomic** (tmp + rename).

| Device | `values` | Short check on start | Full calibration |
| -- | -- | -- | -- |
| Graphene FET (`graphene_fet`) | clean-air baseline curve R(V_G), Dirac point, normalisation peak R_peak | first sweep classified as clean air: R_peak within 25 %, Dirac point within 2 gate steps | mean of 30 clean-air sweeps (the user provides clean air) |
| MOS / MQ (`mos`) | R₀ (replaces the declared `r0_ohm`) | after the heater warm-up (180 s): R_s / R₀ within 50 % of the clean-air ratio | mean R_s of 60 clean-air readings ÷ the datasheet clean-air ratio, after the warm-up |
| Electrochemical (`electrochemical`) | — (factory calibrated) | stabilisation 60 s → «заводська калібровка» | — |
| Particle (`particle`) | — (factory calibrated) | fan spin-up 30 s | — |
| ENS160 (`index_only`) | — | start-up 180 s | — |
| Microphone (`mic`) | background level (dB) + noise power spectrum | 3 s of non-speech audio: median level within ±6 dB | 60 s of non-speech background → 20th percentile + spectrum |
| Camera (`camera`) | motion background image (160 px, PNG) + per-cell noise levels + jitter | 15 frames: median difference from the stored background ≤ 20 grey levels | 30 s of frames without motion → the learned MOG2 background + noise levels |

The thresholds are in `config.yaml` `sensor_calibration.kinds` (`max_age_days`, `max_failures`, `recheck_sec`,
`warmup_sec`, tolerances).

## States (`core/sensor_calibration.py` `DeviceCalibration`)

| State | Text (dashboard) | Meaning |
| -- | -- | -- |
| `ready` | «відкалібровано (збережено, дата)» / «(нова калібровка, дата)» | checked; full protection |
| `factory` | «заводська калібровка» | nothing to calibrate |
| `warming_up` | «прогрів, N хв» | the stored values are used, δ × `warmup_uncertainty_factor` (2); this is not «не відкалібровано» |
| `checking` | «коротка перевірка (збережено, дата)» | the stored values are used, δ × 2 until the check passes |
| `needs_check` | «потребує перевірки» | the check failed: δ × 2. The check is repeated after `recheck_sec`, and recalibration starts only after `max_failures` (3) failures in a row |
| `calibrating` | «калібрування… N %» | the full calibration; the stored values (if any) stay in use until it ends; can be cancelled |
| `uncalibrated` | «не відкалібровано» | no record, calibration cancelled |

How the status gets out:
- **Events.** Every change is a `calibration_status` event (logged, repeated every `refresh_sec`). It carries the
  source (stored / factory / new), the date and age, the last check, the failures, the progress, the instructions,
  `uncertainty_factor`, and `ready_after_sec` (the time from start to usable).
- **HFS.** The HFS takes `uncertainty_factor` per device and widens δ of its subsystem in that location
  (`calibration_factor` in `hfs_risk.subsystems`).

## Dashboard

The Devices page has a section «Калібрування» (`views/calibration.js`):
- **Per device:** status, source, date and age, ready time, last check.
- **«Перекалібрувати»** per device and per location. It asks for confirmation with instructions (e.g. «забезпечте
  чисте повітря» for the gas baseline), shows progress and has «Скасувати».
- **«Історія»** lists the previous versions with **«Відновити»**.

These actions are admin only (TOD-52 roles).

API:
- `GET /api/calibration`;
- `POST /api/calibration/recalibrate {device_id | location_id, confirm: true}`;
- `POST /api/calibration/cancel {device_id | location_id}`;
- `POST /api/calibration/restore {uid, index, confirm: true}`.

The API publishes `calibration_control {action, device_ids, by}`, which the orchestrator forwards to the gas,
camera and microphone processes. For a restore, the dashboard writes the file and the worker reloads it.

## Ready time

`python -m tools.calibration_restart` → `reports/calibration_restart/` measures the first start and a restart one
minute later for every kind. It uses the real calibrators with a simulated clock.

| Sensor | Restart | First start |
| -- | -- | -- |
| Microphone | ≈ 3 s, the short check | 60 s of calibration |
| Camera | ≈ 1.5 s | 30 s |
| Graphene | ≈ 1 s (1 sweep) | 30 s |
| MOS | 180 s, only the physical heater warm-up (with the stored R₀) | 180 + 60 s |

## Failures

- **Missing or corrupt file.** Only that device recalibrates. A corrupt file is moved to `.corrupt-<time>` and logged.
- **Too old.** A full recalibration runs while the old values stay in use.
- **Failed check.** «потребує перевірки», wider δ, and a repeat after `recheck_sec`. There is no silent
  recalibration on the first failure.
