# Ready time after a restart (C1 / TOD-54)

Simulated clock and synthetic signals through the real calibrators (`python -m tools.calibration_restart`); policy from `config.yaml` `sensor_calibration`. The live system reports the same per device as `ready_after_sec` (dashboard «Калібрування»).

| Sensor | First start: ready after, s | full calibration | Restart after 1 min: ready after, s | full calibration | restart status |
| -- | -- | -- | -- | -- | -- |
| mic | 60.03 | yes | 3.06 | no | відкалібровано (збережено, 2026-11-08) |
| camera | 30.1 | yes | 1.5 | no | відкалібровано (збережено, 2026-11-08) |
| graphene_fet | 30.0 | yes | 1.0 | no | відкалібровано (збережено, 2026-11-08) |
| mos | 239.0 | yes | 180.0 | no | відкалібровано (збережено, 2026-11-08) |
| electrochemical | 60.0 | no | 60.0 | no | заводська калібровка |
| particle | 30.0 | no | 30.0 | no | заводська калібровка |

All devices ready: **239.0 s** on the first start, **180.0 s** after a restart (devices start in parallel: the slowest one decides). Warm-up times (MOS heater, electrochemical stabilisation, particle fan) are physical and stay; during them the stored calibration is used with a wider uncertainty.
