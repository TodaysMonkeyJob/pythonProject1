# Camera benchmarks (A3 / TOD-43)

## Light vs actions mode (actions process)

| | Light mode | Actions mode | Unloaded in place | After the planned restart |
| -- | -- | -- | -- | -- |
| Resident memory, MB | 50 | 392 | 571 | 50 (a fresh light process) |
| Pose per frame, ms (mean / p95) | 0 (skipped) | 16.1 / 18.0 | 0 | 0 |

Unloading in place does not give the memory back (torch / ultralytics stay imported, the allocator keeps its pages), so in the orchestrator the actions process turns actions off by exiting with `PLANNED_RESTART_EXIT` and is restarted at once in light mode.

Models load in 1.64 s; imgsz 320, 2 threads, 3 persons in the test frame (`bus.jpg`). The vision detector (YOLO + motion) runs the same in both modes.

## Motion detection

`python -m tools.camera_bench` — synthetic clips, 60 s at 10 fps (see the module docstring). Before = the original MOG2 detector, after = the A3 adaptive one.

| Clip | False motion rate before | after | Recall before | after | Crossings detected before | after |
| -- | -- | -- | -- | -- | -- | -- |
| lighting | 4.2% | 0.0% | — | — | — | — |
| foliage | 1.8% | 0.4% | — | — | — | — |
| noise | 0.0% | 0.0% | — | — | — | — |
| walk | 33.3% | 9.0% | 100.0% | 100.0% | 6 | 6 |

Crossings after the 30-frame warm-up only (at 1 fps the first 30 s learn the background).

## Frame rate (A3 detector, walk clip)

| fps | Recall | Crossings detected | Mean latency, s | False motion rate |
| -- | -- | -- | -- | -- |
| 1 | 88.9% | 3 / 3 | 1.00 | 0.0% |
| 2 | 96.9% | 4 / 4 | 0.38 | 5.9% |
| 3 | 100.0% | 5 / 5 | 0.33 | 5.3% |
| 5 | 100.0% | 5 / 5 | 0.16 | 10.5% |
| 10 | 100.0% | 6 / 6 | 0.12 | 9.0% |
| 15 | 100.0% | 6 / 6 | 0.07 | 10.3% |

Reading the tables: the walk clip combines everything (gusts + a light switch + a person); the remaining false frames of the A3 detector are small (≤ 3 % of the frame) foliage blobs right after the light-switch hold-off and during gusts — far below the old detector's full-frame reports. Walking is detected in every crossing from 1 fps, with 100 % recall from 3 fps (latency ≤ 0.33 s): **3 fps is the minimum for reliable walking detection**; the cameras run at 15.

Synthetic clips only — repeat on labelled real footage before quoting these numbers as field results.
