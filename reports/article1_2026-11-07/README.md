# Article 1 evaluation — autonomous hierarchical fuzzy system (2026-11-07)

`python -m experiments.hfs.run --seed 42` at commit `ef9919a`: 48 scenarios (`experiments/hfs/scenarios.yaml`) × 6 configurations, replayed through the real `HierarchicalFuzzySystem` (60 s of events per scenario, seeded noise).

**Labels: draft — author's expert labels, to be confirmed by the supervisor.** They were written from the requirements before running; the numbers below measure agreement with these draft labels, not ground truth.

## Overall level vs the expert labels

| Configuration | Macro-F1 | Accuracy | Brier | ECE | Corridor width (mean) | Corridor reaches the label |
| -- | -- | -- | -- | -- | -- | -- |
| `tri_type1` | 0.978 | 97.9% | 0.062 | 0.115 | 0.000 | 97.9% |
| `tri_it2` (deployed) | 0.960 | 95.8% | 0.100 | 0.094 | 0.114 | 95.8% |
| `gauss_in_type1` | 0.166 | 25.0% | 1.190 | 0.571 | 0.000 | 25.0% |
| `gauss_in_it2` | 0.469 | 56.2% | 0.767 | 0.295 | 0.141 | 68.8% |
| `gauss_all_type1` | 0.176 | 25.0% | 1.053 | 0.477 | 0.000 | 22.9% |
| `tri_it2_no_escalation` | 0.902 | 89.6% | 0.198 | 0.032 | 0.112 | 89.6% |

Macro-F1 by group (deployed configuration): audio 1.000, camera 0.936, combined 0.867, gas 1.000.

### Disagreements of the deployed configuration with the labels

| Scenario | Label | System | Why the label (scenarios.yaml) |
| -- | -- | -- | -- |
| c03 | low | medium | a visitor with a known person during working hours |
| x02 | high | medium | two medium channels at once |

Confusion (rows = label, columns = system):

| label \ system | low | medium | high | critical |
| -- | -- | -- | -- | -- |
| low | 10 | 1 | 0 | 0 |
| medium | 0 | 12 | 0 | 0 |
| high | 0 | 1 | 13 | 0 |
| critical | 0 | 0 | 0 | 11 |

### Reading the results

* **Agreement is high by construction.** The labels and the rule tables come from the same requirements (the supervisor's examples), so the deployed system agrees with 96% of them; the disagreements are the informative part — e.g. a visitor with a known person (1 of 2 unknown = 0.5) lies between the terms part and most and reads medium. The labels need an independent expert pass.
* **Gaussian membership functions break the low end.** A Gaussian never reaches 0, so in a quiet scene every rule still fires a little and the centroid is pulled up: only 0 of 11 low scenarios stay low (risk up to 0.62), and the critical ones are compressed (0.80–0.86 instead of 0.87). The triangular terms on a Ruspini partition are 0 outside their support — that is what keeps the table cells separate and the surface monotonic (table below).
* **The corridor is widest when somebody speaks**: mean width 0.26 with speech vs 0.07 without — the danger-word accuracy (δ = 0.465, 23 noisy synthetic samples) is the least measured input; recording real danger words narrows it most.

## Type-1 vs interval type-2

The crisp risk is the same (δ = 0 reproduces type-1 exactly); type-2 adds the corridor [y_l, y_r] from the measured input errors. Corridor width vs the errors (δ × scale, deployed rule bases):

| δ scale | Mean corridor width | Corridor reaches the label | Macro-F1 |
| -- | -- | -- | -- |
| ×0 | 0.000 | 97.9% | 0.978 |
| ×0.5 | 0.062 | 97.9% | 0.978 |
| ×1 | 0.114 | 95.8% | 0.960 |
| ×2 | 0.184 | 95.8% | 0.960 |

## Triangular vs Gaussian membership functions (slide 8, item 1)

Monotonicity measure = share of grid steps where the risk decreases while a risk-raising input increases (every input of these rule bases raises the risk), over a grid of all other inputs.

| Level | Triangular | Gaussian inputs | Gaussian inputs + risk terms |
| -- | -- | -- | -- |
| gas | 0 / 32400 (0.0%), max fall 0.0000, 0 level drops | 4414 / 32400 (13.6%), max fall 0.0458, 48 level drops | 4764 / 32400 (14.7%), max fall 0.0437, 117 level drops |
| camera | 0 / 3630 (0.0%), max fall 0.0000, 0 level drops | 162 / 3630 (4.5%), max fall 0.0462, 11 level drops | 172 / 3630 (4.7%), max fall 0.0429, 26 level drops |
| audio | 0 / 8232 (0.0%), max fall 0.0000, 0 level drops | 615 / 8232 (7.5%), max fall 0.0185, 1 level drops | 702 / 8232 (8.5%), max fall 0.0201, 1 level drops |
| top | 0 / 3630 (0.0%), max fall 0.0000, 0 level drops | 365 / 3630 (10.1%), max fall 0.0180, 0 level drops | 369 / 3630 (10.2%), max fall 0.0180, 1 level drops |

Surfaces: `surfaces/<level>_<triangular|gaussian_inputs|gaussian_all>_<x>__<y>.png` (the other inputs at the middle of their universe).

## With vs without multi-gas escalation (TOD-50)

Without it the highest gas alone sets the gas risk (no second / third gas, no group index). Macro-F1 0.960 with, 0.902 without. Scenarios that change: g09 (high → medium, label high), g14 (high → medium, label high), g15 (critical → high, label critical).

## Latency (this machine — not the Raspberry Pi)

| Level | Type-1 mean / p95, ms | IT2 mean / p95, ms |
| -- | -- | -- |
| gas | 0.663 / 0.742 | 1.546 / 3.008 |
| camera | 0.266 / 0.301 | 0.443 / 0.528 |
| audio | 0.262 / 0.283 | 0.737 / 1.17 |
| top | 0.223 / 0.262 | 0.326 / 0.445 |

**The Raspberry Pi has not been measured** (no Pi was available): run `python -m experiments.hfs.run --no-surfaces`, `python -m tools.camera_bench` and `python -m tools.audio_bench` on it.

## Camera: light vs actions mode, false motion (reports/camera_bench/)

Actions process: 51 MB in light mode vs 396 MB with the pose / action models (16.3 ms pose per frame), back to 51 MB after the planned restart. The camera mode does not enter the fuzzy rules (load only).
False motion rate, original → adaptive detector (synthetic clips): lighting 4.2% → 0.0%; foliage 1.8% → 0.4%; noise 0.0% → 0.0%; walk 33.3% → 9.0%.

## Per scenario

| Scenario | Label | tri_type1 | tri_it2 | gauss_in_type1 | gauss_in_it2 | gauss_all_type1 | tri_it2_no_escalation |
| -- | -- | -- | -- | -- | -- | -- | -- |
| g01 | low | low 0.13 | low 0.13 | high 0.58 ✗ | medium 0.35 [0.30–0.35] ✗ | high 0.58 ✗ | low 0.13 |
| g02 | medium | medium 0.35 | medium 0.35 [0.30–0.41] | high 0.73 ✗ | high 0.54 [0.42–0.62] ✗ | high 0.72 ✗ | medium 0.35 [0.30–0.41] |
| g03 | high | high 0.65 | high 0.65 [0.52–0.74] | critical 0.85 ✗ | high 0.73 [0.71–0.79] | critical 0.82 ✗ | high 0.65 [0.52–0.74] |
| g04 | critical | critical 0.87 | critical 0.87 [0.65–0.87] | critical 0.86 | critical 0.80 [0.78–0.84] | critical 0.83 | critical 0.87 [0.72–0.87] |
| g05 | critical | critical 0.87 | critical 0.87 | critical 0.86 | critical 0.81 | critical 0.83 | critical 0.87 |
| g06 | critical | critical 0.87 | critical 0.87 [0.65–0.87] | critical 0.86 | critical 0.80 [0.78–0.84] | critical 0.83 | critical 0.87 [0.65–0.87] |
| g07 | medium | medium 0.31 | medium 0.31 [0.21–0.41] | high 0.69 ✗ | medium 0.49 [0.35–0.64] | high 0.69 ✗ | medium 0.31 [0.21–0.41] |
| g08 | high | high 0.65 | high 0.65 [0.65–0.65] | critical 0.85 ✗ | critical 0.76 [0.76–0.79] ✗ | critical 0.83 ✗ | high 0.65 |
| g09 | high | high 0.65 | high 0.65 [0.65–0.65] | critical 0.84 ✗ | high 0.69 [0.69–0.70] | critical 0.82 ✗ | medium 0.35 ✗ |
| g10 | high | high 0.73 | high 0.73 [0.69–0.76] | critical 0.85 ✗ | high 0.74 [0.71–0.80] | critical 0.83 ✗ | high 0.73 [0.69–0.76] |
| g11 | medium | medium 0.31 | medium 0.31 [0.25–0.35] | high 0.71 ✗ | high 0.51 [0.37–0.63] ✗ | high 0.70 ✗ | medium 0.31 [0.25–0.35] |
| g12 | medium | medium 0.34 | medium 0.34 [0.29–0.39] | high 0.72 ✗ | high 0.53 [0.41–0.62] ✗ | high 0.72 ✗ | medium 0.34 [0.29–0.39] |
| g13 | critical | critical 0.87 | critical 0.87 [0.65–0.87] | critical 0.86 | critical 0.80 [0.69–0.84] | critical 0.83 | critical 0.87 [0.65–0.87] |
| g14 | high | high 0.65 | high 0.65 | critical 0.85 ✗ | high 0.73 [0.71–0.74] | critical 0.82 ✗ | medium 0.46 ✗ |
| g15 | critical | critical 0.85 | critical 0.85 [0.74–0.87] | critical 0.86 | critical 0.78 [0.74–0.82] | critical 0.83 | high 0.60 [0.50–0.67] ✗ |
| c01 | low | low 0.13 | low 0.13 | high 0.57 ✗ | medium 0.34 ✗ | high 0.58 ✗ | low 0.13 |
| c02 | low | low 0.13 | low 0.13 | high 0.57 ✗ | high 0.55 [0.55–0.57] ✗ | high 0.58 ✗ | low 0.13 |
| c03 | low | medium 0.28 ✗ | medium 0.28 [0.26–0.31] ✗ | high 0.62 ✗ | high 0.60 [0.60–0.61] ✗ | high 0.62 ✗ | medium 0.28 [0.26–0.31] ✗ |
| c04 | medium | medium 0.35 | medium 0.35 | critical 0.78 ✗ | high 0.72 [0.68–0.76] ✗ | high 0.76 ✗ | medium 0.35 |
| c05 | high | high 0.65 | high 0.65 | critical 0.84 ✗ | high 0.74 [0.73–0.78] | critical 0.81 ✗ | high 0.65 |
| c06 | critical | critical 0.87 | critical 0.87 | critical 0.86 | critical 0.80 [0.80–0.83] | critical 0.83 | critical 0.87 |
| c07 | high | high 0.65 | high 0.65 | critical 0.84 ✗ | high 0.75 [0.75–0.77] | critical 0.82 ✗ | high 0.65 |
| c08 | medium | medium 0.35 | medium 0.35 | high 0.57 ✗ | high 0.56 [0.56–0.57] ✗ | high 0.58 ✗ | medium 0.35 |
| c09 | low | low 0.13 | low 0.13 | high 0.57 ✗ | high 0.56 [0.56–0.57] ✗ | high 0.58 ✗ | low 0.13 |
| c10 | high | high 0.65 | high 0.65 | critical 0.84 ✗ | high 0.74 [0.74–0.77] | critical 0.82 ✗ | high 0.65 |
| c11 | medium | medium 0.27 | medium 0.27 [0.26–0.30] | high 0.62 ✗ | high 0.60 [0.60–0.61] ✗ | high 0.62 ✗ | medium 0.27 [0.26–0.30] |
| c12 | low | low 0.13 | low 0.13 | high 0.57 ✗ | high 0.56 [0.56–0.58] ✗ | high 0.58 ✗ | low 0.13 |
| a01 | low | low 0.13 | low 0.13 | high 0.57 ✗ | medium 0.34 ✗ | high 0.58 ✗ | low 0.13 |
| a02 | low | low 0.13 | low 0.13 | high 0.57 ✗ | medium 0.34 ✗ | high 0.58 ✗ | low 0.13 |
| a03 | medium | medium 0.35 | medium 0.35 [0.13–0.65] | high 0.58 ✗ | high 0.56 [0.36–0.85] ✗ | high 0.58 ✗ | medium 0.35 [0.13–0.65] |
| a04 | critical | critical 0.87 | critical 0.87 [0.35–0.87] | critical 0.86 | critical 0.81 [0.35–0.87] | critical 0.83 | critical 0.87 [0.35–0.87] |
| a05 | high | high 0.65 | high 0.65 [0.35–0.87] | critical 0.85 ✗ | high 0.71 [0.13–0.87] | critical 0.82 ✗ | high 0.65 [0.35–0.87] |
| a06 | medium | medium 0.35 | medium 0.35 [0.13–0.87] | high 0.59 ✗ | medium 0.35 [0.13–0.87] | high 0.59 ✗ | medium 0.35 [0.13–0.87] |
| a07 | high | high 0.65 | high 0.65 | critical 0.85 ✗ | high 0.74 [0.74–0.77] | critical 0.82 ✗ | high 0.65 |
| a08 | medium | medium 0.35 | medium 0.35 | high 0.62 ✗ | high 0.61 [0.59–0.64] ✗ | high 0.62 ✗ | medium 0.35 |
| a09 | low | low 0.13 | low 0.13 | high 0.57 ✗ | medium 0.34 ✗ | high 0.58 ✗ | low 0.13 |
| a10 | medium | medium 0.27 | medium 0.27 [0.13–0.65] | high 0.58 ✗ | medium 0.35 [0.14–0.83] | high 0.59 ✗ | medium 0.27 [0.13–0.65] |
| a11 | high | high 0.65 | high 0.65 | critical 0.85 ✗ | high 0.74 [0.74–0.77] | critical 0.82 ✗ | high 0.65 |
| a12 | low | low 0.13 | low 0.13 | high 0.57 ✗ | medium 0.34 ✗ | high 0.58 ✗ | low 0.13 |
| x01 | critical | critical 0.87 | critical 0.87 [0.77–0.87] | critical 0.86 | critical 0.85 [0.84–0.87] | critical 0.83 | critical 0.87 [0.77–0.87] |
| x02 | high | high 0.65 | medium 0.35 [0.29–0.40] ✗ | high 0.74 | high 0.69 [0.43–0.86] | high 0.72 | medium 0.35 [0.29–0.40] ✗ |
| x03 | critical | critical 0.87 | critical 0.87 | critical 0.86 | critical 0.80 [0.80–0.83] | critical 0.83 | critical 0.87 |
| x04 | low | low 0.13 | low 0.13 | high 0.57 ✗ | high 0.55 [0.55–0.57] ✗ | high 0.58 ✗ | low 0.13 |
| x05 | high | high 0.65 | high 0.65 [0.53–0.73] | critical 0.86 ✗ | high 0.76 [0.67–0.87] | critical 0.83 ✗ | high 0.65 [0.53–0.73] |
| x06 | critical | critical 0.87 | critical 0.87 [0.65–0.87] | critical 0.86 | critical 0.86 [0.35–0.87] | critical 0.83 | critical 0.87 [0.65–0.87] |
| x07 | medium | medium 0.35 | medium 0.35 [0.30–0.41] | high 0.68 ✗ | high 0.54 [0.42–0.63] ✗ | high 0.68 ✗ | medium 0.35 [0.30–0.41] |
| x08 | high | high 0.65 | high 0.65 [0.58–0.65] | critical 0.79 ✗ | high 0.76 [0.67–0.83] | high 0.76 | high 0.65 [0.58–0.65] |
| x09 | critical | critical 0.87 | critical 0.87 [0.65–0.87] | critical 0.80 | critical 0.80 [0.79–0.84] | high 0.75 ✗ | critical 0.87 [0.73–0.87] |

Files: metrics.json (everything above), scenarios.csv, variants/ (the rule bases of every configuration), surfaces/.
