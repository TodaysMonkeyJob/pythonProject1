# Gas and particle sensors (A2 / TOD-42)

The gas subsystem compares every substance as a share of its critical concentration, c_norm = c / C_crit
(`config/gases.yaml`). Sensors and their models keep their native units (vol %, ppm, µg/m³). Detecting the sensors
and building the per-sensor inputs is TOD-48. This page records what the candidate sensors measure, and how
accurately, relative to C_crit. The accuracy column feeds the FOU of TOD-45.

## Critical concentrations

| Substance | C_crit | Source | Conversion |
| -- | -- | -- | -- |
| Ammonia NH₃ | 300 ppm (= 0.03 vol %, ≈ 209 mg/m³) | NIOSH IDLH | 1 ppm ≈ 0.70 mg/m³ (M = 17.03) |
| Carbon monoxide CO | 1200 ppm (≈ 1375 mg/m³) | NIOSH IDLH | 1 ppm ≈ 1.15 mg/m³ (M = 28.01) |
| PM2.5 (smoke) | 225.5 µg/m³ | US EPA AQI 2024: start of "Hazardous" | mass concentration, no ppm |
| Ethanol | 3300 ppm (= 0.33 vol %) | NIOSH Pocket Guide (10 % LEL) | M = 46.07 |
| Acetone | 2500 ppm (= 0.25 vol %) | NIOSH Pocket Guide (10 % LEL) | M = 58.08 |
| Toluene | 500 ppm (= 0.05 vol %) | NIOSH Pocket Guide | M = 92.14 |

mg/m³ = ppm · M / 24.45 (25 °C, 1 atm); 1 vol % = 10 000 ppm. PM2.5 is not a gas: it is the mass of particles
≤ 2.5 µm per m³. It has no lethal concentration, so the start of the EPA "Hazardous" category counts as 100 %.

## Sensor survey

Values were checked on 2026-11-06. The "checked against" column says which values come from the manufacturer's
datasheet or wiki and which only from secondary listings. Confirm the latter before using them as FOU input.

| Substance | Sensor | Type | Interface | Range | Resolution / accuracy | Response | Range vs C_crit | Checked against |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| NH₃, ethanol, acetone, toluene | our graphene FET (rGO / porous Si) | FET, R(V_G) curve + trained classifier (TOD-19) | own board | dataset 0.25–14 vol % (NH₃ 0.3–10.8 %) | from the regression model (`reports/gas_metrics.json`) | poll 1 s | the whole range is **above** C_crit (see below) | our data |
| NH₃ | DFRobot Gravity SEN0469 | electrochemical, factory calibrated | I2C / UART | 0–100 ppm | 1 ppm; ±10 % reading or ±5 % FS (greater) | T90 ≤ 150 s | **ends at 0.33 C_crit**: a full-scale reading is an open upper bound | DFRobot wiki (range, resolution, response); accuracy from a DFRobot forum reply |
| CO | DFRobot Gravity SEN0466 | electrochemical, factory calibrated | I2C / UART | 0–1000 ppm | 1 ppm | T90 ≤ 30 s | **ends at 0.83 C_crit** | DFRobot wiki |
| NH₃ | Winsen MQ-137 | MOS, analog (ADS1115 ADC on I2C) | analog | 5–500 ppm | approximate (R_s/R_0 datasheet curve, needs own calibration) | — | covers C_crit (1.67 ×) | Winsen product page |
| CO | Winsen MQ-7 | MOS, analog | analog | — | approximate | — | — | **not checked** (no primary source found) |
| PM2.5 | Plantower PMS5003 | laser scattering | UART | — | — | — | — | **not checked** (manual not reachable) |
| PM2.5 | Sensirion SPS30 | laser scattering | I2C / UART | 0–1000 µg/m³ | ±10 µg/m³ (0–100), ±10 % (100–1000) | 1 s | covers C_crit (4.4 ×) | ±10 µg/m³ from a Sensirion datasheet copy; range from a secondary table |
| PM2.5 | Nova Fitness SDS011 | laser scattering | UART / USB | 0–999.9 µg/m³ | 0.3 µg/m³; max of ±15 % and ±10 µg/m³ | 1 s | covers C_crit (4.4 ×) | a copy of the Nova datasheet |

### Graphene FET dataset vs C_crit

The graphene sensor dataset (`data/gas/data.xls`) is in vol %. Its smallest non-zero concentrations are 0.25–0.4 %
(2500–4000 ppm) and its largest 9–14 %. C_crit is 0.03 % for NH₃, 0.05 % for toluene, 0.25 % for acetone and
0.33 % for ethanol, so every non-zero reading in the dataset is at or above C_crit: NH₃ at 0.3 % is 10 × its
IDLH. With this sensor the gas subsystem can therefore only tell clean air from critical. The deployed graphene
sensor measures **0.25–14 vol %**; a sensor or calibration below 0.03 vol % (300 ppm) is needed before the
"moderate" and "dangerous" terms of NH₃ can be reached. This is why the electrochemical modules are listed above.
Their own problem is the opposite one: their range ends below C_crit (TOD-48 flags it).

## Sampling and inference periods

| What | Period | Where | Why |
| -- | -- | -- | -- |
| Graphene board poll | 1 s | `gas.gas_poll_sec` | one R(V_G) curve per second |
| Electrochemical / particle modules | ~1 s (module output rate) | sensor driver (TOD-48) | the modules update about once a second |
| Gas subsystem inference | 2 s | `config/fuzzy/gas.yaml` `update_sec` | much faster than any dangerous change, low CPU on the Pi |
| Dynamics window | 60 s | `hfs.gas.window_sec` | longer than the T90 response of the electrochemical sensors (30 s CO, ≤ 150 s NH₃), so the slope follows the gas, not sensor noise; ≥ 30 samples at 1 Hz for the least-squares slope |
| Stale data | 30 s | `hfs.gas.stale_sec` | no reading for 30 s → the subsystem reports no data and the overall risk is partial |

The `dynamics` universe is ±0.5 C_crit per minute, i.e. from clean air to the critical level in 2 minutes. `rising`
peaks at 0.1 / min (critical in about 10 minutes) and `sharp_rise` is full from 0.25 / min (critical within about
4 minutes).
