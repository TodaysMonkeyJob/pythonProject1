# Gas and particle sensors (A2 / TOD-42)

The gas subsystem compares every substance as a share of its critical concentration, c_norm = c / C_crit
(`config/gases.yaml`). Sensors and their models keep their native units (vol %, ppm, µg/m³). The sensors are found
and the per-sensor inputs are built automatically (TOD-48, [Drivers and discovery](#drivers-and-discovery-tod-48)). This page records what the candidate sensors measure, and how
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

| Gas | Smallest non-zero in `data.xls` | C_crit | Ratio |
| -- | -- | -- | -- |
| Ethanol | 0.4 vol % | 0.33 vol % (3300 ppm) | 1.2 × |
| Acetone | 0.4 vol % | 0.25 vol % (2500 ppm) | 1.6 × |
| Toluene | 0.25 vol % | 0.05 vol % (500 ppm) | 5 × |
| Ammonia | 0.3 vol % | 0.03 vol % (300 ppm) | 10 × |

The model has never seen a sub-critical concentration. It can identify the gas and tell clean air from "above
critical", but it cannot grade safe, low or medium levels. G1 (TOD-53) acts on this:

- **A non-zero graphene reading counts as at least the start of its training range**, which is ≥ C_crit for every
  gas. This is `modules.gas.data.MODEL_RANGE_VOL_PCT`, which is the driver's per-gas `range` and the worker's
  `detection_floor`.
- **Such a reading is critical.** The gas worker adds `below_model_range` and a Ukrainian `model_note`, and logs the
  note once per gas. The safety invariant (`modules/hfs/gas_safety.py`) makes the reading critical, and the text
  says «нижче діапазону сенсора».
- **Grading below C_crit needs other data:** sensors whose range covers it (electrochemical / MOS / PID, the drivers
  above), or new low-concentration measurements to retrain the model.
- **Scenarios** never feed sub-critical values to the graphene model. The scripted set (`experiments/scripted.py`)
  uses a low-range sensor in ppm (range 0–2 × C_crit).

Every gas value is shown in its sensor's unit together with «% від критичного рівня» (`pct_of_crit` in the readings,
on the dashboard and in the logs). Model card: [`modules/gas/MODEL_CARD.md`](../modules/gas/MODEL_CARD.md).

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

## Drivers and discovery (TOD-48)

With `gas.mode: drivers` the gas module finds the sensors connected to the Raspberry Pi at start-up, and again after
a driver choice on the dashboard. It then reads each one in its native unit. `python -m drivers.gas.discovery`
prints what is connected.

| Driver | Device | Bus | How it is identified | Native unit | Protocol source |
| -- | -- | -- | -- | -- | -- |
| `graphene_fet` | our graphene FET board | serial / USB, 115200 | handshake `ID?` → JSON descriptor | vol % (trained model) | the board protocol below |
| `dfrobot_gravity` / `_uart` | DFRobot Gravity NH₃ SEN0469, CO SEN0466, … | I2C 0x74–0x77 / UART 9600 | checksum-valid answer to command 0x86, gas type byte | ppm | DFRobot_MultiGasSensor library |
| `mq_ads1115` | Winsen MQ via an ADS1115 ADC | I2C 0x48–0x4B | **cannot be identified**: declared, or chosen on the dashboard | ppm (approximate, own calibration) | ADS1115 datasheet registers, the MQ datasheet curve |
| `pms5003` | Plantower PMS5003 | UART 9600 | frames `42 4D` with a valid sum | µg/m³ | Plantower frame layout |
| `sds011` | Nova Fitness SDS011 | UART / USB 9600 | frames `AA C0 … AB` with a valid sum | µg/m³ | Nova datasheet |
| `sps30` | Sensirion SPS30 | I2C 0x69 | product type `00080000` (CRC-8 checked) | µg/m³ | Sensirion embedded-sps driver |
| `ens160` | ScioSense ENS160 | I2C 0x52 / 0x53 | part ID 0x0160 | index (AQI-UBA 1–5, TVOC ppb, eCO2) | Adafruit ENS160 driver |

None of the drivers has been run against real hardware yet. They are tested with recorded and constructed frames
(`tests/test_gas_drivers.py`), so check each one on the Pi before relying on it.

**Discovery.**
1. **I2C:** every bus is scanned for ACKs. Each answering address is offered to the drivers that use it. Exactly one
   positive identification gives the device. Otherwise the device is listed as **unknown**, with the candidate
   drivers: several drivers can share an address, and an ADC cannot say which sensor is behind it.
2. **Serial / USB:** each port is first listened to (sensors that send on their own), then queried (the graphene
   handshake, the DFRobot UART query). A port that sends data no driver recognises is listed as unknown.
3. **Manual devices:** `config/sensors.yaml` (copy `config/sensors.example.yaml`) declares devices by hand.
4. **Your choices:** the dashboard (Devices → «Газові сенсори») saves the driver you pick for an unknown device under
   `choices`. The gas module then runs discovery again.

**Descriptor and readings.** Each sensor reports
`{driver, device_id, model, bus, address, quantities: [{gas, unit, kind, range, resolution, accuracy, sample_period,
source}], needs_model}`.
- One `gas_reading` per sensor and poll carries `readings: [{gas, concentration, unit, range, resolution, accuracy,
  saturated}]`.
- The graphene board's reading keeps `gas`, `gas_confidence` and `concentration_pct` (vol %, as the model outputs
  it).
- Index sensors send `air_quality_index` events instead.

**Per-sensor model.** `config/fuzzy/gas_generated.yaml` is written after every discovery. It is machine-specific and
git-ignored. You can also rebuild it with `python -m modules.hfs.gas_model_builder`. Each concentration quantity
becomes an input in its native unit:
- C_crit is converted to that unit (NH₃ 300 ppm = 0.03 vol % for the graphene model; CO 1200 ppm = 1374.7 mg/m³).
- The terms sit at the fractions of C_crit used by the `max_norm` terms of `config/fuzzy/gas.yaml` (0.1 / 0.3 /
  0.6 / 1.0).
- δ(c) = max(absolute, relative · c, resolution / 2) comes from the descriptor. Where no accuracy is published, δ is
  unknown.
- When the sensor's range ends below C_crit, the input is flagged `saturation`. Index quantities get their own
  inputs.

Every value carries a provenance entry. The gas subsystem applies the same δ to each reading, because the readings
carry their sensor's accuracy and range.

**Range below C_crit.** Example: the NH₃ module reads 0–100 ppm, but C_crit is 300 ppm. A full-scale reading
(100 ppm) is "at least 100 ppm", so its c / C_crit interval is open up to 1.2:
- the risk corridor widens towards critical;
- the HFS result lists the substance under `saturated`;
- the log says the sensor cannot confirm a critical concentration.

### Graphene board protocol

Serial or USB at 115200 baud, one JSON object per line. The board firmware must implement:

| Host sends | Board answers |
| -- | -- |
| `ID?\n` | `{"device": "graphene_fet", "serial": "<id>", "gates": 16, "vg": [0.0, 1.5], "unit": "vol_pct"}` |
| `READ\n` | `{"r": [16 resistances in Ω for V_G = 0.0 … 1.5 V]}` |

The curve goes unchanged to the trained TOD-19 model (`modules/gas/inference.GasPredictor`), which outputs the gas
and its concentration in vol %. A board reporting another unit is rejected, because the model was trained in vol %.

### On the Raspberry Pi

1. Enable the interfaces: `sudo raspi-config` → Interface Options → I2C and Serial Port (login shell over serial:
   off).
2. Add the user to the `i2c` and `dialout` groups.
3. Install the requirements: `smbus2`, `pyserial`.
4. Run `python -m drivers.gas.discovery`, then set `gas.mode: drivers`.
