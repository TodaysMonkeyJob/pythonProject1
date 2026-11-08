# Gas scale audit (G1 / TOD-53, 2026-11-08)

The issue: a running scenario showed acetone at 5 vol % without treating it as deadly. Every gas is now judged against
its critical concentration C_crit (NIOSH IDLH, `config/gases.yaml`). This report lists every place that used the
old percentage scale, and what was done with it.

## What the live run showed (logs/2026-11-07/events.jsonl, read only)

- **The HFS was right.** The gas subsystem gave `critical` whenever acetone was at or above C_crit (`max_norm` 1.2).
  It went back to low or medium only when the replay curve was classified `clean_air`.
- **The location turned red for the wrong reason.** In mode autonomous the danger came from the gas worker's own
  alerts: `gas_threshold` at ≥ 5 % and `gas_rising` at > 2 %/min. Those use the old placeholders of `config.yaml`
  `gas.thresholds`, not the HFS.
- **An acknowledge cleared real danger.** At 21:11:04 the dashboard's Acknowledge set the location to `safe` while
  the sensor read 13.19 vol % acetone (53 × C_crit). Nothing stopped an acknowledge from clearing a deadly reading.
- **Ammonia's corridor was not firmly critical.** The gas subsystem clipped c / C_crit to 1.2 before subtracting
  the sensor error. For 5 vol % ammonia (167 × C_crit) the lower bound came out at 0.84 C_crit, and the corridor was
  [0.65, 0.87] instead of firmly critical.

## Search

The search covered `config/`, `config.yaml`, `core/`, `modules/`, `agents/`, `dashboard/`, `experiments/`, `drivers/`,
`tools/`, `tests/` and the docs. The patterns were:
- `gas.thresholds`, `gas_thresholds`, `thresholds[gas]`, `GAS_LIMITS`;
- `rise_rate_threshold`, `trend_stable_band`, `disagreement_pct`;
- `concentration_pct` compared with constants, and `%` limits in comments;
- the `variants` of `gas_concentration`.

`tests/test_gas_safety.py::test_no_percentage_gas_thresholds_remain` repeats this search on the code and config, so
the old keys cannot come back unnoticed.

| Where | Old scale | Now |
| -- | -- | -- |
| `config.yaml` `gas.thresholds` | ethanol 5, ammonia 3, acetone 5, toluol 3 (%) — "placeholders" | removed; the worker reads C_crit / LEL from `config/gases.yaml` |
| `config.yaml` `gas.rise_rate_threshold`, `trend_stable_band` | 2 %/min, 0.2 %/min | `rise_alert_norm_per_min` 0.25 C_crit/min (= the HFS `dynamics` sharp_rise), `trend_stable_norm` 0.05 C_crit/min |
| `modules/gas/worker.py` alerts | `gas_threshold` ≥ the % threshold (critical), `gas_rising` > 2 %/min (critical) | `gas_threshold` at ratio ≥ 1 (critical), new `gas_explosive` at ≥ LEL (critical, «вибухонебезпечно»), `gas_rising` > 0.25 C_crit/min (warning: how much a rise means is the HFS `dynamics` input); every reading carries `ratio` = c / C_crit |
| location rule `gas_thresholds` (`core/locations.py`, `modules/locations/state.py`, dashboard Locations page, `config/locations.example.yaml`) | {gas: %} → critical in that location | removed; a stored value is dropped with a warning; the location state has the safety invariant instead |
| `config/fuzzy.yaml` `gas_concentration` / `gas_trend` (F15 corridor: GPT path, trust scorer, envelope, GPT view) | universe 0–10 %, base terms for a 3 % limit, `variants` for a 5 % limit (ethanol, acetone) | one scale for every gas: c / C_crit (universe 0–1.2) and C_crit / min. The terms are the former 5 % terms divided by that limit (dangerous from 1.0 C_crit). Callers normalise with `modules.hfs.gas_normalize.fuzzy_gas`. The rules are unchanged: the rules version hash (`dedd67a8cd46`) stays the same |
| `modules/envelope/state.py` `TREND_RATE` | %/min stand-ins | C_crit/min stand-ins |
| `modules/aggregator/llm_view.py` v2.1 `range_pct` / `gas_scale_pct` | the gas's own % variant | the C_crit terms × that gas's C_crit in vol % |
| `modules/aggregator/digest.py` rules line | "acetone ≥ 3.0% is dangerous here" | removed with the rule |
| `modules/trust/evidence.py` twin-sensor disagreement (`config/trust_registry.yaml`, `config/trust_priors.yaml` `policy`) | 1 vol % | 0.2 C_crit (`gas_disagreement_norm`: the former 1 % against a 5 % limit) |
| `dashboard/server.py` `/api/status` `gas_thresholds`; `views/location.js` chart line «поріг» | the % thresholds | `gas_limits` {c_crit_pct, lel_pct}; the chart line is C_crit; a red line with the safety text, «ВИБУХОНЕБЕЗПЕЧНО» when explosive |
| `experiments/pipeline.py` rules_only baseline `GAS_LIMITS` | % thresholds, rise > 2 %/min | ratio ≥ 1, `gas_threshold` / `gas_explosive` alerts → critical; > 0.25 C_crit/min → high |
| `experiments/llm.py` GPT simulator | `gas_threshold` + ≥ 50 000 ppm → critical | `gas_threshold` / `gas_explosive` (= at C_crit / LEL) → critical |
| `experiments/scripted.py` (F20 set) | vol % ramps: gas_threshold 2.0 → 6.5 %, gas_rising 0.3 → 3.3 % **labelled high** (up to 13 × C_crit), smoking with toluene 0.3 → 0.9 % **labelled medium** (up to 18 × C_crit) | shares of C_crit: gas_threshold 0.4 → 1.6 C_crit (critical), gas_rising 0.05 → 0.65 C_crit (high, below C_crit), smoking with a calm toluene background 0.02 → 0.06 C_crit (medium). Alerts follow the worker's new rules. `labels_version: 2` and `gas_scale` are in the dataset manifest and in every experiment manifest |
| `experiments/variants.py` sensor-failure drift | +1.5 vol % (6 × C_crit of acetone) | +0.5 C_crit of the gas read; clean air reads as acetone (a drifted baseline looks like gas) |
| `experiments/hfs/scenarios.yaml` (Article 1) | already relative to C_crit | + g16 (5 vol % acetone → critical, explosive) and g17 (5 vol % ammonia → critical); `labels_version: 2` |
| `tools/test_llm.py` legacy digest case | ammonia 21 000 ppm "rising" (70 × C_crit) | 210 ppm (0.7 C_crit) |
| `tools/fuzzy_docs` | figures and text for the 3 % / 5 % variants | the C_crit scale |
| `modules/hfs/gas.py`, `gas_model_builder.reading_interval` | ±δ applied after clipping at 1.2 | ±δ on the unclipped c / C_crit, then clipped |

## The issue's clarification (comment of 2026-11-08)

- **Values as % of C_crit.**
  - Every reading carries `pct_of_crit` («% від критичного рівня»): the gas worker, every driver reading, and the
    scripted scenarios.
  - `modules.hfs.gas_normalize.reading_text` writes «acetone 5 vol % (2000 % від критичного рівня)». It is used by
    the location state reasons, the incident timelines and the CSV export.
  - The HFS gas explanation, the headline and the safety text use the same wording.
  - The dashboard shows the raw value plus «% від критичного рівня»:
    - the event list, for every sensor format, which also fixes the `undefined%` of the driver readings;
    - the territory gas panel;
    - the chart, now in % of C_crit with the line at 100 % (bars capped at 300 %).
  - The fuzzy terms were already anchored at 0.1 / 0.3 / 0.6 / 1.0 C_crit and converted to each sensor's unit
    (TOD-48, `gas_model_builder`).
- **The graphene model's range.**
  - The smallest non-zero concentrations of `data.xls` are 0.4 / 0.4 / 0.25 / 0.3 vol %: 1.2 / 1.6 / 5 / 10 × C_crit
    for ethanol / acetone / toluene / ammonia. This is `modules.gas.data.MODEL_RANGE_VOL_PCT`, which a test checks
    against the data.
  - A non-zero prediction counts as at least that start (`detection_floor`, the driver's per-gas `range`), so it is
    ≥ C_crit and critical.
  - The reading carries `below_model_range` and a Ukrainian `model_note`, and the worker logs the note once per gas.
  - Model card: `modules/gas/MODEL_CARD.md`; note in `docs/sensors.md`.
- **Scenarios.** The scripted set (labels v3) no longer feeds the graphene model at all. A generic low-range sensor
  in ppm (range 0–2 × C_crit, ±10 % / ±2 % C_crit) carries every value, from calm to 1.6 × C_crit. The Article 1 set
  was already in sensor units below C_crit (ppm, µg/m³). Its vol % readings are at or above C_crit: g13 = 1 × C_crit,
  g16 / g17 = 5 vol %.
- **Old pipeline.** `flat_reading` lets the F15 corridor, the trust layer and the snapshots read every sensor format
  (driver `readings` in ppm / µg/m³), not only the graphene vol % fields.

## What is new

- `config/gases.yaml`: `lel_pct` + `lel_source`, checked against the NIOSH Pocket Guide on 2026-11-08:

  | Gas | LEL |
  | -- | -- |
  | acetone | 2.5 % |
  | ethanol | 3.3 % |
  | toluene | 1.1 % |
  | ammonia | 15 % |
  | CO | 12.5 % |
  | H₂S | 4.0 % |

  The C_crit entries already existed and were checked again: acetone 2500, ethanol 3300, toluene 500, ammonia 300,
  CO 1200 and CO₂ 40 000 ppm; PM2.5 225.5 µg/m³.
- `modules/hfs/gas_safety.py`, the safety invariant. It is independent of every rule base and agent. If the lower
  bound of any gas reading (the reading minus its sensor error) is ≥ C_crit, the final level is critical. A reading
  at or above the LEL adds `explosive: true` and «вибухонебезпечно».
  - It is applied in:
    - `hfs_risk`: gas subsystem and top level;
    - `agent_decision`: the agents host, after the validation agent, fallback included;
    - the location state: before every decider, and an acknowledge cannot clear it.
  - An overruled record is logged as an error, carries `safety_overruled` and is appended to
    `gas_safety.VIOLATIONS`. `tests/conftest.py` fails any test that causes a violation it did not ask for.

## Updated after review

- `data/rules/versions/*.yaml` (the four approved rule versions): their `gas_concentration` / `gas_trend` blocks are
  now the C_crit ones. The rules and the version hashes are unchanged, and the change is in the changelog
  (`rescale`). A `rollback` can no longer bring back the % terms.
- Offline runs (`experiments.run --offline`, `tools.fit_thresholds`, the rules gate): GPT takes no part. There is no
  call, no cache and no simulator; the fuzzy corridor decides. The earlier offline runs answered missing GPT answers
  with the simulator. The refit described in the thresholds row above was run with `--out` set to an empty scratch
  file, so there were no previous values to keep. Run correctly, the GPT-dependent values are kept.

## Left as they are

| What | Why |
| -- | -- |
| `logs/*/events.jsonl` | the user's history; the old alerts in them stay as they were |
| `reports/experiment_2026-10-19*`, `_2026-10-23`, `experiment_v2_2026-10-24` | the manifests now carry `old_scale` (TOD-53 note). The numbers in them were obtained on the old scale |
| `reports/channel_eval/gas_trend.json` + the `gas_trend` prior in `config/trust_priors.yaml` | measured on the %/min trend label (1380 / 1380 correct). Re-measured on the C_crit label in this task: 111 632 / 111 632 correct, but the longer C_crit ramps give 80 × more correlated samples, which would set u0 = 0. How to sample is a separate decision, so the measured prior stays the old one |
| `config/thresholds.yaml` | the gas `k_var` / spans were in % units (0.9499 / 0.9 on the 0–10 % universe). They are converted with the fitted κ 1.5 kept: 0.23 / 0.18. A full offline refit (`tools.fit_thresholds` on the v2 scripted set) was run but **not adopted**: all 43 GPT answers missed the cache (the payloads changed with the scale) and the level fit was degenerate (cost 0 everywhere → critical 0.40 instead of 0.76). Refit once the v2 payloads have recorded GPT answers (`experiments.run --llm azure`, new calls) |
| `reports/channel_eval/gas_concentration.json` (tolerance ±0.5 vol %) | regressor accuracy in vol %, not a danger level |
| `experiments/prompts/security_analyst_v1.md` | archived prompt text of an old run |
