# Model card: graphene FET gas classifier + concentration regressor (TOD-19)

| | |
| -- | -- |
| What | an R(V_G) curve of our graphene FET sensor (16 gate voltages 0.0–1.5 V) → gas class (ethanol, ammonia, acetone, toluol, clean_air) + concentration in vol % |
| Files | `models/gas_classifier.joblib`, `models/gas_concentration.joblib` (git-ignored; `python -m modules.gas.train`) |
| Metrics | `reports/gas_metrics.json` (trained 2026-10-15, scikit-learn 1.9.1) |
| Data | `data/gas/data.xls`: 144 original curves (4 sheets = gases, 4 clean-air rows); test = 29 original rows split before augmentation |
| Classifier | SVC (poly): macro-F1 1.0 on the test split, 0.997 ± 0.009 over 10 repeated splits |
| Regressor | SVR per gas: test MAE 0.004–0.018 vol % (CV 0.010–0.018) |
| Runtime | `modules/gas/inference.GasPredictor`; the gas worker (`gas.mode: replay / sensor`) and the `graphene_fet` driver (`gas.mode: drivers`) |

## Range: it starts at or above the critical level (G1 / TOD-53)

| Gas | Training range (vol %) | C_crit (NIOSH IDLH) | Start of range |
| -- | -- | -- | -- |
| Ethanol | 0.4 – 14.0 | 0.33 vol % (3300 ppm) | 1.2 × C_crit (120 %) |
| Acetone | 0.4 – 13.2 | 0.25 vol % (2500 ppm) | 1.6 × C_crit (160 %) |
| Toluene (toluol) | 0.25 – 9.0 | 0.05 vol % (500 ppm) | 5 × C_crit (500 %) |
| Ammonia | 0.3 – 10.8 | 0.03 vol % (300 ppm) | 10 × C_crit (1000 %) |

The model has never seen a concentration below C_crit. It can identify the gas and tell clean air from "above
critical". It cannot grade safe, low or medium levels, and a predicted value below the start of the range is an
extrapolation.

How the system uses it:
- **A non-zero prediction counts as at least the start of the range.** That start is
  `modules.gas.data.MODEL_RANGE_VOL_PCT`, which is the driver's per-gas `range` and the worker's `detection_floor`.
  So every gas reading of this model is **≥ C_crit**, and the safety invariant makes it critical.
- **The reading explains itself.** It carries `below_model_range` and the Ukrainian `model_note`, and the worker logs
  the note once per gas.
- **Grading below C_crit needs other data:** low-range sensors (electrochemical / MOS / PID, `drivers/gas`,
  `docs/sensors.md`) or new low-concentration measurements to retrain the model on.
- **Test scenarios below C_crit never use this model** (`experiments/scripted.py`: a low-range sensor in ppm).

## Other limits

- **One gas per curve.** Mixtures were not measured.
- **No smoke.** The sensor measures solvent / ammonia vapours, not smoke (owner review 2026-10-24).
- **Laboratory data only.** All data comes from one sensor in the lab; field drift is watched by the clean-air
  self-test (`modules/trust/health.py`).
- **Pickles depend on the scikit-learn version.** Retrain after upgrading it.
