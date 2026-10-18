# Fuzzy inference core (F15 / TOD-26)

Stage 1 of the thesis contribution "fuzzy-bounded LLM analysis with trust-weighted evidence": the shared Mamdani
engine used by the fuzzy trust layer on the input (TOD-27) and the fuzzy risk corridor on the output (TOD-28).
Deterministic, numpy only (no scikit-fuzzy), never calls the LLM or the network.

```
inputs (crisp)  ──fuzzify──▶  μ per term  ──rules (min / max)──▶  clipped output terms  ──max──▶  centroid ──▶ risk 0..1
gas 4.2 %, acetone            elevated 0.8 …   R3: elevated AND NOT sharply_rising → high 0.8         + fired_rules (why)
```

| Variable | Universe | Terms |
| -- | -- | -- |
| `gas_concentration` (per gas: `gas=acetone` selects its terms) | 0–10 % | clean, slight, elevated, dangerous |
| `gas_trend` | −10…10 %/min | falling, stable, rising, sharply_rising |
| `motion_level` | 0–1 | none, low, high |
| `people_count` | 0–10 | none, one, several, crowd |
| `unknown_share` | 0–1 | none, some, all |
| `action_risk` (`maps.action_risk`: lying 1, possible_smoking 0.6, walking 0.2 …) | 0–1 | low, medium, high |
| `time_of_day_risk` (local hour, per-location schedule) | 0–24 h | normal, off_hours, night |
| `risk` (output) | 0–1 | low, medium, high, critical |

Operators AND = min, OR = max, implication = min, aggregation = max, defuzzification = centroid (all configurable:
prod / probor / sum, bisector / mom / som / lom). A missing input uses the variable's `default` (gas: 0 = no
sensor here); without a default the rules that need it are skipped and listed in `skipped_rules`.

```python
from core.fuzzy import FuzzyEngine

engine = FuzzyEngine.from_config("config/fuzzy.yaml", location_id="warehouse")
result = engine.infer({"people_count": 2, "unknown_share": 0.5, "time_of_day_risk": 19.75})
result["crisp"]                        # 0.57
result["fired_rules"][0]["explanation"]
# 'unknown_share = 0.5 → some 1.00; time_of_day_risk = 19.75 h → off_hours 0.75 ⇒ risk IS high (0.75)'
engine.fuzzify("gas_concentration", 4.0, "acetone")    # {'clean': 0, 'slight': 0, 'elevated': 1, 'dangerous': 0}
```

Per-location overrides — `locations:` in `config/fuzzy.yaml` or a `fuzzy:` block per entry of
`config/locations.yaml`: `terms` (replace / add term definitions), `shift` (move a term; universe edges stay),
`rules` (extra rules, `source: location:<id>` in explanations), `remove_rules`, `weights`. Every combination is
validated on load (unknown variables / terms, uncovered universe regions, duplicate or conflicting rules, bad
weights / operators) and fails with a list of clear messages.

```bash
python -m core.fuzzy --check                                          # base + every location
python -m core.fuzzy gas_concentration=4.2 gas=acetone time_of_day_risk=23.5 --location warehouse
```

Golden cases (tests/test_fuzzy.py): dangerous gas → 0.91, person lying motionless → 0.91, stranger at night → 0.67,
empty room at noon → 0.11, known person working at noon → 0.11. `infer()` ≈ 0.25 ms on the dev Mac (Apple
Silicon); the Pi 4 is roughly 5–10× slower, still well below the 5 ms budget — confirm with
`python -m core.fuzzy …` on the Pi (it prints the time).
