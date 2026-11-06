"""
Top level of the hierarchical fuzzy system (A1 / TOD-41): risk 1 (gas), risk 2 (camera), risk 3 (audio) → overall
risk of the location. Rule base ``config/fuzzy/top.yaml``.

The rule base is the complete table of ``modules.hfs.policy.top`` over the four levels of each subsystem (64 rules)
on a Ruspini partition with product / sum inference and equal-area output terms — exactly monotonic (see
``modules/hfs/policy.py``).

Missing subsystem (no data / device offline): its input takes the default 0 ("no evidence of risk"); by monotonicity
it then neither escalates nor blocks anything and the result rests on the remaining subsystems. The result is marked
``partial`` and lists ``missing``. No subsystem at all → None.

Interval: the subsystems' intervals propagated through the (monotonic) top level — overall risk of every lower end
and of every upper end — widened to contain the crisp value. A5 / TOD-45 replaces it with the IT2 corridor.
"""
import os
from typing import Any, Dict, Optional

from core.fuzzy import FuzzyEngine, load_yaml
from modules.hfs.base import ALPHA, CONFIG_DIR, summarise
from modules.hfs.policy import LEVELS, policy_level  # noqa: F401  (re-exported: the top-level policy)

SUBSYSTEMS = ("gas", "camera", "audio")                       # risk 1, risk 2, risk 3
VARIABLE = {name: f"risk_{name}" for name in SUBSYSTEMS}
class TopLevel:
    def __init__(self, path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        cfg = ((config or {}).get("hfs") or {}).get("top") or {}
        self.path = path or cfg.get("rules") or os.path.join(CONFIG_DIR, "top.yaml")
        data = load_yaml(self.path)
        self.engine = FuzzyEngine.from_dict(data)
        self.update_sec = float(cfg.get("update_sec", data.get("update_sec", 2.0)))

    def combine(self, results: Dict[str, Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """
        :param results: {gas / camera / audio: subsystem result or None (no data)}
        :return: {risk1, risk2, risk3, overall, level, interval, partial, missing, subsystems,
                 fired_rules_by_level {gas, camera, audio, top}} or None when no subsystem has data
        """
        present = {name: results[name] for name in SUBSYSTEMS if results.get(name) is not None}
        if not present:
            return None
        missing = [name for name in SUBSYSTEMS if name not in present]

        def run(pick):
            return self.engine.infer({VARIABLE[name]: pick(r) for name, r in present.items()}, alpha=ALPHA)

        result = summarise(self.engine, run(lambda r: r["risk"]))
        low = run(lambda r: r["interval"][0])["crisp"]
        high = run(lambda r: r["interval"][1])["crisp"]
        crisp = result["risk"]
        risks = {f"risk{i}": (present[name]["risk"] if name in present else None)
                 for i, name in enumerate(SUBSYSTEMS, start=1)}
        return {**risks, "overall": crisp, "level": result["level"],
                "interval": [round(min(low, high, crisp), 4), round(max(low, high, crisp), 4)],
                "partial": bool(missing) or any(r.get("partial") for r in present.values()),
                "missing": missing,
                "subsystems": {name: {k: r[k] for k in ("risk", "level", "interval", "inputs")}
                               for name, r in present.items()},
                "fired_rules_by_level": {**{name: r["fired_rules"] for name, r in present.items()},
                                         "top": result["fired_rules"]}}
