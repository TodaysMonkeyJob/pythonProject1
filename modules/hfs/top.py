"""
Top level of the hierarchical fuzzy system (A1 / TOD-41): risk 1 (gas), risk 2 (camera), risk 3 (audio) → overall
risk of the location. Rule base ``config/fuzzy/top.yaml``.

The rule base is the complete table of ``modules.hfs.policy.top`` over the four levels of each subsystem (64 rules)
on a Ruspini partition with product / sum inference and equal-area output terms — exactly monotonic (see
``modules/hfs/policy.py``).

Missing subsystem (no data / device offline): its input takes the default 0 ("no evidence of risk"); by monotonicity
it then neither escalates nor blocks anything and the result rests on the remaining subsystems. The result is marked
``partial`` and lists ``missing``. No subsystem at all → None.

Corridor (A5 / TOD-45): every subsystem corridor [y_l, y_r] enters the top level as an interval type-2 input — the
midpoint with δ = half its width (``config/uncertainty.yaml`` ``top.risk``) — and Karnik–Mendel gives the overall
corridor [y_l, y_r] (widened to contain the type-1 overall risk of the crisp subsystem risks). ``widened_by`` names
the subsystems whose uncertainty widens it most.
"""
import os
from typing import Any, Dict, Optional

from core.fuzzy import FuzzyEngine, load_yaml
from core.fuzzy.engine2 import FuzzyEngine2
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
        self.engine2 = FuzzyEngine2(self.engine, area_weighted=True)
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

        crisp_inputs = {VARIABLE[name]: r["risk"] for name, r in present.items()}
        centres = {VARIABLE[name]: (r["interval"][0] + r["interval"][1]) / 2 for name, r in present.items()}
        deltas = {VARIABLE[name]: (r["interval"][1] - r["interval"][0]) / 2 for name, r in present.items()}
        it2 = self.engine2.infer(centres, deltas=deltas) if any(d > 0 for d in deltas.values()) else None
        result = summarise(self.engine, self.engine.infer(crisp_inputs, alpha=ALPHA), it2)
        crisp = result["risk"]
        risks = {f"risk{i}": (present[name]["risk"] if name in present else None)
                 for i, name in enumerate(SUBSYSTEMS, start=1)}
        return {**risks, "overall": crisp, "level": result["level"],
                "interval": result["interval"], "corridor_width": result["corridor_width"],
                "widened_by": [{**w, "variable": w["variable"].replace("risk_", "")}
                               for w in result.get("widened_by", [])],
                "partial": bool(missing) or any(r.get("partial") for r in present.values()),
                "missing": missing,
                "subsystems": {name: {k: r[k] for k in ("risk", "level", "interval", "corridor_width", "inputs",
                                                        "deltas", "widened_by", "excluded") if k in r}
                               for name, r in present.items()},
                "fired_rules_by_level": {**{name: r["fired_rules"] for name, r in present.items()},
                                         "top": result["fired_rules"]}}
