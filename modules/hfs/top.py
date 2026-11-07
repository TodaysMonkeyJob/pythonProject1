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

A9 (TOD-51): the inputs are ``max_risk`` / ``second_risk`` / ``third_risk`` of the active subsystems
(``top_escalation.py``: calm band, co-occurrence window, benign / correlated exceptions); the result carries
``escalation`` and a Ukrainian ``explanation``.
"""
import os
from typing import Any, Dict, Optional

from core.fuzzy import FuzzyEngine, load_yaml
from core.fuzzy.engine2 import FuzzyEngine2
from modules.hfs.base import ALPHA, CONFIG_DIR, summarise
from modules.hfs.policy import LEVELS, policy_level  # noqa: F401  (re-exported: the top-level policy)
from modules.hfs.top_escalation import Analysis, TopEscalation, explain

SUBSYSTEMS = ("gas", "camera", "audio")                       # risk 1, risk 2, risk 3
class TopLevel:
    def __init__(self, path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        cfg = ((config or {}).get("hfs") or {}).get("top") or {}
        self.path = path or cfg.get("rules") or os.path.join(CONFIG_DIR, "top.yaml")
        data = load_yaml(self.path)
        self.engine = FuzzyEngine.from_dict(data)
        self.engine2 = FuzzyEngine2(self.engine, area_weighted=True)
        self.update_sec = float(cfg.get("update_sec", data.get("update_sec", 2.0)))
        self.escalation = TopEscalation(cfg, level_of=self.level_of)

    def level_of(self, risk: float) -> str:
        terms = self.engine.fuzzify("max_risk", risk)
        return max(terms, key=terms.get)

    def combine(self, results: Dict[str, Optional[Dict[str, Any]]],
                analysis: Optional[Analysis] = None) -> Optional[Dict[str, Any]]:
        """
        :param results: {gas / camera / audio: subsystem result or None (no data)}
        :param analysis: the escalation analysis of this location (``TopEscalation.analyse`` with the co-occurrence
                         window and the exceptions); None → one stateless evaluation of these results
        :return: {risk1, risk2, risk3, overall, level, interval, partial, missing, subsystems, escalation,
                 explanation, fired_rules_by_level {gas, camera, audio, top}} or None when no subsystem has data
        """
        present = {name: results[name] for name in SUBSYSTEMS if results.get(name) is not None}
        if not present:
            return None
        missing = [name for name in SUBSYSTEMS if name not in present]
        if analysis is None:
            analysis = self.escalation.analyse({name: results.get(name) for name in SUBSYSTEMS}, stateless=True)
        inputs = analysis.inputs()
        it2_inputs = analysis.it2()
        deltas = it2_inputs["deltas"]
        it2 = self.engine2.infer(it2_inputs["centres"], deltas=deltas) if any(d > 0 for d in deltas.values()) \
            else None
        result = summarise(self.engine, self.engine.infer(inputs, alpha=ALPHA), it2)
        crisp = result["risk"]
        counted = analysis.ranked()
        source = {"max_risk": analysis.max_signal, "second_risk": counted[1] if len(counted) >= 2 else None,
                  "third_risk": counted[2] if len(counted) >= 3 else None}
        risks = {f"risk{i}": (present[name]["risk"] if name in present else None)
                 for i, name in enumerate(SUBSYSTEMS, start=1)}
        return {**risks, "overall": crisp, "level": result["level"],
                "interval": result["interval"], "corridor_width": result["corridor_width"],
                "widened_by": [{**w, "variable": source[w["variable"]].name if source.get(w["variable"]) else
                                w["variable"]} for w in result.get("widened_by", [])],
                "partial": bool(missing) or any(r.get("partial") for r in present.values()),
                "missing": missing, "inputs": inputs, "escalation": analysis.payload(),
                "explanation": explain(analysis, result["level"], inputs["third_risk"] > self.escalation.calm),
                "subsystems": {name: {k: r[k] for k in ("risk", "level", "interval", "corridor_width", "inputs",
                                                        "deltas", "widened_by", "excluded", "escalation",
                                                        "explanation", "saturated", "headline") if k in r}
                               for name, r in present.items()},
                "fired_rules_by_level": {**{name: r["fired_rules"] for name, r in present.items()},
                                         "top": result["fired_rules"]}}
