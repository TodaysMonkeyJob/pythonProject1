"""
Fuzzy trust in evidence (F16 / TOD-27, component B): every event that can reach the LLM gets a trust degree
τ ∈ [0, 1], a provenance tag and — for speech / scene text — a claim type::

    τ(e) = min(μ_source(e), μ_asr(e), μ_consistency(e))

* ``μ_source`` — physical sensors 0.95 (0.3 when the device is unhealthy or disagrees with a twin sensor); speech by
  the ``trust.source`` fuzzy rules over speaker identity, face ↔ voice agreement (is the speaking person visible in
  that location?), playback suspicion (speech where working cameras see nobody) and speaker overlap; faces by the
  ``trust.identity`` rules over the gallery similarity; scene text 0.1;
* ``μ_asr`` — the ``trust.asr`` rules over Whisper's confidence (1 when the backend gave none);
* ``μ_consistency`` — 1, except for claims that would lower risk: an instruction to the system → 0.05; a
  de-escalation claim → graded by how alarming the location's sensors are (gas level / trend via the shared
  ``core.fuzzy`` risk variables, recent critical alerts / risky actions): fully alarming → 0.1.

Added to the payload (before the event is logged and reaches the aggregator)::

    {"trust": 0.05, "trust_terms": {"low": 1.0, "medium": 0.0, "high": 0.0},
     "provenance": "speech:unknown_4@warehouse", "claim_type": "deescalation", "playback_suspected": true,
     "trust_reasons": ["speaker unknown", "nobody visible in warehouse during the speech (possible playback)", …],
     "trust_components": {"source": 0.05, "asr": 0.82, "consistency": 0.1}}

Measured inputs (F22 / TOD-33): with ``trust.calibration`` files present (``tools/calibrate_channels.py``) the raw
scores are calibrated probabilities instead of hand-set term breakpoints — μ_asr = P(WER ≤ 10 % | asr_confidence),
speaker = P(same speaker | ECAPA score), identity τ = P(known | SFace similarity), gas τ = min(prior, P(label right |
gas_confidence)); base degrees come from ``config/trust_priors.yaml`` (``@trust_priors:…`` refs). A missing file falls
back to the ``trust.asr`` / ``trust.identity`` rule bases.

Local and deterministic: no network, no LLM; ≈ 0.5 ms per speech event on the dev Mac (budget on the Pi: 10 ms).
"""
import os
from typing import Any, Dict, List, Optional, Tuple


from core.calibration import load_map
from core.events import Event
from core.fuzzy import FuzzyEngine, load_yaml
from core.fuzzy.membership import parse_term
from core.fuzzy.variables import LinguisticVariable
from modules.trust.channels import EvidenceState, channel_of, epoch
from modules.trust.claims import ClaimDetector

TREND_LABEL = {"sharply_rising": 1.0, "rising": 0.7}
GAS_FRESH_SEC = 120.0
TWIN_WINDOW_SEC = 30.0


class TrustScorer:
    def __init__(self, cfg: Dict[str, Any], risk_engine: Optional[FuzzyEngine] = None,
                 state: Optional[EvidenceState] = None, base_dir: str = "config"):
        """
        :param cfg: The ``trust`` section of ``config/fuzzy.yaml``
        :param risk_engine: The base risk engine (TOD-26) — its gas variables grade "how alarming" the sensors are
        :param base_dir: Folder of ``fuzzy.yaml`` — ``trust.calibration`` paths are relative to it
        """
        self.cfg = cfg
        self.base = {"physical": 0.95, "physical_degraded": 0.3, "redundant_disagreement": 0.6, "scene_text": 0.1,
                     "identity_unknown": 0.8, "identity_unsettled": 0.8, **(cfg.get("base") or {})}
        self.consistency_cfg = {"deescalation_floor": 0.1, "instruction": 0.05, **(cfg.get("consistency") or {})}
        self.source = FuzzyEngine.from_dict(cfg["source"])
        self.asr = FuzzyEngine.from_dict(cfg["asr"])
        self.identity = FuzzyEngine.from_dict(cfg["identity"])
        self.terms = LinguisticVariable("trust", 0, 1, 0.01, {t: parse_term(s) for t, s in cfg["terms"].items()})
        self.risk = risk_engine
        self.claims = ClaimDetector(cfg.get("claims") or None)
        self.state = state or EvidenceState(float(cfg.get("presence_stale_sec", 600)),
                                            float(cfg.get("alarm_memory_sec", 300)))
        self.grace = float(cfg.get("presence_grace_sec", 3))
        self.gas_confidence_min = float(cfg.get("gas_confidence_min", 0.5))
        self.gas_disagreement = float(cfg.get("gas_disagreement_pct", 1.0))
        self.maps = {}
        for channel, rel in (cfg.get("calibration") or {}).items():
            mapping = load_map(os.path.join(base_dir, rel))
            if mapping is not None:
                self.maps[channel] = mapping

    @classmethod
    def from_config(cls, path: str = "config/fuzzy.yaml") -> "TrustScorer":
        data = load_yaml(path)
        if "trust" not in data:
            raise KeyError(f"{path} has no trust section")
        return cls(data["trust"], FuzzyEngine.from_dict(data), base_dir=os.path.dirname(os.path.abspath(path)))

    # --- entry point --------------------------------------------------------------------------------------------

    def apply(self, event: Event) -> bool:
        """
        Scores an evidence event in place (payload gets trust fields) and updates the evidence state.

        :return: True if the payload changed (the router re-serialises the event)
        """
        fields = self.score(event)
        self.state.observe(event)
        if fields is None:
            return False
        event.payload.update(fields)
        return True

    def score(self, event: Event) -> Optional[Dict[str, Any]]:
        channel = channel_of(event)
        if channel is None or "trust" in event.payload:      # not evidence, or scored already (e.g. by an edge node)
            return None
        if channel == "physical":
            return self._physical(event)
        if channel == "identity":
            return self._identity(event)
        return self._claim_bearing(event, channel)

    # --- channels -----------------------------------------------------------------------------------------------

    def _result(self, tau: float, provenance: str, reasons: List[str], **extra) -> Dict[str, Any]:
        tau = round(max(0.0, min(1.0, tau)), 3)
        return {"trust": tau, "trust_terms": {t: round(v, 3) for t, v in self.terms.fuzzify(tau).items()},
                "provenance": provenance, "trust_reasons": reasons, **extra}

    def _physical(self, event: Event) -> Dict[str, Any]:
        p, device, location = event.payload, event.device_id, event.location_id
        tau, reasons = self.base["physical"], ["physical sensor"]
        if device in self.state.unhealthy:
            tau, reasons = self.base["physical_degraded"], [f"device {device} is offline / in error"]
        if event.type == "gas_reading":
            confidence = p.get("gas_confidence")
            if device not in self.state.unhealthy:
                tau = float(self.base.get("gas", tau))
            if confidence is not None and "gas" in self.maps:
                calibrated = self.maps["gas"](float(confidence))
                if calibrated < tau:
                    tau = calibrated
                    reasons.append(f"gas label right with P = {calibrated:.2f} at confidence {float(confidence):.2f}")
            elif confidence is not None and float(confidence) < self.gas_confidence_min:
                tau = min(tau, self.base["physical"] * max(0.3, float(confidence) / self.gas_confidence_min))
                reasons.append(f"gas classifier unsure ({float(confidence):.2f})")
            now = epoch(event.timestamp) or 0.0
            for other, reading in self.state.at(location).gas.items():
                if other == device or now - reading["t"] > TWIN_WINDOW_SEC or reading["concentration"] is None:
                    continue
                delta = abs(float(p.get("concentration_pct") or 0) - float(reading["concentration"]))
                if delta > self.gas_disagreement:
                    tau = min(tau, self.base["redundant_disagreement"])
                    reasons.append(f"disagrees with {other} by {delta:.1f} %")
                else:
                    reasons.append(f"agrees with {other}")
        kind = "sensor" if event.source == "gas" else event.source
        return self._result(tau, f"{kind}:{device}@{location}", reasons)

    def _identity(self, event: Event) -> Dict[str, Any]:
        p = event.payload
        provenance = f"camera:{event.device_id}@{event.location_id}"
        if p.get("person_id") is None:
            return self._result(self.base["identity_unsettled"], provenance, ["person seen, face not identified yet"])
        if not p.get("known"):
            return self._result(self.base["identity_unknown"], provenance, ["unknown person (physical observation)"])
        similarity = float(p.get("similarity") or 0.0)
        if "face" in self.maps:
            tau = self.maps["face"](similarity)
            return self._result(tau, provenance, [f"face match {similarity:.2f} → P(known) {tau:.2f}"])
        tau = self.identity.infer({"face_match": similarity})["crisp"]
        return self._result(tau, provenance, [f"face match {similarity:.2f}"])

    def _claim_bearing(self, event: Event, channel: str) -> Dict[str, Any]:
        p, location = event.payload, event.location_id
        claim = self.claims.detect(p.get("text"))
        reasons: List[str] = []
        extra: Dict[str, Any] = {"claim_type": claim.claim_type}
        spans = [span for kind, span in claim.spans if kind == claim.claim_type]
        if spans:
            extra["claim_spans"] = spans[:5]
        if channel == "scene_text":
            source = self.base["scene_text"]
            reasons.append("text seen in the frame (sign / screen) — weak evidence")
            provenance = f"scene_text:{event.device_id}@{location}"
            asr = 1.0
        else:
            source, asr, playback, provenance = self._speech(event, reasons)
            extra["playback_suspected"] = playback
        consistency = self._consistency(event, claim.claim_type, reasons)
        extra["trust_components"] = {"source": round(source, 3), "asr": round(asr, 3),
                                     "consistency": round(consistency, 3)}
        return self._result(min(source, asr, consistency), provenance, reasons, **extra)

    def _speech(self, event: Event, reasons: List[str]) -> Tuple[float, float, bool, str]:
        p, location = event.payload, event.location_id
        start = epoch(p.get("start")) or epoch(event.timestamp) or 0.0
        end = epoch(p.get("end")) or start
        status, speaker_id = p.get("speaker_status"), p.get("person_id")
        if "speaker" in self.maps and p.get("speaker_score") is not None \
                and status in ("known", "unknown", "uncertain"):
            score = float(p["speaker_score"])
            speaker = self.maps["speaker"](score)
            reasons.append(f"known speaker {p.get('speaker_name') or speaker_id} ({score:.2f})" if status == "known"
                           else f"speaker {status}")
            reasons.append(f"P(same speaker) {speaker:.2f}")
        elif status == "known":
            score = float(p.get("speaker_score") or 0.0)
            speaker = 0.7 + 0.3 * min(1.0, max(0.0, (score - 0.3) / 0.4))
            reasons.append(f"known speaker {p.get('speaker_name') or speaker_id} ({score:.2f})")
        elif status == "unknown":
            speaker = 0.1
            reasons.append("speaker unknown")
        else:
            speaker = 0.4
            reasons.append("speaker uncertain" if status == "uncertain" else "no speaker identification")
        visible = self.state.visible(location, end)
        if speaker_id and speaker_id in visible:
            face_voice = 1.0
            reasons.append("the speaker is visible")
        elif any(pid is None for pid in visible):
            face_voice = 0.65
            reasons.append("unidentified people visible")
        elif visible:
            face_voice = 0.0
            reasons.append("only other identified people are visible (face ↔ voice mismatch)")
        else:
            face_voice = 0.35
            reasons.append("speaker off camera")
        playback = self.state.has_working_camera(location) and \
            not self.state.person_seen_between(location, start, end, self.grace)
        if playback:
            reasons.append(f"nobody visible in {location} during the speech (possible playback)")
        inputs = {"speaker": speaker, "face_voice": face_voice, "playback": 1.0 if playback else 0.0,
                  "overlap": 1.0 if p.get("speaker_overlap") else 0.0}
        if p.get("speaker_overlap"):
            reasons.append("another speaker talked at the same time")
        source = self.source.infer(inputs)["crisp"]
        confidence = p.get("asr_confidence")
        if confidence is None:
            asr = 1.0
        else:
            asr = self.maps["asr"](float(confidence)) if "asr" in self.maps \
                else self.asr.infer({"asr_confidence": float(confidence)})["crisp"]
            if asr < 0.7:
                reasons.append(f"low ASR confidence ({float(confidence):.2f})")
        who = p.get("speaker_name") or (speaker_id[:8] if speaker_id else "unattributed")
        return source, asr, playback, f"speech:{who}@{location}"

    def _consistency(self, event: Event, claim_type: str, reasons: List[str]) -> float:
        if claim_type == "instruction_to_system":
            reasons.append("instruction to the system")
            return float(self.consistency_cfg["instruction"])
        if claim_type != "deescalation":
            return 1.0
        alarm, why = self.alarm_degree(event.location_id, epoch(event.timestamp) or 0.0)
        reasons.append("claims everything is safe" + (f", but {'; '.join(why)}" if why else ""))
        floor = float(self.consistency_cfg["deescalation_floor"])
        return 1.0 - (1.0 - floor) * alarm

    def alarm_degree(self, location_id: Optional[str], now: float) -> Tuple[float, List[str]]:
        """:return: (how alarming the location's physical evidence is 0..1, readable contradictions)"""
        degree, why = 0.0, []
        reading = self.state.latest_gas(location_id)
        if reading and now - reading["t"] <= GAS_FRESH_SEC:
            if self.risk is not None and reading["concentration"] is not None:
                level = self.risk.fuzzify("gas_concentration", float(reading["concentration"]), reading["gas"])
                high = max(level.get("elevated", 0.0), level.get("dangerous", 0.0))
                if high > 0:
                    degree = max(degree, high)
                    why.append(f"gas {reading['gas']} = {reading['concentration']} %")
            if reading["rate"] is not None and self.risk is not None:
                trend = self.risk.fuzzify("gas_trend", float(reading["rate"]))
                rising = max(trend.get("rising", 0.0), trend.get("sharply_rising", 0.0))
            else:
                rising = TREND_LABEL.get(str(reading.get("trend")), 0.0)
            if rising > 0:
                degree = max(degree, rising)
                rate = f" ({reading['rate']} %/min)" if reading["rate"] is not None else ""
                why.append(f"contradicts gas_trend = {reading.get('trend') or 'rising'}{rate}")
        for what in self.state.recent_alarms(location_id, now):
            degree = 1.0
            why.append(f"recent {what}")
        return degree, list(dict.fromkeys(why))
