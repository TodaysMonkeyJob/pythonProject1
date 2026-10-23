"""
Confirmations / refutations of the trust registry channels from the routed event stream (F23 / TOD-34).

Channel: Δr (confirm) / Δs (refute)

* objects — a track lives longer than ``flicker_sec``, a face / pose is found on it, another camera of the location
  sees the same class / the track disappears within ``flicker_sec``; operator;
* motion — an object track on the camera within ``corroborate_sec`` / operator;
* pose_action — the action held ≥ ``action_window_sec`` / replaced sooner (a flip); operator either way;
* smoking — gas_confirmed or gas rising in the location within ``smoking_gas_sec`` / operator;
* face_identity — the same person again on the track, the voice of the visible person matches / merge, unmerge,
  rename of that person; operator;
* scene_text — an escalating text while the sensors alarm / a calming or instructing text they contradict;
* vad — Whisper returned text / a ``trust_evidence`` refute (Whisper heard no speech);
* asr — operator correction: a small edit confirms, more than ``transcript_edit_ratio`` refutes;
* speaker_identity — the speaker is the visible person / only other identified people are visible; operator;
* face_voice_link — ``identity_linked`` / merge or unmerge of a linked person;
* gas_class / gas_concentration / gas_trend — the twin sensor agrees (≤ every ``twin_every_sec``), the clean-air
  self-test passes (≤ every ``selftest_every_sec``) / the twin disagrees, the self-test fails, operator alarm false;
* person_registry — a day without corrections / merge, unmerge, rename, delete.

Health: ``device_health`` events (worker self-diagnostics) → ``set_health``; ``device_offline`` → check
``offline``; a microphone's ``digital_silence`` while its location shows people → ``stuck_silence``; more than
``identity.unknown_per_hour`` new unknown persons → ``unknown_storm`` on every camera's face_identity. Operator
feedback (dashboard) arrives as ``trust_feedback`` {channel_id, verdict confirm/refute, reason}.
"""
from collections import deque
from typing import Any, Deque, Dict, Optional, Set, Tuple

from core.events import Event
from modules.trust.channels import epoch
from modules.trust.registry import TrustRegistry

IDENTITY_DEVICE = "identity"
CORRECTIONS = ("identity_merged", "identity_unmerged", "identity_renamed", "identity_deleted")


def kind_of_event(event: Event) -> Optional[str]:
    """Device kind of the event's source device (None = not a device event)"""
    if event.source == "camera" or event.type in ("object_detected", "motion", "scene_active", "action",
                                                  "person_identified", "scene_text"):
        return "camera"
    if event.source in ("audio", "microphone") or event.type in ("speech", "sound", "mic_state"):
        return "mic"
    if event.source == "gas" or event.type == "gas_reading":
        return "gas"
    return None


def edit_ratio(a: str, b: str) -> float:
    """Character-level Levenshtein distance / max length"""
    a, b = a or "", b or ""
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y))
        previous = current
    return previous[-1] / max(len(a), len(b))


class TrustEvidence:
    def __init__(self, registry: TrustRegistry, cfg: Dict[str, Any], scorer=None):
        """
        :param cfg: The whole ``config/trust_registry.yaml``
        :param scorer: The ``TrustScorer`` (evidence state: who is visible, alarm degree of a location)
        """
        self.registry = registry
        self.ev = cfg.get("evidence") or {}
        self.gas_cfg = cfg.get("gas") or {}
        self.identity_cfg = cfg.get("identity") or {}
        self.mic_cfg = cfg.get("mic") or {}
        self.scorer = scorer
        self.tracks: Dict[Tuple[str, Any], Dict[str, Any]] = {}        # (camera, track) → appeared, class, person
        self.actions: Dict[Tuple[str, Any], Tuple[str, float]] = {}
        self.last_object: Dict[str, float] = {}                         # camera → last object event time
        self.smoking: Dict[str, float] = {}                             # location → possible_smoking time
        self.twin_at: Dict[Tuple[str, str], float] = {}
        self.selftest_at: Dict[str, float] = {}
        self.person_cameras: Dict[str, Set[str]] = {}                   # person_id → cameras that identified them
        self.person_mics: Dict[str, Set[str]] = {}                      # person_id → mics linked to them
        self.unknown_created: Deque[float] = deque()
        self.silent_since: Dict[str, float] = {}                        # mic → digital silence since
        self.last_correction: Optional[float] = None
        self.quiet_confirmed: Optional[float] = None
        self.reviews: Dict[str, Dict[str, Any]] = {}                   # F25: open escalation reviews by review_id
        self.suggestions = None                     # modules.rules.suggester.SuggestionTriggers (F27, orchestrator)

    # --- helpers -------------------------------------------------------------------------------------------------

    def _channel(self, device_id: Optional[str], kind: str, channel: str, location: Optional[str],
                 now: float) -> Optional[str]:
        if not device_id or device_id == "unknown":
            return None
        self.registry.ensure_device(device_id, kind, location, now)
        cid = f"{device_id}/{channel}"
        return cid if cid in self.registry.channels else None

    def _confirm(self, cid: Optional[str], reason: str, now: float, n: float = 1.0, source: str = "evidence") -> None:
        if cid:
            self.registry.confirm(cid, n, reason, now, source)

    def _refute(self, cid: Optional[str], reason: str, now: float, n: float = 1.0, source: str = "evidence") -> None:
        if cid:
            self.registry.refute(cid, n, reason, now, source)

    def _devices(self, kind: str, location: Optional[str] = None):
        return [d for d, info in self.registry.devices.items() if info.get("kind") == kind
                and (location is None or info.get("location_id") == location)]

    # --- entry point ---------------------------------------------------------------------------------------------

    def observe(self, event: Event, scored: Optional[Dict[str, Any]] = None) -> None:
        now = epoch(event.timestamp) or 0.0
        p, t = event.payload, event.type
        if self.suggestions is not None and hasattr(self.suggestions, "observe"):
            if t == "trust_feedback" and not p.get("location_id"):
                device = str(p.get("channel_id") or "").rpartition("/")[0]
                p = {**p, "location_id": (self.registry.devices.get(device) or {}).get("location_id")}
                event = event.model_copy(update={"payload": p})
            self.suggestions.observe(event, now)                    # F27: rule-suggestion triggers (no bus, no logs)
        if t in ("device_online", "device_updated", "device_offline"):
            device, kind = p.get("device_id") or event.device_id, p.get("kind")
            self.registry.ensure_device(device, kind, p.get("location_id"), now)
            offline = t == "device_offline" or (t == "device_updated" and p.get("status") not in (None, "online"))
            self.registry.set_health(device, {"offline": {"detail": p.get("status")}} if offline else {}, now,
                                     reporter="registry")
            return
        if t == "device_health":
            device = p.get("device_id") or event.device_id
            self.registry.ensure_device(device, p.get("kind") or kind_of_event(event), event.location_id, now)
            failing = dict(p.get("failing") or {})
            if p.get("kind") == "mic":
                failing = self._stuck_silence(device, event.location_id, failing, now)
            self.registry.set_health(device, failing, now, reporter=p.get("reporter") or "device")
            if "baseline_ok" in p:
                self._selftest(device, event.location_id, bool(p["baseline_ok"]), p.get("baseline_detail"), now)
            return
        if t == "trust_feedback":
            self._feedback(p, now)
            return
        if t == "escalation_review":
            self.reviews[str(event.payload.get("review_id"))] = {**event.payload, "at": now}
            if len(self.reviews) > 200:
                self.reviews.pop(next(iter(self.reviews)))
            return
        if t == "escalation_resolved":
            self._resolved(event.payload, now)
            return
        if t == "trust_evidence":
            cid = p.get("channel_id") or self._channel(event.device_id, p.get("kind") or kind_of_event(event) or "",
                                                       p.get("channel", ""), event.location_id, now)
            (self._confirm if p.get("verdict") == "confirm" else self._refute)(cid, p.get("reason", ""), now)
            return
        if t.startswith("identity_"):
            self._identity(event, now)
            return
        kind = kind_of_event(event)
        if kind is None:
            return
        self.registry.ensure_device(event.device_id, kind, event.location_id, now)
        handler = getattr(self, f"_on_{t}", None)
        if handler is not None:
            handler(event, now, scored or {})
        self._quiet_registry(now)

    # --- camera --------------------------------------------------------------------------------------------------

    def _on_object_detected(self, event: Event, now: float, scored) -> None:
        p, cam, loc = event.payload, event.device_id, event.location_id
        key = (cam, p.get("track_id"))
        cid = self._channel(cam, "camera", "objects", loc, now)
        self.last_object[cam] = now
        flicker = float(self.ev.get("flicker_sec", 2))
        if p.get("state") == "disappeared":
            track = self.tracks.pop(key, None)
            if track and not track["confirmed"]:
                if now - track["at"] < flicker:
                    self._refute(cid, f"трек #{p.get('track_id')} ({p.get('class')}) мерехтить", now)
                else:
                    self._confirm(cid, f"трек #{p.get('track_id')} ({p.get('class')}) стабільний", now)
            self.actions.pop(key, None)
            return
        others = [k for k, tr in self.tracks.items() if k[0] != cam and tr["class"] == p.get("class")
                  and tr["location"] == loc]
        self.tracks[key] = {"at": now, "class": p.get("class"), "location": loc, "confirmed": False}
        if others:
            self.tracks[key]["confirmed"] = True
            self._confirm(cid, f"{p.get('class')} бачить і інша камера локації", now)
        self._settle_tracks(cam, now)

    def _settle_tracks(self, cam: str, now: float) -> None:
        flicker = float(self.ev.get("flicker_sec", 2))
        for key, track in self.tracks.items():
            if key[0] == cam and not track["confirmed"] and now - track["at"] >= flicker:
                track["confirmed"] = True
                self._confirm(f"{cam}/objects", f"трек #{key[1]} ({track['class']}) стабільний", now)

    def _on_motion(self, event: Event, now: float, scored) -> None:
        cam = event.device_id
        self._settle_tracks(cam, now)
        if event.payload.get("state") == "end":
            return
        window = float(self.ev.get("corroborate_sec", 5))
        if any(k[0] == cam for k in self.tracks) or now - self.last_object.get(cam, float("-inf")) <= window:
            self._confirm(self._channel(cam, "camera", "motion", event.location_id, now), "рух збігся з об'єктом", now)

    def _on_scene_active(self, event: Event, now: float, scored) -> None:
        self._settle_tracks(event.device_id, now)

    def _on_action(self, event: Event, now: float, scored) -> None:
        p, cam, loc = event.payload, event.device_id, event.location_id
        self._settle_tracks(cam, now)
        key = (cam, p.get("track_id"))
        action = str(p.get("action"))
        window = float(self.ev.get("action_window_sec", 4))
        pose = self._channel(cam, "camera", "pose_action", loc, now)
        previous = self.actions.get(key)
        if previous and previous[0] != action:
            if now - previous[1] < window:
                self._refute(pose, f"дія #{p.get('track_id')} {previous[0]} → {action} за {now - previous[1]:.0f} с",
                             now)
            else:
                self._confirm(pose, f"дія #{p.get('track_id')} {previous[0]} трималася {now - previous[1]:.0f} с", now)
        if track := self.tracks.get(key):
            if not track["confirmed"]:
                track["confirmed"] = True
                self._confirm(f"{cam}/objects", f"на треку #{key[1]} розпізнано позу", now)
        self.actions[key] = (action, now)
        if action == "possible_smoking":
            smoking = self._channel(cam, "camera", "smoking", loc, now)
            if p.get("gas_confirmed"):
                self._confirm(smoking, "газ підтвердив куріння", now)
            else:
                self.smoking[loc] = now

    def _on_person_identified(self, event: Event, now: float, scored) -> None:
        p, cam, loc = event.payload, event.device_id, event.location_id
        key = (cam, p.get("track_id"))
        person = p.get("person_id")
        if track := self.tracks.get(key):
            if not track["confirmed"]:
                track["confirmed"] = True
                self._confirm(f"{cam}/objects", f"на треку #{key[1]} розпізнано обличчя", now)
            if person and track.get("person") == person and not track.get("identity_confirmed"):
                track["identity_confirmed"] = True
                self._confirm(self._channel(cam, "camera", "face_identity", loc, now),
                              "та сама людина на треку повторно", now)
            track["person"] = person or track.get("person")
        if person:
            self.person_cameras.setdefault(person, set()).add(cam)
        if person and not p.get("known") and p.get("identity", "").startswith("unknown_"):
            self._unknown(person, now)

    def _on_scene_text(self, event: Event, now: float, scored) -> None:
        claim = scored.get("claim_type") or event.payload.get("claim_type")
        if self.scorer is None or claim in (None, "none"):
            return
        alarm, _ = self.scorer.alarm_degree(event.location_id, now)
        cid = self._channel(event.device_id, "camera", "scene_text", event.location_id, now)
        if claim in ("deescalation", "instruction_to_system") and alarm >= 0.5:
            self._refute(cid, "текст у кадрі суперечить сенсорам", now)
        elif claim == "escalation" and alarm >= 0.5:
            self._confirm(cid, "текст у кадрі узгоджується з сенсорами", now)

    # --- microphone ----------------------------------------------------------------------------------------------

    def _on_speech(self, event: Event, now: float, scored) -> None:
        p, mic, loc = event.payload, event.device_id, event.location_id
        if str(p.get("text") or "").strip():
            self._confirm(self._channel(mic, "mic", "vad", loc, now), "Whisper повернув текст", now)
        person = p.get("person_id")
        if not person or p.get("speaker_status") != "known" or self.scorer is None:
            return
        visible = self.scorer.state.visible(loc, now)
        speaker = self._channel(mic, "mic", "speaker_identity", loc, now)
        if person in visible:
            self._confirm(speaker, "голос збігся з видимою людиною", now)
            for cam in self._devices("camera", loc):
                self._confirm(f"{cam}/face_identity", "голос тієї ж людини підтвердив обличчя", now)
        elif visible and all(v is not None for v in visible):
            self._refute(speaker, "видно лише інших розпізнаних людей", now)

    def _stuck_silence(self, mic: str, location: Optional[str], failing: Dict[str, Any], now: float) -> Dict[str, Any]:
        """The mic's ``digital_silence`` becomes ``stuck_silence`` once it lasts ``mic.stuck_silence_min`` while
        the location's cameras see people (a dead microphone in an occupied room)."""
        silent = "digital_silence" in failing
        failing = {k: v for k, v in failing.items() if k != "digital_silence"}
        if not silent:
            self.silent_since.pop(mic, None)
            return failing
        since = self.silent_since.setdefault(mic, now)
        minutes = (now - since) / 60
        people = bool(self.scorer.state.visible(location, now)) if self.scorer is not None else False
        if minutes >= float(self.mic_cfg.get("stuck_silence_min", 10)) and people:
            failing["stuck_silence"] = {"value": round(minutes, 1), "detail": f"{minutes:.0f} хв", "since": since}
        return failing

    # --- gas -----------------------------------------------------------------------------------------------------

    def _on_gas_reading(self, event: Event, now: float, scored) -> None:
        p, sensor, loc = event.payload, event.device_id, event.location_id
        if self.scorer is not None:
            limit = float(self.gas_cfg.get("disagreement_pct", 1.0))
            for other, reading in self.scorer.state.at(loc).gas.items():
                if other == sensor or now - reading["t"] > 30 or reading["concentration"] is None:
                    continue
                delta = abs(float(p.get("concentration_pct") or 0) - float(reading["concentration"]))
                pair = tuple(sorted((sensor, other)))
                if delta > limit:
                    for device in pair:
                        for channel in ("gas_class", "gas_concentration", "gas_trend"):
                            self._refute(self._channel(device, "gas", channel, loc, now),
                                         f"розходиться з {pair[1] if device == pair[0] else pair[0]} на {delta:.1f} %",
                                         now)
                        self.registry.set_health(device, {"twin_disagreement": {"detail": f"Δ {delta:.1f} %"}}, now,
                                                 reporter="twin")
                else:
                    for device in pair:
                        self.registry.set_health(device, {}, now, reporter="twin")
                    if now - self.twin_at.get(pair, float("-inf")) >= float(self.ev.get("twin_every_sec", 600)):
                        self.twin_at[pair] = now
                        for device in pair:
                            for channel in ("gas_class", "gas_concentration", "gas_trend"):
                                self._confirm(self._channel(device, "gas", channel, loc, now),
                                              "другий сенсор локації погоджується", now)
        if loc in self.smoking and (p.get("trend") == "rising" or float(p.get("rise_rate_pct_per_min") or 0) > 0):
            if now - self.smoking[loc] <= float(self.ev.get("smoking_gas_sec", 60)):
                for cam in self._devices("camera", loc):
                    self._confirm(f"{cam}/smoking", "газ зростає після підозри на куріння", now)
            del self.smoking[loc]

    def _selftest(self, sensor: str, location: Optional[str], ok: bool, detail: Optional[str], now: float) -> None:
        if ok:
            if now - self.selftest_at.get(sensor, float("-inf")) < float(self.ev.get("selftest_every_sec", 3600)):
                return
            self.selftest_at[sensor] = now
        for channel in ("gas_class", "gas_concentration", "gas_trend"):
            cid = self._channel(sensor, "gas", channel, location, now)
            if ok:
                self._confirm(cid, "самоперевірка чистого повітря пройдена", now, source="selftest")
            else:
                self._refute(cid, f"самоперевірка не пройдена ({detail})" if detail else "самоперевірка не пройдена",
                             now, source="selftest")

    # --- identity ------------------------------------------------------------------------------------------------

    def _identity(self, event: Event, now: float) -> None:
        p, t = event.payload, event.type
        registry_cid = self._channel(IDENTITY_DEVICE, "identity", "person_registry", None, now)
        persons = {x for x in (p.get("person_id"), p.get("source_id"), p.get("target_id")) if x}
        persons |= set(p.get("merged") or [])
        if t == "identity_created":
            if str(p.get("display_name") or "").startswith("unknown_"):
                self._unknown(p.get("person_id"), now)
            return
        if t == "identity_linked":
            mic = event.device_id if self.registry.devices.get(event.device_id, {}).get("kind") == "mic" else None
            mics = [mic] if mic else self._devices("mic", event.location_id)
            for m in mics:
                for person in persons:
                    self.person_mics.setdefault(person, set()).add(m)
                self._confirm(f"{m}/face_voice_link", "голос пов'язано з обличчям", now)
            return
        if t in CORRECTIONS:
            self.last_correction = now
            what = {"identity_merged": "злиття", "identity_unmerged": "розділення",
                    "identity_renamed": "перейменування", "identity_deleted": "видалення"}[t]
            self._refute(registry_cid, f"{what} людини", now)
            cameras = set().union(*(self.person_cameras.get(x, set()) for x in persons)) if persons else set()
            for cam in cameras:
                self._refute(f"{cam}/face_identity", f"{what} людини в реєстрі", now)
            if t in ("identity_merged", "identity_unmerged"):
                mics = set().union(*(self.person_mics.get(x, set()) for x in persons)) if persons else set()
                for m in mics:
                    self._refute(f"{m}/face_voice_link", f"{what} скасувало зв'язок голосу", now)

    def _unknown(self, person: Optional[str], now: float) -> None:
        self.unknown_created.append(now)
        while self.unknown_created and now - self.unknown_created[0] > 3600:
            self.unknown_created.popleft()
        limit = int(self.identity_cfg.get("unknown_per_hour", 10))
        storm = len(self.unknown_created) > limit
        for cam in self._devices("camera"):
            self.registry.set_health(cam, {"unknown_storm": {"value": len(self.unknown_created),
                                                             "detail": f"{len(self.unknown_created)} за годину"}}
                                     if storm else {}, now, reporter="identity")

    def _quiet_registry(self, now: float) -> None:
        days = float(self.ev.get("registry_quiet_days", 1))
        start = self.quiet_confirmed if self.quiet_confirmed is not None else self.last_correction
        if start is None:
            self.quiet_confirmed = now
            return
        if now - max(start, self.last_correction or start) >= days * 86400:
            self.quiet_confirmed = now
            self._confirm(self._channel(IDENTITY_DEVICE, "identity", "person_registry", None, now),
                          f"{days:g} доби без виправлень", now)

    # --- operator ------------------------------------------------------------------------------------------------

    def _resolved(self, p: Dict[str, Any], now: float) -> None:
        """F25: the operator accepted / rejected GPT's escalation → the llm channel + the rule-suggestion store"""
        review = self.reviews.pop(str(p.get("review_id")), {})
        accepted = bool(p.get("accepted"))
        cid = self._channel("llm", "llm", "risk_assessment", None, now)
        level = review.get("risk_level_llm") or p.get("risk_level_llm")
        who = p.get("by") or "operator"
        if accepted:
            self._confirm(cid, f"{who} прийняв ескалацію GPT до «{level}»", now, source="operator")
        else:
            self._refute(cid, f"{who} відхилив ескалацію GPT до «{level}»", now, source="operator")
        if self.suggestions is not None:
            self.suggestions.record_outcome(
                str(p.get("review_id")), accepted, now, who, review.get("location_id") or p.get("location_id"), level,
                review.get("risk_level_final"), p.get("reason"),
                {k: review.get(k) for k in ("summary", "reasons", "U", "h", "risk_llm", "inputs", "snapshot_id")})

    def _feedback(self, p: Dict[str, Any], now: float) -> None:
        cid = p.get("channel_id")
        if not cid or cid not in self.registry.channels:
            return
        verdict, reason = p.get("verdict"), p.get("reason") or "оцінка оператора"
        if p.get("original") is not None and p.get("corrected") is not None:    # transcript correction
            ratio = edit_ratio(str(p["original"]), str(p["corrected"]))
            verdict = "refute" if ratio > float(self.ev.get("transcript_edit_ratio", 0.2)) else "confirm"
            reason = f"оператор виправив транскрипт ({ratio:.0%} символів)"
        (self._confirm if verdict == "confirm" else self._refute)(cid, reason, now, source="operator")
