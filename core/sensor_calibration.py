"""
Persistent sensor calibration (C1 / TOD-54): calibrate once per device, keep it across restarts, run only a short check
on start. See ``docs/calibration.md``.

* ``device_uid`` — the device's identity: driver + serial / hardware id + bus address + firmware (TOD-48 descriptor);
  cameras and microphones use their stable registry ``device_id`` (kind + node + hardware key). A new uid (sensor
  replaced, firmware updated) has no record → full calibration.
* ``CalibrationStore`` — one JSON per device in ``sensor_calibration.dir`` (``data/calibration/devices/<uid>.json``,
  git-ignored, survives restarts and updates): ``{uid, kind, device_id, values, method, date, environment,
  validation, history: [previous records]}``; atomic writes; ``restore`` brings back a previous version; a corrupt
  file is moved aside (``.corrupt-<time>``) and logged — only that device recalibrates.
* ``DeviceCalibration`` — the state of one device in its worker::

      start ─► stored record, not too old ─► [warming_up N min] ─► checking ─► ready (stored, date)
         │                                                         └ fail ─► needs_check (+ wider uncertainty)
         │                                                                    └ max_failures in a row ─► calibrating
         ├─► factory-calibrated kind ─► [warming_up] ─► factory
         └─► no / corrupt / too old record ─► calibrating (progress, cancel) ─► ready (new)
      manual request (dashboard) ─► calibrating;  cancel ─► back to the stored record (or uncalibrated)

  Warm-up is not calibration: the stored values are used, with ``uncertainty_factor`` (wider δ, TOD-45) until the
  heater / stabilisation time has passed. Every change is published as ``calibration_status`` (Ukrainian ``text``,
  source stored / factory / new, date, age, last check, failures, progress, ``ready_after_sec``).

The worker owns the measurement (what a short check or a full calibration is for its sensor); this module owns the
policy (``sensor_calibration.kinds`` in config.yaml) and the files.
"""
import copy
import datetime as dt
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from system_setup.logger_setup import init_logger

logger = init_logger()

DEFAULT_DIR = os.path.join("data", "calibration", "devices")
HISTORY = 10
READY, FACTORY, WARMING_UP, CHECKING, NEEDS_CHECK, CALIBRATING, UNCALIBRATED = (
    "ready", "factory", "warming_up", "checking", "needs_check", "calibrating", "uncalibrated")
USABLE = (READY, FACTORY)                   # the device is fully protected
DEFAULT_POLICY = {"max_age_days": 30, "warmup_sec": 0, "max_failures": 3, "factory": False, "recheck_sec": 60,
                  "refresh_sec": 300}         # refresh: the status is repeated this often (a restarted dashboard)
INSTRUCTIONS_UK = {
    "graphene_fet": "Забезпечте чисте повітря біля сенсора (провітріть, приберіть розчинники) на час калібрування.",
    "mos": "Забезпечте чисте повітря; сенсор має бути прогрітий.",
    "mic": "Забезпечте звичайну тишу в приміщенні: без розмов і гучних звуків на час калібрування.",
    "camera": "Приберіть людей і рухомі предмети з кадру на час калібрування; світло — як зазвичай.",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def device_uid(driver: str, hw_id: Optional[str] = None, bus: Optional[str] = None,
               firmware: Optional[str] = None) -> str:
    """:return: a file-safe uid, readable prefix + hash of the identity (``graphene_fet-1a2b3c4d5e6f``)"""
    identity = json.dumps([driver, hw_id, bus, firmware])
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return f"{re.sub(r'[^A-Za-z0-9_]+', '_', str(driver))[:24]}-{digest}"


def parse_iso(value: Optional[str]) -> Optional[float]:
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


class CalibrationStore:
    """The calibration records of every device, one JSON file each."""

    def __init__(self, directory: Optional[str] = DEFAULT_DIR, history: int = HISTORY):
        """:param directory: None = in memory only (tests, replays)"""
        self.dir, self.history_max = directory, int(history)
        self.memory: Dict[str, Dict[str, Any]] = {}
        self.corrupt: Dict[str, str] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CalibrationStore":
        cfg = config.get("sensor_calibration") or {}
        return cls(cfg.get("dir", DEFAULT_DIR), cfg.get("history", HISTORY))

    def path(self, uid: str) -> Optional[str]:
        return None if self.dir is None else os.path.join(self.dir, f"{uid}.json")

    def load(self, uid: str) -> Optional[Dict[str, Any]]:
        """:return: the record or None (missing; a corrupt file is moved aside and logged)"""
        path = self.path(uid)
        if path is None:
            return copy.deepcopy(self.memory.get(uid))
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            if not isinstance(record, dict) or not isinstance(record.get("values"), dict):
                raise ValueError("no values")
            return record
        except (OSError, ValueError) as e:
            aside = f"{path}.corrupt-{int(time.time())}"
            try:
                os.replace(path, aside)
            except OSError:
                aside = path
            self.corrupt[uid] = f"{e} (moved to {os.path.basename(aside)})"
            logger.error(f"calibration {uid}: unreadable file ({e}) — kept as {aside}; this device recalibrates")
            return None

    def _write(self, uid: str, record: Dict[str, Any]) -> None:
        path = self.path(uid)
        if path is None:
            self.memory[uid] = copy.deepcopy(record)
            return
        os.makedirs(self.dir, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    def save(self, uid: str, kind: str, device_id: Optional[str], values: Dict[str, Any], method: str,
             environment: Optional[Dict[str, Any]] = None, validation: Optional[Dict[str, Any]] = None,
             by: str = "system") -> Dict[str, Any]:
        """A new calibration; the current one moves to the history (newest first). :return: the record"""
        previous = self.load(uid)
        history = []
        if previous is not None:
            history = [{k: v for k, v in previous.items() if k != "history"}] + list(previous.get("history") or [])
        record = {"uid": uid, "kind": kind, "device_id": device_id, "values": values, "method": method,
                  "date": now_iso(), "environment": environment or {}, "validation": validation or {}, "by": by,
                  "history": history[:self.history_max]}
        self._write(uid, record)
        return record

    def update_validation(self, uid: str, validation: Dict[str, Any]) -> None:
        """The last short check of the stored calibration (no new version)."""
        record = self.load(uid)
        if record is not None:
            record["validation"] = {**(record.get("validation") or {}), **validation}
            self._write(uid, record)

    def restore(self, uid: str, index: int = 0, by: str = "dashboard") -> Dict[str, Any]:
        """
        Makes the ``index``-th previous version current again (the current one goes to the history).
        :raise KeyError: no such version
        """
        record = self.load(uid)
        history = list((record or {}).get("history") or [])
        if record is None or not 0 <= index < len(history):
            raise KeyError(f"{uid}: no previous calibration #{index}")
        chosen = history.pop(index)
        current = {k: v for k, v in record.items() if k != "history"}
        restored = {**chosen, "restored_at": now_iso(), "restored_by": by,
                    "history": ([current] + history)[:self.history_max]}
        self._write(uid, restored)
        return restored

    def records(self) -> List[Dict[str, Any]]:
        """:return: every stored record without its history values (dashboard)"""
        if self.dir is None:
            uids = sorted(self.memory)
        elif os.path.isdir(self.dir):
            uids = sorted(n[:-5] for n in os.listdir(self.dir) if n.endswith(".json"))
        else:
            uids = []
        out = []
        for uid in uids:
            record = self.load(uid)
            if record is not None:
                out.append(summary(record))
        return out


def summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """:return: a record for display: no bulky values, its history as dates / methods"""
    def short(r: Dict[str, Any]) -> Dict[str, Any]:
        return {k: r.get(k) for k in ("date", "method", "by", "validation", "environment", "restored_at")
                if r.get(k) is not None}
    return {"uid": record.get("uid"), "kind": record.get("kind"), "device_id": record.get("device_id"),
            **short(record), "values": {k: v for k, v in (record.get("values") or {}).items()
                                        if isinstance(v, (int, float, str, bool))},
            "history": [short(h) for h in record.get("history") or []]}


def policy_of(config: Dict[str, Any], kind: str) -> Dict[str, Any]:
    cfg = config.get("sensor_calibration") or {}
    return {**DEFAULT_POLICY, **{k: cfg[k] for k in ("warmup_uncertainty_factor", "needs_check_uncertainty_factor")
                                 if k in cfg}, **((cfg.get("kinds") or {}).get(kind) or {})}


def age_days(record: Optional[Dict[str, Any]], now: Optional[float] = None) -> Optional[float]:
    at = parse_iso((record or {}).get("date"))
    return None if at is None else max(0.0, ((now or time.time()) - at) / 86400.0)


class DeviceCalibration:
    """The calibration state of one device (inside its worker). The worker measures; this decides and stores."""

    def __init__(self, store: CalibrationStore, uid: str, kind: str, device_id: Optional[str],
                 policy: Optional[Dict[str, Any]] = None, emit: Callable[[Dict[str, Any]], Any] = lambda p: None,
                 clock: Callable[[], float] = time.time, wall: Callable[[], float] = time.time):
        """
        :param emit: publishes a ``calibration_status`` payload (called on every change)
        :param clock: monotonic-ish time of the worker (warm-up, ready time); ``wall``: dates and ages
        """
        self.store, self.uid, self.kind, self.device_id = store, uid, kind, device_id
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.emit, self.clock, self.wall = emit, clock, wall
        self.record: Optional[Dict[str, Any]] = None
        self.state, self.reason, self.source = UNCALIBRATED, "", "new"
        self.failures, self.progress = 0, 0.0
        self.last_check: Optional[Dict[str, Any]] = None
        self.started_at: Optional[float] = None
        self.warm_until: Optional[float] = None
        self.ready_after: Optional[float] = None
        self.checked_at: Optional[float] = None
        self.emitted_at = float("-inf")
        self._before_calibration: Optional[str] = None
        self._pending: Optional[str] = None          # a full calibration due after the warm-up (cold MOS heater)

    # --- decisions -------------------------------------------------------------------------------------------------

    def start(self) -> str:
        """Called once when the device is identified. :return: the new state"""
        self.started_at = self.clock()
        warm = float(self.policy.get("warmup_sec") or 0)
        self.warm_until = self.started_at + warm if warm > 0 else None
        if self.policy.get("factory"):
            self.source = "factory"
            return self._set(WARMING_UP if self.warm_until else FACTORY, "заводська калібровка")
        self.record = self.store.load(self.uid)
        corrupt = self.store.corrupt.pop(self.uid, None)
        due = None
        if self.record is None:
            due = f"файл калібровки пошкоджений: {corrupt}" if corrupt \
                else "нова калібровка: пристрій без збереженої калібровки"
        else:
            age = age_days(self.record, self.wall())
            if age is not None and age > float(self.policy["max_age_days"]):
                due = f"калібровка застаріла: {age:.0f} дн. > {self.policy['max_age_days']} дн."
        if due is not None:
            if self.warm_until:                      # calibrate a warm sensor, not a cold one
                self._pending = due
                return self._set(WARMING_UP, f"прогрів перед калібруванням ({due})")
            return self.begin_calibration(due)
        self.source = "stored"
        return self._set(WARMING_UP if self.warm_until else CHECKING, "збережена калібровка")

    def tick(self) -> str:
        """Ends the warm-up; repeats a failed check after ``recheck_sec``; re-publishes the status every
        ``refresh_sec``. :return: the state"""
        if self.clock() - self.emitted_at >= float(self.policy["refresh_sec"]) and self.started_at is not None:
            self._publish()
        if self.state == WARMING_UP and self.warm_until is not None and self.clock() >= self.warm_until:
            self.warm_until = None
            if self._pending:
                due, self._pending = self._pending, None
                return self.begin_calibration(due)
            return self._set(FACTORY if self.source == "factory" else CHECKING, "прогрів завершено")
        if self.state == NEEDS_CHECK and self.checked_at is not None \
                and self.clock() - self.checked_at >= float(self.policy["recheck_sec"]):
            return self._set(CHECKING, f"повторна перевірка ({self.failures}/{self.policy['max_failures']} невдалих)")
        return self.state

    def check_result(self, ok: bool, detail: str = "", **measured) -> str:
        """The worker's short check of the stored calibration."""
        self.last_check = {"ok": bool(ok), "detail": detail, "at": now_iso(), **measured}
        self.checked_at = self.clock()
        self.store.update_validation(self.uid, {"last_check": self.last_check})
        if ok:
            self.failures = 0
            return self._set(READY, "коротка перевірка пройдена")
        self.failures += 1
        if self.failures >= int(self.policy["max_failures"]):
            return self.begin_calibration(f"перевірка не пройдена {self.failures} рази поспіль: {detail}")
        return self._set(NEEDS_CHECK, f"перевірка не пройдена ({self.failures}/{self.policy['max_failures']}): "
                                      f"{detail}")

    def begin_calibration(self, reason: str) -> str:
        if self.state != CALIBRATING:
            self._before_calibration = self.state
        self.progress = 0.0
        return self._set(CALIBRATING, reason)

    def request(self, by: str = "dashboard") -> str:
        """Manual recalibration (dashboard)."""
        return self.begin_calibration(f"ручне перекалібрування ({by})")

    def cancel(self) -> str:
        if self.state != CALIBRATING:
            return self.state
        self.progress = 0.0
        if self.record is None:
            return self._set(UNCALIBRATED, "калібрування скасовано; збереженої калібровки немає")
        before = self._before_calibration           # a manual request on a ready device goes back to ready
        return self._set(before if before in (READY, CHECKING, NEEDS_CHECK) else NEEDS_CHECK,
                         "калібрування скасовано: використовується збережена калібровка")

    def set_progress(self, share: float) -> None:
        share = round(max(0.0, min(1.0, float(share))), 2)
        if self.state == CALIBRATING and share - self.progress >= 0.1:
            self.progress = share
            self._publish()

    def complete(self, values: Dict[str, Any], method: str, environment: Optional[Dict[str, Any]] = None,
                 validation: Optional[Dict[str, Any]] = None, by: str = "system") -> Dict[str, Any]:
        """A full calibration finished: stored as the new version. :return: the record"""
        self.record = self.store.save(self.uid, self.kind, self.device_id, values, method, environment,
                                      validation, by)
        self.failures, self.progress, self.source = 0, 1.0, "new"
        self._set(READY, "калібрування завершено")
        return self.record

    def reload(self) -> str:
        """The file changed outside the worker (restore on the dashboard): use it and check it."""
        self.record = self.store.load(self.uid)
        if self.record is None:
            return self.begin_calibration("збережена калібровка зникла")
        self.source, self.failures = "stored", 0
        return self._set(CHECKING, "відновлено попередню калібровку")

    # --- what the worker needs ---------------------------------------------------------------------------------------

    @property
    def values(self) -> Dict[str, Any]:
        return dict((self.record or {}).get("values") or {})

    @property
    def usable(self) -> bool:
        return self.state in USABLE

    def uncertainty_factor(self) -> float:
        """δ multiplier of this device's readings (TOD-45): wider while warming up / unchecked / failing"""
        if self.state == WARMING_UP:
            return float(self.policy.get("warmup_uncertainty_factor", 2.0))
        if self.state in (NEEDS_CHECK, CALIBRATING, UNCALIBRATED, CHECKING):
            return float(self.policy.get("needs_check_uncertainty_factor", 2.0))
        return 1.0

    def status(self) -> Dict[str, Any]:
        record = self.record or {}
        age = age_days(record, self.wall()) if record else None
        out = {"device_id": self.device_id, "uid": self.uid, "kind": self.kind, "status": self.state,
               "text": self.text(), "source": self.source, "reason": self.reason, "date": record.get("date"),
               "age_days": None if age is None else round(age, 1), "last_check": self.last_check,
               "failures": self.failures, "max_failures": int(self.policy["max_failures"]),
               "uncertainty_factor": self.uncertainty_factor(), "versions": 1 + len(record.get("history") or [])
               if record else 0, "instructions": INSTRUCTIONS_UK.get(self.kind)}
        if self.state == CALIBRATING:
            out["progress"] = self.progress
        if self.warm_until is not None and self.state == WARMING_UP:
            out["warmup_left_sec"] = round(max(0.0, self.warm_until - self.clock()), 1)
        if self.ready_after is not None:
            out["ready_after_sec"] = self.ready_after
        return out

    def text(self) -> str:
        date = str((self.record or {}).get("date") or "")[:10]
        if self.state == READY:
            return f"відкалібровано ({'збережено' if self.source == 'stored' else 'нова калібровка'}, {date})"
        if self.state == FACTORY:
            return "заводська калібровка"
        if self.state == WARMING_UP:
            left = max(0.0, (self.warm_until or 0) - self.clock()) / 60
            return f"прогрів, {left:.0f} хв" if left >= 1 else "прогрів, < 1 хв"
        if self.state == CHECKING:
            return f"коротка перевірка (збережено, {date})"
        if self.state == NEEDS_CHECK:
            return "потребує перевірки"
        if self.state == CALIBRATING:
            return f"калібрування… {self.progress * 100:.0f} %"
        return "не відкалібровано"

    def _set(self, state: str, reason: str) -> str:
        changed = state != self.state or reason != self.reason
        self.state, self.reason = state, reason
        if state in USABLE and self.ready_after is None and self.started_at is not None:
            self.ready_after = round(self.clock() - self.started_at, 2)
            logger.info(f"calibration {self.device_id or self.uid}: {self.text()} after {self.ready_after} s")
        if changed:
            self._publish()
        return state

    def _publish(self) -> None:
        self.emitted_at = self.clock()
        self.emit(self.status())
