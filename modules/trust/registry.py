"""
Trust registry (F23 / TOD-34): trust per device × channel — calibrated prior, dynamic evidence, device health and a
Ukrainian explanation of every change, stored in SQLite (``data/trust_registry.db``; ``:memory:`` in replays)::

    r_t = γ·r_{t−1} + Δr,   s_t = γ·s_{t−1} + Δs,   τ_t = (r_t + 1)/(r_t + s_t + 2) · h_t,   u_t = 2/(r_t + s_t + 2)

* ``channel_id = <device_id>/<channel>``; channels per device kind, priors (r0, s0 from ``config/trust_priors.yaml``,
  TOD-33; none → ``prior_status: unmeasured``, r0 = s0 = 0), γ per channel kind (per day, applied lazily by elapsed
  time) and health factors in ``config/trust_registry.yaml``;
* health h ∈ [0, 1] = product of the factors of the failing checks a reporter (device self-diagnostics, the router)
  sent for the device (or only for listed channel kinds); a drop applies at once, a recovery at most
  ``health.recovery_per_min`` per minute → trust recovers gradually after a repair;
* ``channels`` (current state) + append-only ``history`` (r, s, h, τ, u, the cause and the explanation): any past τ
  is reproducible from the stored counters (τ = (r+1)/(r+s+2)·h).

Times are event times (seconds since the epoch), so a replay gives the same history as the live run.
"""
import json
import math
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEVICE_LABEL = {"camera": "камера", "mic": "мікрофон", "gas": "сенсор газу", "identity": "реєстр людей",
                "llm": "аналітик GPT"}
CHANNEL_LABEL = {"objects": "об'єкти", "motion": "рух", "pose_action": "пози / дії", "smoking": "куріння",
                 "face_identity": "обличчя", "scene_text": "текст у кадрі", "vad": "наявність мови",
                 "asr": "транскрипт", "speaker_identity": "голос", "face_voice_link": "зв'язок голос ↔ обличчя",
                 "gas_class": "тип газу", "gas_concentration": "концентрація", "gas_trend": "динаміка газу",
                 "person_registry": "реєстр людей", "risk_assessment": "ескалації GPT"}
KIND_ALIASES = {"microphone": "mic", "browser": "camera", "rtsp": "camera"}
DAY = 86400.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, device_kind TEXT NOT NULL, channel_kind TEXT NOT NULL,
    location_id TEXT, method TEXT, r REAL NOT NULL, s REAL NOT NULL, h REAL NOT NULL, tau REAL NOT NULL,
    u REAL NOT NULL, gamma REAL NOT NULL, prior_json TEXT NOT NULL, prior_status TEXT NOT NULL,
    state_json TEXT NOT NULL, updated_at REAL NOT NULL, explanation TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, channel_id TEXT NOT NULL, at REAL NOT NULL, r REAL NOT NULL,
    s REAL NOT NULL, h REAL NOT NULL, tau REAL NOT NULL, u REAL NOT NULL, cause_type TEXT NOT NULL,
    cause TEXT NOT NULL, explanation TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS history_channel ON history (channel_id, at);
"""


def plural_ua(n: int, one: str, few: str, many: str) -> str:
    d, dd = n % 10, n % 100
    return one if d == 1 and dd != 11 else few if 2 <= d <= 4 and not 12 <= dd <= 14 else many


def ua(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def duration_ua(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} с"
    if seconds < 5400:
        return f"{round(seconds / 60)} хв"
    if seconds < 2 * DAY:
        return f"{round(seconds / 3600)} год"
    return f"{round(seconds / DAY)} доби"


@dataclass
class Channel:
    channel_id: str
    device_id: str
    device_kind: str
    channel_kind: str
    location_id: Optional[str]
    r: float
    s: float
    gamma: float
    prior: Dict[str, Any]
    prior_status: str
    method: str = "beta_evidence"
    at: float = 0.0                                  # time r, s were last decayed / updated
    h: float = 1.0                                   # h at h_since
    h_since: float = 0.0
    h_target: float = 1.0
    reasons: List[Dict[str, Any]] = field(default_factory=list)   # failing checks applied to this channel
    stats: Dict[str, int] = field(default_factory=dict)            # confirm / refute counters by source
    explanation: str = ""
    previous_tau: Optional[float] = None

    def counts(self, now: float) -> tuple:
        days = max(0.0, now - self.at) / DAY
        factor = self.gamma ** days if days > 0 else 1.0
        return self.r * factor, self.s * factor

    def health(self, now: float, recovery_per_min: float) -> float:
        if self.h_target <= self.h:
            return self.h_target
        return min(self.h_target, self.h + recovery_per_min * max(0.0, now - self.h_since) / 60.0)

    def state_json(self, recovery_per_min: float = 0.1) -> str:
        return json.dumps({"at": self.at, "h_since": self.h_since, "h_target": self.h_target, "h_base": self.h,
                           "recovery_per_min": recovery_per_min,
                           "reasons": self.reasons, "stats": self.stats, "previous_tau": self.previous_tau},
                          ensure_ascii=False)


class TrustRegistry:
    def __init__(self, cfg: Dict[str, Any], path: str = ":memory:"):
        """
        :param cfg: Parsed ``config/trust_registry.yaml`` (load it with ``core.fuzzy.load_yaml`` — priors are refs)
        :param path: SQLite file (``data/trust_registry.db``) or ``:memory:``
        """
        self.cfg = cfg
        self.device_channels = {k: list(v) for k, v in (cfg.get("device_channels") or {}).items()}
        self.priors = cfg.get("priors") or {}
        self.gammas = {k: (v["value"] if isinstance(v, dict) else v) for k, v in (cfg.get("gamma") or {}).items()}
        health = cfg.get("health") or {}
        self.recovery = float(health.get("recovery_per_min", 0.1))
        self.checks = health.get("checks") or {}
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.channels: Dict[str, Channel] = {}
        self.devices: Dict[str, Dict[str, Any]] = {}                 # device_id → {kind, location_id}
        self.device_checks: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}  # device → reporter → check → info
        self._load()

    @classmethod
    def from_config(cls, path: str = "config/trust_registry.yaml", db_path: str = ":memory:") -> "TrustRegistry":
        from core.fuzzy import load_yaml

        return cls(load_yaml(path), db_path)

    # --- persistence ---------------------------------------------------------------------------------------------

    def _load(self) -> None:
        for row in self.db.execute("SELECT channel_id, device_id, device_kind, channel_kind, location_id, method, r, "
                                   "s, h, gamma, prior_json, prior_status, state_json, explanation FROM channels"):
            state = json.loads(row[12])
            channel = Channel(row[0], row[1], row[2], row[3], row[4], row[6], row[7], row[9], json.loads(row[10]),
                              row[11], row[5], state.get("at", 0.0), row[8], state.get("h_since", 0.0),
                              state.get("h_target", 1.0), state.get("reasons", []), state.get("stats", {}), row[13],
                              state.get("previous_tau"))
            self.channels[channel.channel_id] = channel
            self.devices.setdefault(channel.device_id, {"kind": channel.device_kind,
                                                        "location_id": channel.location_id})

    def _write(self, ch: Channel, now: float, cause_type: str, cause: str) -> None:
        r, s = ch.counts(now)                      # stored counters are as of `now` → τ reproducible from the row
        ch.r, ch.s, ch.at = r, s, max(ch.at, now)
        h = ch.health(now, self.recovery)
        tau, u = (r + 1) / (r + s + 2) * h, 2 / (r + s + 2)
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO channels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (ch.channel_id, ch.device_id, ch.device_kind, ch.channel_kind, ch.location_id, ch.method,
                             r, s, h, tau, u, ch.gamma, json.dumps(ch.prior, ensure_ascii=False), ch.prior_status,
                             ch.state_json(self.recovery), now, ch.explanation))
            self.db.execute("INSERT INTO history (channel_id, at, r, s, h, tau, u, cause_type, cause, explanation) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?)", (ch.channel_id, now, r, s, h, tau, u, cause_type, cause,
                                                             ch.explanation))

    def close(self) -> None:
        self.db.close()

    # --- channels ------------------------------------------------------------------------------------------------

    def kind_of(self, kind: Optional[str]) -> Optional[str]:
        kind = KIND_ALIASES.get(kind or "", kind)
        return kind if kind in self.device_channels else None

    def prior_for(self, channel_kind: str) -> Dict[str, Any]:
        entry = self.priors.get(channel_kind)
        if isinstance(entry, dict) and entry.get("prior_status") == "measured":
            return {"prior_status": "measured", "r0": float(entry["r0"]), "s0": float(entry["s0"]),
                    "tau0": entry.get("tau0"), "u0": entry.get("u0"), "wilson_lb": entry.get("wilson_lb"),
                    "source": entry.get("source"), "explanation": entry.get("explanation")}
        return {"prior_status": "unmeasured", "r0": 0.0, "s0": 0.0, "tau0": 0.5, "u0": 1.0, "source": None,
                "explanation": "апріорі не виміряно (немає валідаційних даних)"}

    def ensure_device(self, device_id: str, kind: Optional[str], location_id: Optional[str] = None,
                      now: float = 0.0) -> List[str]:
        """Creates every channel of the device kind (idempotent); a new location is recorded. :return: channel ids"""
        kind = self.kind_of(kind)
        if not device_id or kind is None:
            return []
        info = self.devices.setdefault(device_id, {"kind": kind, "location_id": location_id})
        if location_id and info.get("location_id") != location_id:
            info["location_id"] = location_id
            for ch in self.device(device_id):
                ch.location_id = location_id
                self._write(ch, now, "device", f"location → {location_id}")
        return [self.ensure_channel(device_id, kind, channel_kind, location_id, now)
                for channel_kind in self.device_channels[kind]]

    def ensure_channel(self, device_id: str, device_kind: str, channel_kind: str, location_id: Optional[str] = None,
                       now: float = 0.0) -> str:
        channel_id = f"{device_id}/{channel_kind}"
        if channel_id in self.channels:
            return channel_id
        prior = self.prior_for(channel_kind)
        ch = Channel(channel_id, device_id, device_kind, channel_kind, location_id, prior["r0"], prior["s0"],
                     float(self.gammas.get(channel_kind, 1.0)), prior, prior["prior_status"], at=now, h_since=now)
        self.devices.setdefault(device_id, {"kind": device_kind, "location_id": location_id})
        self.channels[channel_id] = ch
        self._apply_health(ch, now)
        ch.explanation = self.explain(ch, now, None)
        self._write(ch, now, "created", f"prior {prior['prior_status']}")
        return channel_id

    def device(self, device_id: str) -> List[Channel]:
        return [c for c in self.channels.values() if c.device_id == device_id]

    def get(self, channel_id: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        ch = self.channels.get(channel_id)
        if ch is None:
            return None
        now = ch.at if now is None else now
        r, s = ch.counts(now)
        h = ch.health(now, self.recovery)
        return {"channel_id": ch.channel_id, "device_id": ch.device_id, "device_kind": ch.device_kind,
                "channel_kind": ch.channel_kind, "location_id": ch.location_id, "method": ch.method,
                "r": round(r, 4), "s": round(s, 4), "h": round(h, 4), "tau": round((r + 1) / (r + s + 2) * h, 4),
                "u": round(2 / (r + s + 2), 4), "gamma": ch.gamma, "prior": ch.prior, "prior_status": ch.prior_status,
                "reasons": ch.reasons, "stats": ch.stats, "updated_at": ch.at, "explanation": ch.explanation}

    def uncertainty(self, channel_id: str, now: float) -> Optional[float]:
        """
        u of the channel discounted by its health (subjective-logic discounting of the opinion by h):
        u_eff = 1 − h·(1 − u) — a healthy channel keeps its count uncertainty, a failed one (h = 0) knows nothing.
        F24: widens the FOU of the inputs this channel feeds.
        """
        ch = self.channels.get(channel_id)
        if ch is None:
            return None
        r, s = ch.counts(now)
        return 1.0 - ch.health(now, self.recovery) * (1.0 - 2 / (r + s + 2))

    def tau(self, channel_id: str, now: float) -> Optional[float]:
        ch = self.channels.get(channel_id)
        if ch is None:
            return None
        r, s = ch.counts(now)
        return (r + 1) / (r + s + 2) * ch.health(now, self.recovery)

    # --- evidence ------------------------------------------------------------------------------------------------

    def update(self, channel_id: str, dr: float, ds: float, reason: str, now: float, source: str = "evidence") -> bool:
        """Δr / Δs on one channel (after forgetting since its last update). :return: False for an unknown channel"""
        ch = self.channels.get(channel_id)
        if ch is None:
            return False
        before = self.tau(channel_id, now)
        r, s = ch.counts(now)
        ch.r, ch.s, ch.at = r + dr, s + ds, now
        key = f"{source}_{'confirm' if dr > 0 else 'refute'}"
        ch.stats[key] = ch.stats.get(key, 0) + int(round(max(dr, ds)))
        ch.previous_tau = before
        ch.explanation = self.explain(ch, now, {"source": source, "dr": dr, "ds": ds, "reason": reason})
        self._write(ch, now, source, f"{'+' if dr > 0 else '−'}{max(dr, ds):g}: {reason}")
        return True

    def confirm(self, channel_id: str, n: float = 1.0, reason: str = "", now: float = 0.0,
                source: str = "evidence") -> bool:
        return self.update(channel_id, float(n), 0.0, reason, now, source)

    def refute(self, channel_id: str, n: float = 1.0, reason: str = "", now: float = 0.0,
               source: str = "evidence") -> bool:
        return self.update(channel_id, 0.0, float(n), reason, now, source)

    # --- health --------------------------------------------------------------------------------------------------

    def set_health(self, device_id: str, failing: Dict[str, Dict[str, Any]], now: float,
                   reporter: str = "device") -> List[str]:
        """
        Replaces the failing checks one reporter knows for a device (``{}`` = all its checks pass).

        :param failing: check name → {"value": …, "detail": "…", "since": epoch, "channels": [...]} (factor, reason
            and default channels from ``health.checks``; a "factor" given here overrides the configured one)
        :return: channel ids whose health changed
        """
        current = self.device_checks.setdefault(device_id, {})
        old = current.get(reporter, {})
        merged = {}
        for name, info in (failing or {}).items():
            spec = self.checks.get(name, {})
            since = old.get(name, {}).get("since", info.get("since", now))
            merged[name] = {"check": name, "factor": float(info.get("factor", spec.get("factor", 0.5))),
                            "reason": spec.get("reason", name), "detail": info.get("detail"),
                            "value": info.get("value"), "since": since,
                            "channels": info.get("channels", spec.get("channels"))}
        if {k: (v["factor"], v["detail"]) for k, v in merged.items()} == \
                {k: (v["factor"], v.get("detail")) for k, v in old.items()}:
            current[reporter] = merged
            return []
        current[reporter] = merged
        changed = []
        for ch in self.device(device_id):
            before = self.tau(ch.channel_id, now)
            if self._apply_health(ch, now):
                ch.previous_tau = before
                ch.explanation = self.explain(ch, now, None)
                names = ", ".join(sorted(merged)) or "усі перевірки пройдено"
                self._write(ch, now, "health", f"{reporter}: {names}")
                changed.append(ch.channel_id)
        return changed

    def _apply_health(self, ch: Channel, now: float) -> bool:
        reasons = [info for checks in self.device_checks.get(ch.device_id, {}).values() for info in checks.values()
                   if not info.get("channels") or ch.channel_kind in info["channels"]]
        target = math.prod(r["factor"] for r in reasons) if reasons else 1.0
        current = ch.health(now, self.recovery)
        if abs(target - ch.h_target) < 1e-9 and [r["check"] for r in reasons] == [r["check"] for r in ch.reasons]:
            return False
        ch.h, ch.h_since, ch.h_target = current, now, target
        ch.reasons = sorted(reasons, key=lambda r: r["factor"])
        return True

    # --- explanations ----------------------------------------------------------------------------------------------

    def explain(self, ch: Channel, now: float, cause: Optional[Dict[str, Any]]) -> str:
        r, s = ch.counts(now)
        h = ch.health(now, self.recovery)
        tau = (r + 1) / (r + s + 2) * h
        head = f"Довіра {ua(tau)}: {DEVICE_LABEL.get(ch.device_kind, ch.device_kind)} {ch.device_id}"
        parts: List[str] = []
        for reason in ch.reasons:
            text = reason["reason"]
            if reason.get("detail"):
                text += f" ({reason['detail']})"
            if reason.get("since") is not None and now - reason["since"] >= 1:
                text += f" {duration_ua(now - reason['since'])}"
            parts.append(text)
        if not ch.reasons and h < ch.h_target - 1e-6:
            parts.append("перевірки знову пройдено")
        if parts or h < 0.999:
            parts.append(f"h = {ua(h)}" + (f" (відновлюється до {ua(ch.h_target)})" if h < ch.h_target - 1e-6 else ""))
        operator = ch.stats.get("operator_refute", 0)
        if operator:
            noun = {"asr": "транскриптів виправлено", "pose_action": "дій відхилено",
                    "face_identity": "ідентифікацій виправлено"}.get(ch.channel_kind, "спростувань")
            parts.append(f"оператор: {operator} {noun}")
        selftest = ch.stats.get("selftest_refute", 0)
        if selftest:
            parts.append(f"{selftest} {plural_ua(selftest, 'самоперевірку', 'самоперевірки', 'самоперевірок')} "
                         "не пройдено")
        if cause:
            sign = f"+{cause['dr']:g} підтвердження" if cause["dr"] > 0 else f"−{cause['ds']:g} спростування"
            parts.append(f"{sign}: {cause['reason']}" if cause.get("reason") else sign)
        if ch.prior_status == "unmeasured" and not parts:
            return (f"Довіра {ua(tau)}: {CHANNEL_LABEL.get(ch.channel_kind, ch.channel_kind)} ({ch.device_id}) — "
                    "апріорі не виміряно (немає валідаційних даних); значення уточнюватиметься з підтверджень.")
        if not parts:
            source = ch.prior.get("explanation") or "виміряна апріорна оцінка"
            return f"{head} — {CHANNEL_LABEL.get(ch.channel_kind, ch.channel_kind)}: апріорі {source}."
        tail = f"; до цього {ua(ch.previous_tau)}" if ch.previous_tau is not None else ""
        return f"{head} — {CHANNEL_LABEL.get(ch.channel_kind, ch.channel_kind)}: {'; '.join(parts)}{tail}."

    # --- maintenance / export ------------------------------------------------------------------------------------

    def tick(self, now: float) -> None:
        """Re-evaluates gradual health recovery (writes a history row when a recovering channel reaches its target)."""
        for ch in self.channels.values():
            if ch.h < ch.h_target and ch.health(now, self.recovery) >= ch.h_target - 1e-9:
                ch.h, ch.h_since = ch.h_target, now
                ch.explanation = self.explain(ch, now, None)
                self._write(ch, now, "health", "recovered")

    def snapshot(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        return [self.get(cid, now) for cid in sorted(self.channels)]

    def history(self, channel_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        rows = self.db.execute("SELECT at, r, s, h, tau, u, cause_type, cause, explanation FROM history "
                               "WHERE channel_id = ? ORDER BY id DESC LIMIT ?", (channel_id, limit)).fetchall()
        keys = ("at", "r", "s", "h", "tau", "u", "cause_type", "cause", "explanation")
        return [dict(zip(keys, row, strict=True)) for row in reversed(rows)]

    def config_snapshot(self) -> Dict[str, Any]:
        """What an experiment manifest stores: channels per kind, priors used, γ, health factors."""
        return {"device_channels": self.device_channels,
                "priors": {k: self.prior_for(k) for kinds in self.device_channels.values() for k in kinds},
                "gamma": self.cfg.get("gamma"), "health": self.cfg.get("health")}


# --- read-only access (dashboard process; the router owns the writes) -----------------------------------------------

def _readonly(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)


def read_channels(path: str, now: Optional[float] = None) -> List[Dict[str, Any]]:
    """:return: Every channel row of a registry file, τ / h / r / s evaluated at ``now`` (forgetting + recovery)"""
    import time

    now = time.time() if now is None else now
    db = _readonly(path)
    try:
        cols = ("channel_id", "device_id", "device_kind", "channel_kind", "location_id", "method", "r", "s", "h", "tau",
                "u", "gamma", "prior_json", "prior_status", "state_json", "updated_at", "explanation")
        rows = []
        for row in db.execute(f"SELECT {', '.join(cols)} FROM channels ORDER BY channel_id"):
            item = dict(zip(cols, row, strict=True))
            state = json.loads(item.pop("state_json"))
            item["prior"] = json.loads(item.pop("prior_json"))
            item.update(reasons=state.get("reasons", []), stats=state.get("stats", {}),
                        h_target=state.get("h_target", item["h"]), previous_tau=state.get("previous_tau"))
            at = float(state.get("at", item["updated_at"]))
            factor = item["gamma"] ** (max(0.0, now - at) / DAY)
            r, s = item["r"] * factor, item["s"] * factor
            h, target = float(state.get("h_base", item["h"])), item["h_target"]
            if target > h:
                h = min(target, h + float(state.get("recovery_per_min", 0.1)) * max(0.0, now - state.get(
                    "h_since", at)) / 60.0)
            else:
                h = target
            item.update(r=round(r, 4), s=round(s, 4), h=round(h, 4), tau=round((r + 1) / (r + s + 2) * h, 4),
                        u=round(2 / (r + s + 2), 4))
            rows.append(item)
        return rows
    finally:
        db.close()


def read_history(path: str, channel_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    db = _readonly(path)
    try:
        rows = db.execute("SELECT at, r, s, h, tau, u, cause_type, cause, explanation FROM history "
                          "WHERE channel_id = ? ORDER BY id DESC LIMIT ?", (channel_id, limit)).fetchall()
    finally:
        db.close()
    keys = ("at", "r", "s", "h", "tau", "u", "cause_type", "cause", "explanation")
    return [dict(zip(keys, row, strict=True)) for row in reversed(rows)]

