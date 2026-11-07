"""
Operating mode (A1 / TOD-41, M1 / TOD-52): ``autonomous`` (Article 1) or ``multiagent`` (Article 2). See
``docs/modes.md``.

* ``autonomous`` — every decision is made locally by the hierarchical fuzzy system (``modules/hfs``): no agents, no
  GPT. Started this way, the LLM worker, GPT gating, the risk envelope and GPT rule suggestions are not started (nor
  imported), the aggregator queues nothing for GPT, Whisper runs locally — no cloud calls.
* ``multiagent`` — the fuzzy system runs as the fuzzy agent, the enabled AI agents add claims and the validation agent
  makes the final decision, always within the fuzzy corridor. The GPT pipeline runs too, but only logs its
  ``llm_decision`` (it does not decide). ``hybrid`` (TOD-41's name) is read as ``multiagent``.

The fuzzy system always runs, so a switch never leaves a location without a decision.

Mode at start (``start_mode``): ``python main.py --mode …`` / env ``IOT_MODE`` → the value saved from the dashboard
(``mode_store.state_path``, ``data/mode.json``) → ``config.yaml`` ``mode`` (shipped: ``autonomous``). A config without
the key (library use, tests) means ``multiagent`` — TOD-41's default ``hybrid``. At runtime the dashboard
switches it (``mode_control`` event → the orchestrator → ``mode_changed`` to every module); every switch is saved and
appended to the audit log (``mode.audit_path``). Optional per-location overrides (``locations`` of the saved state).
"""
import copy
import datetime as dt
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from system_setup.logger_setup import init_logger

logger = init_logger()

AUTONOMOUS, MULTIAGENT = "autonomous", "multiagent"
HYBRID = "hybrid"                         # TOD-41 name of multiagent, still accepted
MODES = (AUTONOMOUS, MULTIAGENT)
ALIASES = {HYBRID: MULTIAGENT}
LABEL_UK = {AUTONOMOUS: "Автономний (стаття 1)", MULTIAGENT: "Мультиагентний (стаття 2)"}
ENV = "IOT_MODE"
# Modules that talk to GPT; never imported when started autonomous (tests/test_hfs.py checks sys.modules)
LLM_PACKAGES = ("modules.llm", "modules.envelope", "modules.rules", "openai")


def normalise(mode: Any) -> str:
    """:return: ``autonomous`` / ``multiagent`` (``hybrid`` → multiagent); :raise ValueError: anything else"""
    value = str(mode or "").strip().lower()
    value = ALIASES.get(value, value)
    if value not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return value


def mode_of(config: Dict[str, Any]) -> str:
    """:return: the global mode of a config (missing = multiagent, TOD-41's default hybrid)"""
    return normalise(config.get("mode") or MULTIAGENT)


def is_autonomous(config: Dict[str, Any]) -> bool:
    return mode_of(config) == AUTONOMOUS


def location_mode(config: Dict[str, Any], location_id: Optional[str]) -> str:
    """:return: the requested mode of a location (its override, else the global mode)"""
    override = (config.get("mode_locations") or {}).get(location_id or "")
    return normalise(override) if override else mode_of(config)


def any_multiagent(config: Dict[str, Any]) -> bool:
    return mode_of(config) == MULTIAGENT or MULTIAGENT in {normalise(m) for m in
                                                          (config.get("mode_locations") or {}).values()}


class ModeStore:
    """The mode chosen on the dashboard (survives restarts) and the append-only audit log of every switch."""

    def __init__(self, state_path: Optional[str] = "data/mode.json",
                 audit_path: Optional[str] = "data/mode_audit.jsonl"):
        self.state_path, self.audit_path = state_path, audit_path

    def load(self) -> Optional[Dict[str, Any]]:
        """:return: {mode, locations, by, at} or None (nothing saved / unreadable)"""
        if not self.state_path or not os.path.isfile(self.state_path):
            return None
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f) or {}
            return {**data, "mode": normalise(data["mode"]),
                    "locations": {k: normalise(v) for k, v in (data.get("locations") or {}).items()}}
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"mode state {self.state_path} unreadable ({e}) — ignored")
            return None

    def save(self, mode: str, locations: Mapping[str, str], by: str) -> Dict[str, Any]:
        state = {"mode": normalise(mode), "locations": {k: normalise(v) for k, v in locations.items()}, "by": by,
                 "at": now_iso()}
        if self.state_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.state_path)
        return state

    def audit(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        entry = {"at": now_iso(), **entry}
        if self.audit_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.audit_path)), exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.audit_path or not os.path.isfile(self.audit_path):
            return []
        with open(self.audit_path, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def store_of(config: Dict[str, Any]) -> ModeStore:
    cfg = config.get("mode_store") or {}
    return ModeStore(cfg.get("state_path", "data/mode.json"), cfg.get("audit_path", "data/mode_audit.jsonl"))


def start_mode(config: Dict[str, Any], cli: Optional[str] = None,
               env: Optional[Mapping[str, str]] = None) -> Tuple[str, Dict[str, str], str]:
    """
    Start-up precedence: ``--mode`` → env ``IOT_MODE`` → saved from the dashboard → ``config.yaml`` → multiagent.

    :return: (mode, per-location overrides, source: cli | env | saved | config | default)
    """
    env = os.environ if env is None else env
    saved = store_of(config).load()
    locations = dict((saved or {}).get("locations") or {})
    if cli:
        return normalise(cli), locations, "cli"
    if env.get(ENV):
        return normalise(env[ENV]), locations, "env"
    if saved:
        return saved["mode"], locations, "saved"
    if config.get("mode"):
        return normalise(config["mode"]), locations, "config"
    return MULTIAGENT, locations, "default"


def resolve(config: Dict[str, Any], cli: Optional[str] = None,
            env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """:return: a copy with ``mode`` / ``mode_locations`` / ``mode_source`` set by ``start_mode`` (once)"""
    if config.get("mode_source"):
        return config
    mode, locations, source = start_mode(config, cli, env)
    config = {**config, "mode": mode, "mode_locations": locations, "mode_source": source}
    logger.info(f"mode {mode} (from {source})" + (f", per location {locations}" if locations else ""))
    return config


def apply_mode(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    :return: a copy of the config with every cloud path switched off when started autonomous with no location
             overriding to multiagent (unchanged otherwise)
    """
    if any_multiagent(config):
        return config
    config = copy.deepcopy(config)
    config.setdefault("llm", {})["enabled"] = False
    config.setdefault("envelope", {})["enabled"] = False
    config.setdefault("rules", {}).setdefault("suggestions", {})["enabled"] = False
    aggregator = config.setdefault("aggregator", {})
    aggregator["queue_for_llm"] = False             # snapshots / incidents stay local; nothing waits for GPT
    aggregator["global_snapshot_min"] = 0
    mic = config.setdefault("audio", {}).setdefault("mic", {})
    backend = mic.get("whisper_backend", "azure")
    if backend in ("azure", "azure_only"):
        if backend == "azure_only":
            logger.warning("mode autonomous: audio.mic.whisper_backend azure_only → local (no cloud calls)")
        mic["whisper_backend"] = "local"
    config.setdefault("hfs", {})["enabled"] = True
    return config
