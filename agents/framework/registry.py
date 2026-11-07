"""
Agent registry (B2 / TOD-49): discovers agents, validates their manifests and switches them on / off at runtime.

* Discovery: every folder ``agents/<name>/`` with an ``agent.yaml`` (``agents.roots`` adds more folders), plus the
  Python entry points of group ``iot_guard.agents`` (an installed package whose entry point names a module; the
  ``agent.yaml`` lies next to that module). A broken manifest is listed in ``errors`` and never loaded — the other
  agents still start. Adding an agent = dropping a folder; nothing in the core changes.
* Enabled = ``enabled_default`` of the manifest < ``agents.enabled_agents.<name>`` of config.yaml < the dashboard
  switch, kept in ``agents.state_path`` (JSON) so it survives restarts. ``required`` agents (the fuzzy agent) cannot be
  switched off; ``cloud_model`` agents never start in ``mode: autonomous``; at most one agent per role ``fuzzy`` /
  ``validation`` is on.
* The agent's code is imported only when it is enabled; disabling calls ``stop()`` and drops the object, so its
  models are freed.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from agents.framework.base import Agent
from agents.framework.manifest import Manifest, ManifestError, load_manifest
from core.mode import is_autonomous
from system_setup.logger_setup import init_logger

logger = init_logger()

AGENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY_POINT_GROUP = "iot_guard.agents"
SKIP = ("framework", "__pycache__")
SINGLE_ROLES = ("fuzzy", "validation")


def discover(roots: List[str], entry_points: bool = True) -> Tuple[Dict[str, Manifest], Dict[str, str]]:
    """:return: (name → manifest, name / path → error)"""
    manifests: Dict[str, Manifest] = {}
    errors: Dict[str, str] = {}
    paths = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry, "agent.yaml")
            if entry not in SKIP and not entry.startswith((".", "_")) and os.path.isfile(path):
                paths.append(path)
    if entry_points:
        paths += _entry_point_manifests(errors)
    for path in paths:
        try:
            manifest = load_manifest(path)
        except ManifestError as e:
            errors[os.path.basename(os.path.dirname(path))] = str(e)
            logger.error(f"agent not loaded: {e}")
            continue
        if manifest.name in manifests:
            errors[manifest.name] = f"{path}: duplicate agent name (first: {manifests[manifest.name].path})"
            logger.error(errors[manifest.name])
            continue
        manifests[manifest.name] = manifest
    return manifests, errors


def _entry_point_manifests(errors: Dict[str, str]) -> List[str]:
    from importlib.metadata import entry_points

    paths = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            module = importlib.import_module(ep.value.partition(":")[0])
            path = os.path.join(os.path.dirname(os.path.abspath(module.__file__)), "agent.yaml")
            if not os.path.isfile(path):
                raise ManifestError(f"no agent.yaml next to {module.__file__}")
            paths.append(path)
        except Exception as e:     # a broken plug-in never stops the others
            errors[ep.name] = f"entry point {ep.name}: {e}"
    return paths


def load_class(manifest: Manifest) -> type:
    """Imports the agent's code: ``<module>.py`` inside its folder, else a dotted module path."""
    module_name, _, class_name = manifest.entry.partition(":")
    file = os.path.join(manifest.path or "", *module_name.split(".")) + ".py"
    if os.path.isfile(file):
        digest = hashlib.sha1(os.path.abspath(file).encode()).hexdigest()[:8]     # same name, other folder
        key = f"_iot_agent_{manifest.name}_{module_name.replace('.', '_')}_{digest}"
        module = sys.modules.get(key)
        if module is None:
            spec = importlib.util.spec_from_file_location(key, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, Agent)):
        raise ManifestError(f"{manifest.name}: {manifest.entry} is not an Agent subclass")
    return cls


class AgentRegistry:
    def __init__(self, config: Dict[str, Any], roots: Optional[List[str]] = None, entry_points: Optional[bool] = None,
                 state_path: Optional[str] = None):
        cfg = config.get("agents") or {}
        self.config, self.cfg = config, cfg
        roots = roots if roots is not None else [AGENTS_DIR, *(cfg.get("roots") or [])]
        use_eps = cfg.get("entry_points", True) if entry_points is None else entry_points
        self.manifests, self.errors = discover(roots, use_eps)
        self.state_path = state_path if state_path is not None else cfg.get("state_path")
        self.autonomous = is_autonomous(config)
        self.instances: Dict[str, Agent] = {}
        self.wanted: Dict[str, bool] = {name: bool(m.enabled_default) for name, m in self.manifests.items()}
        for name, on in (cfg.get("enabled_agents") or {}).items():
            if name in self.wanted:
                self.wanted[name] = bool(on)
        self.wanted.update({k: v for k, v in self._saved().items() if k in self.wanted})
        for name, manifest in self.manifests.items():
            if manifest.required:
                self.wanted[name] = True

    # --- persisted switches -------------------------------------------------------------------------------------
    def _saved(self) -> Dict[str, bool]:
        if not self.state_path or not os.path.isfile(self.state_path):
            return {}
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return {k: bool(v) for k, v in (json.load(f) or {}).items()}
        except (OSError, ValueError) as e:
            logger.warning(f"agents state {self.state_path} unreadable ({e}); manifest / config defaults used")
            return {}

    def _save(self) -> None:
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.wanted, f, indent=1, sort_keys=True)
        os.replace(tmp, self.state_path)

    # --- enable / disable ---------------------------------------------------------------------------------------
    def blocked(self, name: str, on: bool) -> Optional[str]:
        """:return: why the switch cannot be set (None = allowed)"""
        manifest = self.manifests.get(name)
        if manifest is None:
            return f"unknown agent {name!r}"
        if not on and manifest.required:
            return f"{name} is required (role {manifest.role}) and cannot be switched off"
        if on and manifest.backend.kind == "cloud_model" and self.autonomous:
            return f"{name} uses a cloud model; mode autonomous makes no cloud calls"
        if on and manifest.role in SINGLE_ROLES:
            other = [n for n in self.instances if n != name and self.manifests[n].role == manifest.role]
            if other:
                return f"{other[0]} already has role {manifest.role}"
        return None

    def agent_config(self, manifest: Manifest) -> Dict[str, Any]:
        """:return: the manifest's ``config`` defaults overridden by ``agents.settings.<name>``"""
        return {**manifest.config, **((self.cfg.get("settings") or {}).get(manifest.name) or {})}

    def _start(self, name: str) -> Optional[str]:
        manifest = self.manifests[name]
        reason = self.blocked(name, True)
        if reason:
            return reason
        try:
            agent = load_class(manifest)(manifest, self.agent_config(manifest), self.config)
            agent.start()
        except Exception as e:
            logger.exception(f"agent {name} failed to start")
            return f"{name} failed to start: {e}"
        self.instances[name] = agent
        logger.info(f"agent {name} ({manifest.role} {manifest.version}) on")
        return None

    def _stop(self, name: str) -> None:
        agent = self.instances.pop(name, None)
        if agent is None:
            return
        try:
            agent.stop()
        except Exception:
            logger.exception(f"agent {name} failed to stop cleanly")
        logger.info(f"agent {name} off")

    def start_enabled(self) -> Dict[str, str]:
        """:return: agent → reason for the enabled agents that could not start"""
        problems = {}
        for name in sorted(self.manifests, key=lambda n: (not self.manifests[n].required, n)):
            if self.wanted.get(name) and name not in self.instances:
                reason = self._start(name)
                if reason:
                    problems[name] = reason
                    logger.warning(f"agent {name} not started: {reason}")
        return problems

    def set_enabled(self, name: str, on: bool) -> Optional[str]:
        """Runtime switch (dashboard). :return: the reason it was refused, None when done"""
        reason = self.blocked(name, on)
        if reason:
            return reason
        if on and name not in self.instances:
            reason = self._start(name)
            if reason:
                return reason
        elif not on:
            self._stop(name)
        self.wanted[name] = on
        self._save()
        return None

    def stop_all(self) -> None:
        for name in list(self.instances):
            self._stop(name)

    # --- lookups ------------------------------------------------------------------------------------------------
    def enabled(self, role: Optional[str] = None) -> List[str]:
        return [n for n in sorted(self.instances) if role is None or self.manifests[n].role == role]

    def first(self, role: str) -> Optional[Agent]:
        names = self.enabled(role)
        return self.instances[names[0]] if names else None

    def weights(self) -> Dict[str, float]:
        """agent → measured accuracy from its eval file (unmeasured = 0, never a typed value)"""
        return {name: float(m.evaluation().get("accuracy") or 0.0) for name, m in self.manifests.items()}
