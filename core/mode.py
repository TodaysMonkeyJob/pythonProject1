"""
Operating mode (A1 / TOD-41): ``mode: autonomous | hybrid`` in ``config.yaml``.

* ``autonomous`` (Article 1) — every decision is made locally by the hierarchical fuzzy system (``modules/hfs``):
  the LLM worker, GPT gating, the risk envelope and GPT rule suggestions are not started, the aggregator queues
  nothing for GPT, Whisper runs locally (faster-whisper) — no cloud calls. The location state follows ``hfs_risk``.
* ``hybrid`` (default, Article 2) — the current behaviour: fuzzy layers around GPT; the HFS runs as a shadow (its
  ``hfs_risk`` events are logged and shown, they do not change the location state).
"""
import copy
from typing import Any, Dict

from system_setup.logger_setup import init_logger

logger = init_logger()

AUTONOMOUS, HYBRID = "autonomous", "hybrid"
MODES = (AUTONOMOUS, HYBRID)
# Modules that talk to GPT; never imported in autonomous mode (tests/test_hfs.py checks sys.modules)
LLM_PACKAGES = ("modules.llm", "modules.envelope", "modules.rules", "openai")


def mode_of(config: Dict[str, Any]) -> str:
    """:return: ``autonomous`` or ``hybrid`` (missing = hybrid)"""
    mode = str(config.get("mode") or HYBRID).strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {config.get('mode')!r}")
    return mode


def is_autonomous(config: Dict[str, Any]) -> bool:
    return mode_of(config) == AUTONOMOUS


def apply_mode(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    :return: a copy of the config with every cloud path switched off in autonomous mode (unchanged in hybrid)
    """
    if not is_autonomous(config):
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
