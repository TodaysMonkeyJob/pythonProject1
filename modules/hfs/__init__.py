"""
A1 (TOD-41) autonomous hierarchical fuzzy system: three level-1 subsystems (gas → risk 1, camera → risk 2,
audio → risk 3, rule bases ``config/fuzzy/<name>.yaml``) and a top level (``config/fuzzy/top.yaml``) → overall risk
per location. No LLM, no network. See ``modules/hfs/README.md``.
"""


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator. Imports lazily so the package import stays light.

    :param event_queue: Bus queue
    :param control_queue: "stop" + forwarded evidence events
    :param config: Full merged config
    """
    from modules.hfs.worker import run as run_worker
    run_worker(event_queue, control_queue, config)
