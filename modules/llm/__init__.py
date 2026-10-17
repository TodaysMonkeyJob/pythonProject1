def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (F6). Imports lazily so the package import stays light.

    :param event_queue: Bus queue
    :param control_queue: "stop"
    :param config: Full merged config
    """
    from modules.llm.worker import run as run_worker
    run_worker(event_queue, control_queue, config)
