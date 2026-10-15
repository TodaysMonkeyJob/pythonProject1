def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator. Imports lazily so loading modules.gas.data stays light.

    :param event_queue: Bus queue to publish events to
    :param control_queue: Queue with control messages ("stop", ...)
    :param config: Full merged config
    """
    from modules.gas.worker import run as run_worker
    run_worker(event_queue, control_queue, config)
