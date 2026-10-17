def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (F7). Imports lazily so the package import stays light.

    :param event_queue: Bus queue
    :param control_queue: "stop", every forwarded bus event and module heartbeats
    :param config: Full merged config
    """
    from dashboard.server import run as run_server
    run_server(event_queue, control_queue, config)
