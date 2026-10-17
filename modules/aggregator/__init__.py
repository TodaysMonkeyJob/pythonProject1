def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (F5). Imports lazily so the package import stays light.

    :param event_queue: Bus queue
    :param control_queue: "stop" and every forwarded bus event
    :param config: Full merged config
    """
    from modules.aggregator.aggregator import run as run_aggregator
    run_aggregator(event_queue, control_queue, config)
