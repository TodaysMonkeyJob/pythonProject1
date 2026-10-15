def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator. Imports lazily so importing the package stays light.

    :param event_queue: Bus queue to publish events to
    :param control_queue: Queue with control messages ("stop", ...)
    :param config: Full merged config
    """
    from modules.vision.detector import run as run_detector
    run_detector(event_queue, control_queue, config)
