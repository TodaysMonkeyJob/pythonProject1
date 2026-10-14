from modules.gas.adapter import GasModule


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator.

    :param event_queue: Bus queue to publish events to
    :param control_queue: Queue with control messages ("stop", ...)
    :param config: Full merged config
    """
    GasModule(event_queue, control_queue, config).run()
