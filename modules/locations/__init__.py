"""F12 per-location state; ``run`` is the module entry point for the orchestrator."""


def run(event_queue, control_queue, config):
    from modules.locations.module import run as module_run

    module_run(event_queue, control_queue, config)
