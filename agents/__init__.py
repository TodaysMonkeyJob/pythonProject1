"""
B2 (TOD-49) agents: plug-in AI agents on the event bus (``agents/<name>/agent.yaml`` + code), hosted in one module
process. Framework in ``agents/framework``; how to write an agent: ``docs/agents.md``.
"""


def run(event_queue, control_queue, config):
    """
    Module entry point for the orchestrator (all events forwarded; the host filters them per agent scope).

    :param event_queue: Bus queue
    :param control_queue: "stop" + forwarded events
    :param config: Full merged config
    """
    from agents.module import AgentsModule

    AgentsModule(event_queue, control_queue, config).run()
