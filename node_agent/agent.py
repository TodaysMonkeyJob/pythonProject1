"""
Edge node agent for a Raspberry Pi (F11 / TOD-22): runs the full local pipeline (cameras, microphones, gas — found
automatically) and sends events + camera previews to the hub. No MQTT broker or other programs needed.

    python -m node_agent.agent --pair 123456 --name "Pi kitchen"   # once: code from the hub dashboard (Devices)
    python -m node_agent.agent                                     # run (systemd: deploy/iot-guard.service)

The hub is found by its UDP announcement on the LAN (or ``--hub https://192.168.0.112:8443``). At pairing the hub's
certificate is saved and pinned; the token and node id are kept in ``data/node_agent/`` (never commit). Processing
stays on the Pi; the hub aggregates, analyses (GPT) and shows everything. Face identification runs on the hub's
registry only, so it is off on edge nodes by default.
"""
import argparse
import json
import os
import ssl
import sys
import urllib.request
from typing import List, Optional
from urllib.parse import urlparse

if __package__ in (None, ""):  # allow `python node_agent/agent.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config  # noqa: E402

STATE_DIR = "data/node_agent"
EDGE_OFF = ("aggregator", "llm", "dashboard", "identity")   # the hub does these


def load_state(folder: str = STATE_DIR) -> Optional[dict]:
    try:
        with open(os.path.join(folder, "state.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def pair(hub: str, code: str, name: str, folder: str = STATE_DIR) -> dict:
    """Saves the hub certificate (pinned from now on), exchanges the code for a token, stores the node state."""
    parsed = urlparse(hub)
    os.makedirs(folder, exist_ok=True)
    cafile = os.path.join(folder, "hub.pem")
    with open(cafile, "w", encoding="utf-8") as f:
        f.write(ssl.get_server_certificate((parsed.hostname, parsed.port or 443)))
    request = urllib.request.Request(f"{hub.rstrip('/')}/api/pair", method="POST",
                                     data=json.dumps({"code": code, "name": name, "kind": "edge"}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, context=ssl.create_default_context(cafile=cafile), timeout=15) as response:
        answer = json.load(response)
    state = {"hub": hub.rstrip("/"), "node_id": answer["node_id"], "token": answer["token"], "cafile": cafile,
             "name": name}
    path = os.path.join(folder, "state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.chmod(path, 0o600)
    return state


def edge_config(config: dict, state: dict) -> dict:
    """The hub-less part of the config for this node + the uplink module."""
    network = config.setdefault("network", {})
    network.update(node_id=state["node_id"], discovery=True, node_port=0, announce=False)
    for section in EDGE_OFF:
        config.setdefault(section, {})["enabled"] = False
    config["uplink"] = {"hub": state["hub"], "token": state["token"], "cafile": state.get("cafile"),
                        **config.get("uplink", {})}
    return config


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="IoT Guard edge node (Raspberry Pi)")
    parser.add_argument("--pair", metavar="CODE", help="pair with the hub using the code from its dashboard")
    parser.add_argument("--hub", help="https://<hub-ip>:8443 (default: found on the LAN)")
    parser.add_argument("--name", default=os.uname().nodename)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    if args.pair:
        from dashboard.node import find_hub

        hub = args.hub or find_hub(30)
        if not hub:
            raise SystemExit("No hub announcement on the LAN within 30 s — give --hub https://<hub-ip>:8443")
        state = pair(hub, args.pair, args.name)
        print(f"Paired with {state['hub']} as {state['node_id']}. Start: python -m node_agent.agent")
        return

    state = load_state()
    if state is None:
        raise SystemExit("Not paired yet: python -m node_agent.agent --pair <code from the hub dashboard>")
    from core.orchestrator import ModuleSpec, Orchestrator

    config = edge_config(load_config(args.config), state)
    specs = Orchestrator.module_specs(config) + [ModuleSpec("uplink", "modules.uplink:run", ["*"])]
    Orchestrator(config, specs).run()


if __name__ == "__main__":
    main()
