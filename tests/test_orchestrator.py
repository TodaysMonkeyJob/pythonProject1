import json
import os
import time

import pytest

from core.config import load_config
from core.orchestrator import DEFAULT_MODULES, ModuleSpec, Orchestrator

D = "tests.orchestrator_dummies"


def make(tmp_path, specs, **orchestrator_cfg):
    config = {
        "orchestrator": {"startup_grace_sec": 30, "stop_timeout_sec": 5, **orchestrator_cfg},
        "logging": {"dir": str(tmp_path / "logs")},
        "test": {"marker": str(tmp_path / "marker")},
        "audio": {"mic": {"idle_timeout_sec": 1, "whisper_backend": "none"}},
    }
    return Orchestrator(config, [ModuleSpec(name, f"{D}:{fn}", subscribe) for name, fn, subscribe in specs])


def pump(orchestrator, seconds, until=None):
    """Runs the router + watchdog loop for up to `seconds` (or until `until()` is true)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        orchestrator.poll(timeout=0.05)
        orchestrator.check_health()
        if until and until():
            return True
    return False


def logged(tmp_path):
    lines = []
    if not (tmp_path / "logs").exists():
        return lines                                                         # nothing but heartbeats so far
    for day in os.listdir(tmp_path / "logs"):
        with open(tmp_path / "logs" / day / "events.jsonl") as f:
            lines += [json.loads(line) for line in f]
    return lines


def test_heartbeats_tracked_and_graceful_shutdown(tmp_path):
    orch = make(tmp_path, [("steady", "steady", []), ("other", "other", [])])
    orch.start()
    try:
        assert pump(orch, 20, until=lambda: all(m.last_heartbeat for m in orch.modules.values()))
    finally:
        started = time.monotonic()
        orch.shutdown()
    assert time.monotonic() - started < 5
    assert all(m.process.exitcode == 0 for m in orch.modules.values())        # stopped via "stop", not killed
    assert not any(e["type"] == "heartbeat" for e in logged(tmp_path))        # heartbeats are not logged


def test_crashed_module_is_restarted_and_logged(tmp_path):
    orch = make(tmp_path, [("crasher", "crasher", []), ("steady", "steady", [])])
    orch.start()
    try:
        crasher = orch.modules["crasher"]
        assert pump(orch, 30, until=lambda: crasher.restarts == 1 and crasher.last_heartbeat)
        steady_pid = orch.modules["steady"].process.pid
    finally:
        orch.shutdown()
    assert steady_pid == orch.modules["steady"].process.pid                   # the other module kept running
    alerts = [e for e in logged(tmp_path) if e["type"] == "alert"]
    assert alerts[0]["source"] == "system" and alerts[0]["payload"]["module"] == "crasher"
    assert alerts[0]["payload"]["cause"] == "exited with code 3" and alerts[0]["payload"]["restarts"] == 1


def test_silent_module_is_restarted_after_missed_heartbeats(tmp_path):
    orch = make(tmp_path, [("hanger", "hanger", [])], missed_heartbeats=0.4)   # 2 s without a heartbeat
    orch.start()
    try:
        hanger = orch.modules["hanger"]
        assert pump(orch, 30, until=lambda: hanger.restarts == 1)
    finally:
        orch.shutdown()
    cause = [e for e in logged(tmp_path) if e["type"] == "alert"][0]["payload"]["cause"]
    assert cause.startswith("no heartbeat for")


def test_killed_module_restarts_quickly_others_unaffected(tmp_path):
    orch = make(tmp_path, [("steady", "steady", []), ("other", "other", [])])
    orch.start()
    try:
        assert pump(orch, 20, until=lambda: all(m.last_heartbeat for m in orch.modules.values()))
        victim, survivor = orch.modules["steady"], orch.modules["other"]
        survivor_pid = survivor.process.pid
        killed_at = time.monotonic()
        os.kill(victim.process.pid, 9)
        assert pump(orch, 15, until=lambda: victim.restarts == 1 and victim.last_heartbeat is not None
                    and victim.last_heartbeat > killed_at)
        restart_sec = time.monotonic() - killed_at
    finally:
        orch.shutdown()
    assert restart_sec <= 15                                                  # acceptance: back within 15 s
    assert survivor.process.pid == survivor_pid


def test_router_fans_out_to_subscribers_and_logs(tmp_path):
    orch = make(tmp_path, [("scene", "scene_source", []), ("echo", "echo", ["scene_active"]),
                           ("steady", "steady", [])])
    orch.start()
    try:
        assert pump(orch, 30, until=lambda: any(e["type"] == "sound" for e in logged_now(orch, tmp_path)))
    finally:
        orch.shutdown()
    events = logged(tmp_path)
    assert [e["type"] for e in events if e["type"] in ("scene_active", "sound")] == ["scene_active", "sound"]
    assert next(e for e in events if e["type"] == "sound")["payload"] == {"echo": "scene_active"}


def logged_now(orch, tmp_path):
    """Events logged so far (the logger flushes after every write)."""
    return logged(tmp_path)


def test_person_enters_mic_active_within_one_second(tmp_path):
    orch = make(tmp_path, [("scene", "scene_source", []), ("audio", "silent_mic", ["scene_active"])])
    orch.start()
    try:
        assert pump(orch, 40, until=lambda: any(e["type"] == "mic_state" for e in logged_now(orch, tmp_path)))
    finally:
        orch.shutdown()
    events = logged(tmp_path)
    scene = next(e for e in events if e["type"] == "scene_active")
    active = next(e for e in events if e["type"] == "mic_state")
    assert active["payload"] == {"state": "active", "reason": "scene_active", "module": "audio"}
    assert active["source"] == "system"
    from datetime import datetime
    delay = (datetime.fromisoformat(active["timestamp"][:-1]) - datetime.fromisoformat(scene["timestamp"][:-1]))
    assert delay.total_seconds() <= 1.0


def test_module_specs_from_config():
    config = load_config()
    names = [s.name for s in Orchestrator.module_specs(config)]
    assert names == ["gas", "actions", "identity", "aggregator", "llm", "dashboard"]   # cameras / mics: per device
    config["network"]["discovery"] = False
    names = [s.name for s in Orchestrator.module_specs(config)]
    assert names == ["gas", "vision", "actions", "identity", "audio", "aggregator", "llm", "dashboard"]
    config["identity"]["enabled"] = False
    config["orchestrator"]["modules"] = {"gas": {"enabled": False}, "audio": {"target": "modules.audio:run"}}
    specs = {s.name: s for s in Orchestrator.module_specs(config)}
    assert "identity" not in specs and "gas" not in specs
    assert specs["audio"].target == "modules.audio:run" and specs["audio"].subscribe[0] == "scene_active"
    assert DEFAULT_MODULES["actions"][2] == ["gas_reading", "alert", "object_detected"]
    assert "identity_renamed" in DEFAULT_MODULES["identity"][2]
    assert DEFAULT_MODULES["vision"][2] == ["person_identified"]


def test_child_config_disables_module_self_logging():
    child = Orchestrator._child_config({"vision": {"log_events": True}, "identity": {}})
    assert child["vision"]["log_events"] is False and child["identity"]["log_events"] is False
    assert child["actions"]["log_events"] is False


@pytest.mark.parametrize("bad", ["not json", '{"type": "nope"}'])
def test_malformed_events_are_dropped(tmp_path, bad):
    orch = make(tmp_path, [])
    orch._route(bad)
    assert orch.routed == 0


def test_discovered_devices_get_their_own_processes(tmp_path):
    from core.discovery import FoundDevice

    cam = FoundDevice("camera", "local", "USB cam", f"test:{tmp_path.name}", {"source": 7})
    mic = FoundDevice("mic", "local", "Test mic", f"mic:{tmp_path.name}", {"device": "Test mic"})
    found = [cam, mic]
    config = {
        "orchestrator": {"startup_grace_sec": 30, "stop_timeout_sec": 5,
                         "modules": {"vision": {"target": f"{D}:steady"}, "audio": {"target": f"{D}:steady"}}},
        "logging": {"dir": str(tmp_path / "logs")},
        "network": {"discovery": True, "devices_path": str(tmp_path / "devices.db"), "node_id": "testhub",
                    "discovery_interval_sec": 0.2},
        "vision": {"enabled": True, "model": "yolo11n.pt", "resolution": [64, 48]}, "audio": {"enabled": True},
    }
    orch = Orchestrator(config, [], device_scan=lambda cfg: list(found))
    orch.start()
    try:
        pipelines = [n for n in orch.modules if "@" in n]
        assert sorted(n.split("@")[0] for n in pipelines) == ["audio", "vision"]
        assert pump(orch, 20, until=lambda: all(orch.modules[n].last_heartbeat for n in pipelines))
        found.remove(cam)                                                   # camera unplugged
        assert pump(orch, 5, until=lambda: not any(n.startswith("vision@") for n in orch.modules))
        found.append(cam)                                                   # plugged back
        assert pump(orch, 5, until=lambda: any(n.startswith("vision@") for n in orch.modules))
    finally:
        orch.shutdown()
    events = logged(tmp_path)
    kinds = [(e["type"], e["payload"].get("kind")) for e in events if e["type"].startswith("device_")]
    assert kinds.count(("device_online", "camera")) == 2 and ("device_offline", "camera") in kinds
    assert all(e["device_id"] and e["node_id"] and e["location_id"] for e in events)       # every event
    assert {e["node_id"] for e in events} == {"testhub"}
    assert not orch.device_buffers                                          # shared memory released


def test_child_processes_get_small_thread_pools():
    from core.orchestrator import thread_env

    env = thread_env({"vision": {"threads": 3}}, "vision")
    assert env["OMP_NUM_THREADS"] == "3" and env["VECLIB_MAXIMUM_THREADS"] == "3"
    assert env["OMP_WAIT_POLICY"] == "PASSIVE"                       # no busy-waiting on shared cores
    assert thread_env({"orchestrator": {"child_threads": 4}}, "aggregator")["OMP_NUM_THREADS"] == "4"
    assert thread_env({}, None)["MKL_NUM_THREADS"] == "2"
