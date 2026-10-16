"""Tiny modules for orchestrator tests (imported by spawned child processes)."""
import os
import time

import numpy as np

from core.bus import publish
from core.events import Event, make_event
from core.module import BaseModule


class _Dummy(BaseModule):
    def step(self) -> None:
        time.sleep(0.05)


def steady(event_queue, control_queue, config):
    """Heartbeats until stopped."""
    type("Steady", (_Dummy,), {"name": "steady"})(event_queue, control_queue, config).run()


def other(event_queue, control_queue, config):
    type("Other", (_Dummy,), {"name": "other"})(event_queue, control_queue, config).run()


def crasher(event_queue, control_queue, config):
    """Crashes on the first start (marker file), runs normally after the restart."""
    marker = config["test"]["marker"]
    if not os.path.exists(marker):
        open(marker, "w").close()
        time.sleep(0.3)
        os._exit(3)
    type("Crasher", (_Dummy,), {"name": "crasher"})(event_queue, control_queue, config).run()


def hanger(event_queue, control_queue, config):
    """One heartbeat, then hangs (ignores "stop") on the first start; normal after the restart."""
    marker = config["test"]["marker"]
    if not os.path.exists(marker):
        open(marker, "w").close()
        publish(event_queue, make_event("system", "heartbeat", {"module": "hanger"}))
        while True:
            time.sleep(1)
    type("Hanger", (_Dummy,), {"name": "hanger"})(event_queue, control_queue, config).run()


def scene_source(event_queue, control_queue, config):
    """Publishes scene_active once, shortly after start."""
    class Scene(_Dummy):
        name = "scene"
        sent = False

        def step(self):
            if not self.sent and time.monotonic() - self.t0 > 1.0:
                self.emit("scene_active", {"active": True})
                self.sent = True
            time.sleep(0.02)

    module = Scene(event_queue, control_queue, config)
    module.source = "camera"
    module.t0 = time.monotonic()
    module.run()


def echo(event_queue, control_queue, config):
    """Re-publishes every forwarded event type as a `sound` event {"echo": type}."""
    class Echo(_Dummy):
        name = "echo"

        def handle_control(self, message):
            self.emit("sound", {"echo": Event.from_json(message).type})

    Echo(event_queue, control_queue, config).run()


class SilentSource:
    def open(self):
        self.t = time.monotonic()

    def read(self):
        n = int((time.monotonic() - self.t) / 0.03)
        self.t += n * 0.03
        return [np.zeros(480, np.int16)] * n

    def close(self):
        pass


def silent_mic(event_queue, control_queue, config):
    """Real MicController on a silent fake microphone (no audio hardware in tests)."""
    from modules.audio.mic_controller import MicController

    MicController(event_queue, control_queue, config, frame_source=SilentSource(), vad=lambda frame: False,
                  transcriber=lambda audio, sr: {"text": "x", "language": None}).run()
