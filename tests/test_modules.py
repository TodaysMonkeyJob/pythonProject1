import queue
import threading

from core.bus import CONTROL_STOP, create_queue, receive
from core.events import Event
from modules.audio.adapter import AudioModule, parse_transcription
from modules.gas.adapter import GasModule, classify
from modules.gas.simulated import SimulatedGasReader

SENSORS = {
    "mq2": {"unit": "ppm", "warning": 300, "critical": 1000},
    "mq7": {"unit": "ppm", "warning": 35, "critical": 200},
}


def _drain(q):
    events = []
    while True:
        try:
            events.append(Event.from_json(q.get_nowait()))
        except queue.Empty:
            return events


def _gas_module(reader):
    return GasModule(queue.Queue(), queue.Queue(), {"gas": {"sensors": SENSORS}}, reader=reader)


def test_classify():
    limits = SENSORS["mq2"]
    assert classify(10, limits) is None
    assert classify(300, limits) == "warning"
    assert classify(1000, limits) == "critical"
    assert classify(5000, {}) is None


def test_gas_publishes_reading_without_alert():
    module = _gas_module(lambda: {"mq2": 20.0, "mq7": 3.0})
    module.poll_once()
    events = _drain(module.event_queue)

    assert [e.type for e in events] == ["gas_reading"]
    assert events[0].source == "gas"
    assert events[0].severity == "info"
    assert events[0].payload["readings"] == {"mq2": 20.0, "mq7": 3.0}
    assert events[0].payload["units"] == {"mq2": "ppm", "mq7": "ppm"}


def test_gas_publishes_alerts_above_thresholds():
    module = _gas_module(lambda: {"mq2": 400.0, "mq7": 250.0})
    module.poll_once()
    reading, *alerts = _drain(module.event_queue)

    assert reading.type == "gas_reading"
    assert reading.severity == "critical"
    assert {(a.type, a.payload["sensor"], a.severity) for a in alerts} == {
        ("alert", "mq2", "warning"),
        ("alert", "mq7", "critical"),
    }
    mq7 = next(a for a in alerts if a.payload["sensor"] == "mq7")
    assert mq7.payload["threshold"] == 200


def test_gas_reader_error_becomes_alert():
    def broken():
        raise OSError("i2c timeout")

    module = _gas_module(broken)
    module.poll_once()
    (event,) = _drain(module.event_queue)
    assert event.type == "alert"
    assert event.payload["reason"] == "sensor_error"


def test_gas_run_through_process_queues_with_heartbeat_and_stop():
    event_queue, control_queue = create_queue(), create_queue()
    config = {"gas": {"sensors": SENSORS, "poll_interval": 0.05, "reader": "simulated"}}
    module = GasModule(event_queue, control_queue, config)
    thread = threading.Thread(target=module.run)
    thread.start()

    received = [receive(event_queue, timeout=5) for _ in range(3)]
    control_queue.put(CONTROL_STOP)
    thread.join(timeout=5)

    assert not thread.is_alive()
    types = [(e.source, e.type) for e in received]
    assert types[0] == ("system", "heartbeat")
    assert received[0].payload == {"module": "gas"}
    assert ("gas", "gas_reading") in types


def test_simulated_reader_stays_below_warning():
    reader = SimulatedGasReader(SENSORS, seed=1)
    for _ in range(200):
        readings = reader()
    assert set(readings) == set(SENSORS)
    assert all(0 <= readings[s] < SENSORS[s]["warning"] for s in SENSORS)


def test_parse_transcription():
    assert parse_transcription("hello there; oleh") == ("hello there", "oleh")
    assert parse_transcription("a; b; unknown") == ("a; b", "unknown")
    assert parse_transcription("no speaker") == ("no speaker", "")


class FakeRecorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_shut_down = False
        self._results = queue.Queue()
        self._results.put("turn on the light; oleh")

    def text(self):
        try:
            return self._results.get(timeout=0.05)
        except queue.Empty:
            return ""

    def shutdown(self):
        self.is_shut_down = True


def test_audio_publishes_speech_events():
    event_queue, control_queue = create_queue(), create_queue()
    config = {"audio": {"recorder": {"language": "en"}}}
    module = AudioModule(event_queue, control_queue, config, recorder_factory=FakeRecorder)
    thread = threading.Thread(target=module.run)
    thread.start()

    received = [receive(event_queue, timeout=5) for _ in range(2)]
    control_queue.put(CONTROL_STOP)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert module.recorder.kwargs == {"language": "en"}
    assert module.recorder.is_shut_down
    speech = next(e for e in received if e.type == "speech")
    assert speech.source == "audio"
    assert speech.payload == {"text": "turn on the light", "speaker": "oleh"}
