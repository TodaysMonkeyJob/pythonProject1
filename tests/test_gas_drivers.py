"""A7 (TOD-48) gas sensor drivers: protocol frames, discovery on fake buses, native units, per-sensor models."""
import json
import struct

import pytest
import yaml

from drivers.gas.base import DriverError
from drivers.gas.discovery import Discovery, load_config, report, save_choice, write_state
from drivers.gas.electrochemical import checksum, command, parse_concentration
from drivers.gas.particle import crc8, find_pms_frame, find_sds_frame
from modules.hfs.gas_model_builder import build, reading_interval
from modules.hfs.gas_normalize import GasTable, convert

# Nova SDS011 datasheet example frame: PM2.5 = 0x04D4 / 10 = 123.6, PM10 = 0x0A3A / 10 = 261.8, id A160
SDS_FRAME = bytes.fromhex("aac0d4043a0aa1601dab")


def pms_frame(pm25: int, pm1: int = 3, pm10: int = 9) -> bytes:
    words = [0x424D, 28, pm1, pm25, pm10, pm1, pm25, pm10, 0, 0, 0, 0, 0, 0, 0]
    body = struct.pack(">15H", *words)
    return body + struct.pack(">H", sum(body) & 0xFFFF)


def dfrobot_reply(ppm_x10: int, gas_type: int = 0x02, decimals: int = 1) -> bytes:
    frame = [0xFF, 0x86, ppm_x10 >> 8, ppm_x10 & 0xFF, gas_type, decimals, 0, 0, 0]
    frame[8] = checksum(frame)
    return bytes(frame)


def sps_words(data: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 2):
        pair = data[i:i + 2]
        out += pair + bytes([crc8(pair)])
    return bytes(out)


# --- fake buses -----------------------------------------------------------------------------------------------------

class FakeI2C:
    """address → handler(last write, n, last multi-byte write) → reply bytes"""

    def __init__(self, devices, name="i2c-1"):
        self.devices, self.name, self.last, self.frame = devices, name, {}, {}

    def present(self, address):
        return address in self.devices

    def write(self, address, data):
        if address not in self.devices:
            raise OSError("NACK")
        self.last[address] = list(data)
        if len(data) > 1:
            self.frame[address] = list(data)

    def read(self, address, n):
        if address not in self.devices:
            raise OSError("NACK")
        handler = self.devices[address]
        reply = handler(self.last.get(address, []), n, self.frame.get(address, [])) \
            if handler.__code__.co_argcount == 3 else handler(self.last.get(address, []), n)
        return bytes(reply)[:n]

    def write_read(self, address, data, n, delay=0.0):
        self.write(address, data)
        return self.read(address, n)


def dfrobot(ppm_x10=420, gas_type=0x02):
    def handler(last, n, frame):
        if len(frame) == 10 and frame[3] == 0x86:              # register 0 + frame, then register 0 again
            return dfrobot_reply(ppm_x10, gas_type)
        return bytes(9)                                         # mode change ack (ignored)
    return handler


def sps30(pm25=37.5):
    def handler(last, n):
        code = (last[0] << 8) | last[1] if len(last) >= 2 else None
        if code == 0xD002:
            return sps_words(b"00080000")
        if code == 0xD033:
            return sps_words(b"SPS30SN12345678\0".ljust(32, b"\0"))
        if code == 0x0202:
            return sps_words(b"\x00\x01")
        if code == 0x0300:
            return sps_words(struct.pack(">10f", 20.0, pm25, 40.0, 45.0, 1, 2, 3, 4, 5, 0.6))
        return b""
    return handler


def ens160(aqi=3):
    def handler(last, n):
        reg = last[0] if last else None
        return {0x00: b"\x60\x01", 0x20: b"\x00", 0x21: bytes([aqi]), 0x22: (420).to_bytes(2, "little"),
                0x24: (800).to_bytes(2, "little")}.get(reg, b"\x00" * n)
    return handler


def ads1115(raw=16384):
    def handler(last, n):
        if last == [0x01]:                                     # config register (power-on default 0x8583)
            return b"\x85\x83"
        return raw.to_bytes(2, "big", signed=True)             # conversion register
    return handler


def garbage(last, n):
    return bytes([0x12] * n)


class FakeSerial:
    """A port with one device that only understands its baud rate."""

    def __init__(self, path, baud, device, device_baud):
        self.name, self.baud, self.device, self.ok = path, baud, device, baud == device_baud
        self.outbox = b""

    def write(self, data):
        if self.ok:
            self.outbox += self.device(data)

    def read(self, n):
        if not self.ok:
            return b""
        if not self.outbox:
            self.outbox = self.device(None)
        out, self.outbox = self.outbox[:n], self.outbox[n:]
        return out

    def readline(self, limit=4096):
        if not self.ok:
            return b""
        line, _, rest = self.outbox.partition(b"\n")
        self.outbox = rest
        return line + b"\n" if line else b""

    def reset(self):
        self.outbox = b""

    def close(self):
        pass


def graphene_board(request):
    if request == b"ID?\n":
        return json.dumps({"device": "graphene_fet", "serial": "GFET-07", "gates": 16, "vg": [0.0, 1.5],
                           "unit": "vol_pct"}).encode() + b"\n"
    if request == b"READ\n":
        return json.dumps({"r": [1000.0 + i for i in range(16)]}).encode() + b"\n"
    return b""


def sds011_stream(request):
    return SDS_FRAME * 3 if request is None else b""


def pms_stream(request):
    return pms_frame(55) * 3 if request is None else b""


def setup(i2c_devices=None, ports=None, config=None):
    ports = ports or {}

    def open_serial(path, baud):
        device, device_baud = ports[path]
        return FakeSerial(path, baud, device, device_baud)

    bus = FakeI2C(i2c_devices or {})
    return Discovery(config or {"devices": [], "choices": {}}, open_i2c=lambda n: bus, open_serial=open_serial,
                     i2c_numbers=[1], ports=list(ports)), bus


# --- protocol frames ------------------------------------------------------------------------------------------------

def test_dfrobot_frames():
    frame = command(0x86)
    assert frame[:3] == [0xFF, 0x01, 0x86] and frame[8] == checksum(frame) == 0x79
    assert parse_concentration(dfrobot_reply(425, 0x02, 1)) == {"ppm": 42.5, "gas_type": 0x02}
    bad = bytearray(dfrobot_reply(425))
    bad[3] ^= 1
    with pytest.raises(DriverError):
        parse_concentration(bytes(bad))


def test_particle_frames():
    assert find_sds_frame(b"\x00\x01" + SDS_FRAME) == {"pm2_5": 123.6, "pm10": 261.8, "id": "a160"}
    assert find_pms_frame(b"\x42" + pms_frame(55))["pm2_5"] == 55.0          # a stray header byte first
    with pytest.raises(DriverError):
        find_pms_frame(pms_frame(55)[:-1] + b"\x00")
    assert crc8(b"\xbe\xef") == 0x92                                          # Sensirion datasheet example


# --- discovery -------------------------------------------------------------------------------------------------------

def test_three_sensors_detected_with_native_units():
    discovery, _ = setup({0x74: dfrobot()}, {"/dev/ttyUSB0": (graphene_board, 115200),
                                             "/dev/ttyAMA0": (sds011_stream, 9600)})
    result = discovery.run()
    units = {d.driver: sorted({q.unit for q in d.quantities}) for d in result["sensors"]}
    assert units == {"dfrobot_gravity": ["ppm"], "graphene_fet": ["vol_pct"], "sds011": ["ug_m3"]}
    assert not result["unknown"] and not result["errors"]
    nh3 = next(d for d in result["sensors"] if d.driver == "dfrobot_gravity")
    assert nh3.model.startswith("DFRobot SEN0469") and nh3.quantities[0].range == [0.0, 100.0]
    ids = {d.device_id for d in result["sensors"]}
    assert len(ids) == 3 and ids == {d.device_id for d in setup(
        {0x74: dfrobot()}, {"/dev/ttyUSB0": (graphene_board, 115200),
                            "/dev/ttyAMA0": (sds011_stream, 9600)})[0].run()["sensors"]}     # stable


def test_i2c_identification_and_unknown_devices():
    discovery, _ = setup({0x69: sps30(), 0x53: ens160(), 0x76: garbage, 0x48: ads1115(), 0x20: garbage})
    result = discovery.run()
    assert {d.driver for d in result["sensors"]} == {"sps30", "ens160"}
    sps = next(d for d in result["sensors"] if d.driver == "sps30")
    assert sps.serial == "SPS30SN12345678"
    unknown = {u["key"]: u for u in result["unknown"]}
    assert set(unknown) == {"i2c-1:0x76", "i2c-1:0x48", "i2c-1:0x20"}
    assert unknown["i2c-1:0x76"]["candidates"] == ["dfrobot_gravity"]          # shares its address, no valid answer
    assert unknown["i2c-1:0x48"]["candidates"] == ["mq_ads1115"]               # an ADC cannot identify the MQ
    assert unknown["i2c-1:0x20"]["reason"] == "no gas driver uses this address"


def test_unknown_serial_port_and_pms():
    discovery, _ = setup(ports={"/dev/ttyUSB1": (pms_stream, 9600),
                                "/dev/ttyUSB2": (lambda r: b"hello?" if r is None else b"", 9600),
                                "/dev/ttyUSB3": (lambda r: b"", 9600)})
    result = discovery.run()
    assert [d.driver for d in result["sensors"]] == ["pms5003"]
    assert [u["key"] for u in result["unknown"]] == ["/dev/ttyUSB2"]           # data, unrecognised; empty port silent


def test_user_choice_is_saved_and_used(tmp_path):
    path = str(tmp_path / "sensors.yaml")
    options = {"channel": 1, "model": "MQ-137", "gas": "ammonia", "range": [5, 500], "curve": {"a": 50.0, "b": -2.0},
               "r0_ohm": 14414.0625}
    save_choice("i2c-1:0x48", "mq_ads1115", options, path, by="op")
    assert yaml.safe_load(open(path, encoding="utf-8"))["choices"]["i2c-1:0x48"]["driver"] == "mq_ads1115"
    with pytest.raises(ValueError):
        save_choice("i2c-1:0x48", "nope", None, path)
    discovery, bus = setup({0x48: ads1115(16384)}, config=load_config(path))
    result = discovery.run()
    assert [d.driver for d in result["sensors"]] == ["mq_ads1115"] and not result["unknown"]
    driver = discovery.drivers(result)[0]
    reading = driver.read()                                                   # 2.048 V → Rs = R0 → ppm = a
    assert reading == {"concentration": 50.0, "volts": 2.048}
    assert bus.last[0x48][0] == 0x00                                          # read the conversion register
    declared = setup({0x48: ads1115()}, config={"devices": [{"driver": "mq_ads1115", "bus": "i2c-1",
                                                             "address": "0x48", **options}], "choices": {}})[0].run()
    assert [d.driver for d in declared["sensors"]] == ["mq_ads1115"] and not declared["unknown"]


def test_mq_without_calibration_is_an_error():
    discovery, _ = setup({0x48: ads1115()}, config={"devices": [], "choices": {"i2c-1:0x48": {
        "driver": "mq_ads1115", "options": {"gas": "ammonia", "curve": {"a": 0, "b": -2}, "r0_ohm": 0}}}})
    result = discovery.run()
    assert not result["sensors"] and "calibrate" in result["errors"][0]


# --- readings in native units ---------------------------------------------------------------------------------------

class FakePredictor:
    def __init__(self):
        self.curves = []

    def predict(self, curve):
        self.curves.append(list(curve))
        return {"gas": "ammonia", "gas_confidence": 0.93, "concentration_pct": 0.5}


def test_graphene_model_runs_in_percent():
    discovery, _ = setup(ports={"/dev/ttyUSB0": (graphene_board, 115200)})
    result = discovery.run()
    driver = discovery.drivers(result)[0]
    driver.predictor = FakePredictor()
    reading = driver.read()
    assert driver.predictor.curves == [[1000.0 + i for i in range(16)]]       # the raw curve, nothing converted
    assert (reading["concentration"], reading["unit"]) == (0.5, "vol_pct")
    assert driver.descriptor.needs_model and driver.descriptor.details["model_unit"] == "vol_pct"
    nh3 = next(q for q in driver.descriptor.quantities if q.gas == "ammonia")
    assert nh3.unit == "vol_pct" and nh3.accuracy["absolute"] > 0              # measured MAE (config/uncertainty.yaml)


def test_index_sensor_and_sps30_read():
    discovery, _ = setup({0x69: sps30(41.25), 0x53: ens160()})
    result = discovery.run()
    readings = {d.descriptor.driver: d.read() for d in discovery.drivers(result)}
    assert readings["sps30"] == {"pm2_5": 41.25}
    assert readings["ens160"] == {"aqi_uba": 3.0, "tvoc": 420.0, "eco2": 800.0, "validity": "normal"}
    ens = next(d for d in result["sensors"] if d.driver == "ens160")
    assert {q.kind for q in ens.quantities} == {"index"}


# --- conversions, generated model, saturation ------------------------------------------------------------------------

def test_conversions():
    table = GasTable.load("config/gases.yaml")
    assert convert(1, "vol_pct", "ppm") == 10_000
    assert convert(1200, "ppm", "mg_m3", 28.01) == pytest.approx(1374.7, abs=0.1)   # ≈ 1380 in the issue
    assert table.get("ammonia").c_crit_in("vol_pct") == pytest.approx(0.03)
    assert table.get("pm2_5").c_crit_in("ug_m3") == 225.5


def descriptors():
    discovery, _ = setup({0x74: dfrobot(), 0x53: ens160()}, {"/dev/ttyUSB0": (graphene_board, 115200),
                                                             "/dev/ttyAMA0": (sds011_stream, 9600)})
    return discovery, discovery.run()


def test_generated_inputs_in_native_units(tmp_path):
    _, result = descriptors()
    content = build(report(result)["sensors"], GasTable.load("config/gases.yaml"))
    inputs = {name: entry for s in content["sensors"].values() for name, entry in s["inputs"].items()}
    nh3_ppm = next(e for e in inputs.values() if e.get("gas") == "ammonia" and e["unit"] == "ppm")
    assert nh3_ppm["c_crit"]["value"] == 300 and nh3_ppm["terms"]["moderate"] == ["tri", 30.0, 90.0, 180.0]
    assert nh3_ppm["delta"] == {"absolute": 5.0, "relative": 0.1, "resolution": 1.0}
    assert nh3_ppm["saturation"] == {"range_high": 100.0, "share_of_c_crit": 0.3333, "open_upper_bound": True}
    assert any("cannot confirm a critical ammonia" in w for w in content["warnings"])
    nh3_pct = next(e for e in inputs.values() if e.get("gas") == "ammonia" and e["unit"] == "vol_pct")
    assert nh3_pct["c_crit"]["value"] == pytest.approx(0.03) and nh3_pct["terms"]["dangerous"][3] == pytest.approx(0.03)
    assert "saturation" not in nh3_pct                                         # range 0.25–14 % is above C_crit
    pm = next(e for e in inputs.values() if e.get("gas") == "pm2_5")
    assert pm["unit"] == "ug_m3" and pm["universe"][1] == 999.9                 # the sensor range is wider
    aqi = next(e for e in inputs.values() if e["quantity"] == "aqi_uba")
    assert aqi["kind"] == "index" and "gas" not in aqi and set(aqi["terms"]) == {"good", "moderate", "poor",
                                                                                 "unhealthy"}
    assert all(e["provenance"] for e in inputs.values())


def test_full_scale_is_an_open_upper_bound():
    from core.events import make_event
    from modules.hfs.gas import GasSubsystem

    table = GasTable.load("config/gases.yaml")
    q = {"range": [0.0, 100.0], "resolution": 1.0, "accuracy": {"relative": 0.1, "absolute": 5.0}}
    low, high, is_open = reading_interval(100.0, "ppm", q, table.get("ammonia"))
    assert is_open and high == 1.2 and low == pytest.approx((100 - 10) / 300, abs=1e-6)
    assert reading_interval(50.0, "ppm", q, table.get("ammonia"))[2] is False
    gas = GasSubsystem({"hfs": {}})

    def corridor(value):
        now = gas.clock()
        gas.observe(make_event("gas", "gas_reading", {"readings": [{"gas": "ammonia", "concentration": value,
                                                                    "unit": "ppm", **q}], "driver": "dfrobot_gravity",
                                                      "sensor_model": "DFRobot SEN0469 (NH3)"},
                               location_id="lab", device_id="gas-x"), now)
        return gas.evaluate("lab", now, True)

    full, half = corridor(100.0), corridor(50.0)
    assert full["saturated"] == ["ammonia"] and "saturated" not in half
    assert full["interval"][1] > half["interval"][1] + 0.2                    # a critical level cannot be excluded


# --- the gas module in drivers mode ----------------------------------------------------------------------------------

def test_drivers_mode_publishes_native_readings(tmp_path):
    from modules.gas.drivers_mode import DriverPoller

    discovery, result_ref = descriptors()
    sent = []
    cfg = {"sensors_state": str(tmp_path / "gas_sensors.json"), "generated_model": str(tmp_path / "gen.yaml"),
           "gases": "config/gases.yaml"}

    def factory():
        fresh, _ = descriptors()
        return fresh

    poller = DriverPoller(cfg, lambda type_, payload, **kw: sent.append((type_, payload, kw)), clock=lambda: 0.0,
                          discovery_factory=factory)
    poller.discover()
    for driver in poller.drivers:
        if driver.descriptor.needs_model:
            driver.predictor = FakePredictor()
    assert poller.step() == 4
    by_driver = {p.get("driver"): (t, p, kw) for t, p, kw in sent}
    t, p, kw = by_driver["dfrobot_gravity"]
    assert t == "gas_reading" and p["readings"][0] == {"gas": "ammonia", "concentration": 42.0, "unit": "ppm",
                                                       "range": [0.0, 100.0], "resolution": 1.0,
                                                       "accuracy": {"relative": 0.1, "absolute": 5.0},
                                                       "saturated": False, "selectivity": "selective"}
    assert kw["device_id"] == p["sensor_id"]
    assert by_driver["sds011"][1]["readings"][0]["unit"] == "ug_m3"
    graphene = by_driver["graphene_fet"][1]
    assert graphene["concentration_pct"] == 0.5 and graphene["unit"] == "vol_pct" and graphene["readings"][0]["unit"] \
        == "vol_pct"
    assert by_driver["ens160"][0] == "air_quality_index"
    state = json.load(open(tmp_path / "gas_sensors.json", encoding="utf-8"))
    assert len(state["sensors"]) == 4 and (tmp_path / "gen.yaml").is_file()
    assert poller.step() == 0                                                  # not due again yet
    broken = poller.drivers[0]
    broken.read = lambda: (_ for _ in ()).throw(OSError("i2c timeout"))
    poller.due.clear()
    poller.step()
    assert any(t == "alert" and p["reason"] == "sensor_error" for t, p, _ in sent)


def test_dashboard_gas_sensors_api(tmp_path):
    from fastapi.testclient import TestClient

    from dashboard.server import DashboardState, create_app
    from tests.test_dashboard import auth_header, config

    cfg = {**config(tmp_path), "gas": {"sensors_state": str(tmp_path / "state.json"),
                                       "sensors_config": str(tmp_path / "sensors.yaml")}}
    state = DashboardState(cfg)
    sent = []
    state.publish = sent.append
    app = TestClient(create_app(state, cfg))
    assert app.get("/api/gas_sensors", headers=auth_header()).status_code == 503
    _, result = descriptors()
    write_state(result, str(tmp_path / "state.json"))
    data = app.get("/api/gas_sensors", headers=auth_header()).json()
    assert {s["driver"] for s in data["sensors"]} == {"dfrobot_gravity", "graphene_fet", "sds011", "ens160"}
    r = app.post("/api/gas_sensors/choice", json={"key": "i2c-1:0x76", "driver": "dfrobot_gravity"},
                 headers=auth_header())
    assert r.status_code == 200 and sent[0].type == "gas_sensor_choice"
    assert yaml.safe_load(open(tmp_path / "sensors.yaml", encoding="utf-8"))["choices"]["i2c-1:0x76"]["by"] == "admin"
    assert app.post("/api/gas_sensors/choice", json={"key": "x", "driver": "nope"},
                    headers=auth_header()).status_code == 400


def test_gas_worker_drivers_mode_rediscovers_on_a_choice(tmp_path):
    import queue

    from core.events import make_event
    from modules.gas.drivers_mode import DriverPoller
    from modules.gas.worker import GasWorker
    from core.orchestrator import missing_files

    config = {"gas": {"mode": "drivers", "sensors_state": "", "generated_model": "",
                      "classifier_path": str(tmp_path / "missing.joblib"), "regressor_path": "x"}}
    assert missing_files("gas", config) == []                 # models are only needed for a detected board
    worker = GasWorker(queue.Queue(), queue.Queue(), config)
    runs = []

    def factory():
        runs.append(1)
        return setup({0x74: dfrobot()})[0]

    worker.poller = DriverPoller(config["gas"], worker.emit, discovery_factory=factory)
    worker.setup()
    worker.step()
    event = make_event("system", "gas_sensor_choice", {"key": "i2c-1:0x76", "driver": "dfrobot_gravity"})
    worker.handle_control(event.to_json())
    worker.teardown()
    assert len(runs) == 2
    published = []
    while not worker.event_queue.empty():
        published.append(json.loads(worker.event_queue.get()))
    reading = next(e for e in published if e["type"] == "gas_reading")
    assert reading["payload"]["readings"][0]["unit"] == "ppm" and reading["device_id"].startswith("gas-dfrobot")
