"""
Factory-calibrated electrochemical modules, DFRobot Gravity series (A7 / TOD-48): NH₃ SEN0469 (0–100 ppm), CO
SEN0466 (0–1000 ppm) and the other gases of the same board — native unit **ppm**.

Protocol, from DFRobot's own library (github.com/DFRobot/DFRobot_MultiGasSensor, checked 2026-11-07): 9-byte frames
``FF 01 <cmd> d1..d5 <check>``, check = two's complement of the sum of bytes 1..7; command 0x78 switches the upload
mode (0x04 = passive: the host asks), 0x86 reads the concentration — answer ``FF 86 <hi> <lo> <gas type>
<decimals> … <check>``, concentration = (hi·256 + lo) · 10^−decimals. I2C addresses 0x74–0x77 (DIP switches; the
frame is written to register 0 and 9 bytes are read back), or UART (baud from ``config/sensors.yaml``, default 9600
as in DFRobot's examples). The gas type byte identifies the module, so it shares an I2C address with nothing
silently: a reply without a valid checksum is not this device.
"""
import time
from typing import Any, Dict, List, Optional

from drivers.gas.base import Descriptor, Driver, DriverError, Quantity

CMD_CHANGE_GET_METHOD, CMD_GET_GAS_CONCENTRATION = 0x78, 0x86
PASSIVE = 0x04
# gas type byte → (substance as in config/gases.yaml, module, range ppm, resolution, accuracy, source)
GAS_TYPES = {
    0x02: ("ammonia", "DFRobot SEN0469 (NH3)", [0.0, 100.0], 1.0, {"relative": 0.10, "absolute": 5.0},
           "DFRobot wiki SEN0469: 0–100 ppm, 1 ppm; ±10 % of reading or ±5 % FS from a DFRobot forum reply "
           "(docs/sensors.md) — confirm against the datasheet"),
    0x04: ("carbon_monoxide", "DFRobot SEN0466 (CO)", [0.0, 1000.0], 1.0, None,
           "DFRobot wiki SEN0466: 0–1000 ppm, 1 ppm; accuracy not published"),
    0x03: ("hydrogen_sulfide", "DFRobot Gravity H2S", None, None, None, "gas type 0x03 (library); range unknown"),
    0x05: ("oxygen", "DFRobot Gravity O2", None, None, None, "gas type 0x05 (library; % vol, not a hazard input)"),
    0x06: ("hydrogen", "DFRobot Gravity H2", None, None, None, "gas type 0x06 (library)"),
    0x2A: ("ozone", "DFRobot Gravity O3", None, None, None, "gas type 0x2A (library)"),
    0x2B: ("sulfur_dioxide", "DFRobot Gravity SO2", None, None, None, "gas type 0x2B (library)"),
    0x2C: ("nitrogen_dioxide", "DFRobot Gravity NO2", None, None, None, "gas type 0x2C (library)"),
    0x2E: ("hydrogen_chloride", "DFRobot Gravity HCl", None, None, None, "gas type 0x2E (library)"),
    0x31: ("chlorine", "DFRobot Gravity Cl2", None, None, None, "gas type 0x31 (library)"),
    0x33: ("hydrogen_fluoride", "DFRobot Gravity HF", None, None, None, "gas type 0x33 (library)"),
    0x45: ("phosphine", "DFRobot Gravity PH3", None, None, None, "gas type 0x45 (library)"),
}


def checksum(frame) -> int:
    return (~sum(frame[1:8]) + 1) & 0xFF


def command(cmd: int, *args: int) -> List[int]:
    frame = [0xFF, 0x01, cmd, *args, *([0] * (5 - len(args))), 0]
    frame[8] = checksum(frame)
    return frame


def parse_concentration(reply: bytes) -> Dict[str, Any]:
    """:return: {ppm, gas_type} of a 0x86 answer; :raise DriverError: not a valid answer"""
    if len(reply) != 9 or reply[0] != 0xFF or reply[1] != CMD_GET_GAS_CONCENTRATION or checksum(reply) != reply[8]:
        raise DriverError(f"not a DFRobot concentration frame: {bytes(reply).hex()}")
    value = (reply[2] << 8) + reply[3]
    scale = {1: 0.1, 2: 0.01}.get(reply[5], 1.0)
    return {"ppm": round(value * scale, 3), "gas_type": reply[4]}


class DFRobotGravity(Driver):
    name = "dfrobot_gravity"
    interface = "i2c"
    addresses = (0x74, 0x75, 0x76, 0x77)

    @staticmethod
    def exchange(bus, address: Optional[int], frame: List[int]) -> bytes:
        if address is None:                                   # UART
            bus.reset()
            bus.write(bytes(frame))
            return bus.read(9)
        return _i2c(bus, address, frame)

    @classmethod
    def probe(cls, bus, address=None, options=None) -> Optional[Descriptor]:
        try:
            cls.exchange(bus, address, command(CMD_CHANGE_GET_METHOD, PASSIVE))
            reading = parse_concentration(cls.exchange(bus, address, command(CMD_GET_GAS_CONCENTRATION)))
        except (OSError, DriverError):
            return None
        known = GAS_TYPES.get(reading["gas_type"])
        if known is None:
            return None
        gas, model, rng, resolution, accuracy, source = known
        return Descriptor(cls.name, model, getattr(bus, "name", "?"), address, quantities=[
            Quantity("concentration", gas, "ppm", range=rng, resolution=resolution, accuracy=accuracy,
                     sample_period=1.0, source=source, selectivity="selective")],
            details={"gas_type": reading["gas_type"]})

    def read(self) -> Dict[str, Any]:
        reading = parse_concentration(self.exchange(self.bus, self.descriptor.address,
                                                    command(CMD_GET_GAS_CONCENTRATION)))
        if reading["gas_type"] != self.descriptor.details.get("gas_type"):
            raise DriverError(f"gas type changed: {reading['gas_type']:#04x}")
        return {"concentration": reading["ppm"]}


def _i2c(bus, address: int, frame: List[int]) -> bytes:
    bus.write(address, [0x00, *frame])                       # register 0 + the frame
    time.sleep(0.05)
    bus.write(address, [0x00])
    return bus.read(address, 9)


class DFRobotGravityUART(DFRobotGravity):
    name = "dfrobot_gravity_uart"
    interface = "serial"
    addresses = ()
    bauds = (9600,)
