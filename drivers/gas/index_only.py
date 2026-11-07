"""
Index-only air-quality sensors (A7 / TOD-48): no concentration of a named substance — a separate input with its own
terms, never mixed with concentrations.

**ScioSense ENS160** (I2C 0x52 / 0x53): part ID 0x0160 (ENS161: 0x0161) little-endian at register 0x00, OPMODE 0x10
(0x02 = standard), STATUS 0x20 (bits 3:2 validity: 0 normal, 1 warm-up, 2 initial start-up, 3 invalid), AQI-UBA
0x21 (1–5), TVOC 0x22 (ppb, little-endian), eCO2 0x24 (ppm equivalent, little-endian) (Adafruit CircuitPython ENS160
driver, checked 2026-11-07; the AQI is read as 3 bits because UBA levels go up to 5 — the Adafruit driver masks 2).
TVOC and eCO2 are estimates from metal-oxide resistances, not measured concentrations: kind ``index``.
"""
from typing import Any, Dict, Optional

from drivers.gas.base import Descriptor, Driver, Quantity

REG_PART_ID, REG_OPMODE, REG_STATUS, REG_AQI, REG_TVOC, REG_ECO2 = 0x00, 0x10, 0x20, 0x21, 0x22, 0x24
PART_IDS = {0x0160: "ScioSense ENS160", 0x0161: "ScioSense ENS161"}
STANDARD = 0x02
VALIDITY = {0: "normal", 1: "warm_up", 2: "start_up", 3: "invalid"}
SOURCE = "ScioSense ENS160: AQI-UBA 1–5, TVOC / eCO2 estimates (Adafruit driver registers)"


class ENS160(Driver):
    name = "ens160"
    interface = "i2c"
    addresses = (0x52, 0x53)

    @classmethod
    def probe(cls, bus, address=0x53, options=None) -> Optional[Descriptor]:
        try:
            part = int.from_bytes(bus.write_read(address, [REG_PART_ID], 2), "little")
            if part not in PART_IDS:
                return None
            bus.write(address, [REG_OPMODE, STANDARD])
        except OSError:
            return None
        return Descriptor(cls.name, PART_IDS[part], getattr(bus, "name", "?"), address, quantities=[
            Quantity("aqi_uba", "aqi_uba", "index", kind="index", range=[1.0, 5.0], resolution=1.0, source=SOURCE),
            Quantity("tvoc", "tvoc", "ppb", kind="index", range=[0.0, 65000.0], resolution=1.0, source=SOURCE),
            Quantity("eco2", "eco2", "ppm", kind="index", range=[400.0, 65000.0], resolution=1.0, source=SOURCE)])

    def read(self) -> Dict[str, Any]:
        address = self.descriptor.address
        status = self.bus.write_read(address, [REG_STATUS], 1)[0]
        validity = VALIDITY[(status >> 2) & 0x03]
        if validity != "normal":
            return {"validity": validity}                      # warming up: no values yet
        aqi = self.bus.write_read(address, [REG_AQI], 1)[0] & 0x07
        tvoc = int.from_bytes(self.bus.write_read(address, [REG_TVOC], 2), "little")
        eco2 = int.from_bytes(self.bus.write_read(address, [REG_ECO2], 2), "little")
        return {"aqi_uba": float(aqi), "tvoc": float(tvoc), "eco2": float(eco2), "validity": validity}
