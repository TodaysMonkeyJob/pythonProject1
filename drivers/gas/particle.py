"""
Optical particle sensors (A7 / TOD-48) — PM2.5 mass concentration, native unit **µg/m³**.

* **Plantower PMS5003** (UART 9600, sends on its own): 32-byte frames ``42 4D 00 1C`` + 13 big-endian words + a
  16-bit sum of the first 30 bytes; word 5 = PM2.5 "atmospheric environment" (the CF = 1 words are for factory
  calibration). Range / accuracy not checked against the manual (docs/sensors.md) → left unknown.
* **Nova Fitness SDS011** (UART 9600, sends every second): 10-byte frames ``AA C0 <PM2.5 lo hi> <PM10 lo hi> <id 2>
  <sum of the 6 data bytes> AB``, values in 0.1 µg/m³; 0–999.9 µg/m³, 0.3 µg/m³, max of ±15 % and ±10 µg/m³
  (Nova datasheet copy, docs/sensors.md).
* **Sensirion SPS30** (I2C 0x69): 16-bit commands, every data word followed by CRC-8 (poly 0x31, init 0xFF); product
  type 0xD002 → "00080000", start measurement 0x0010 with 0x0300 (big-endian floats), data ready 0x0202, read 0x0300 →
  10 floats (mass PM1.0, PM2.5, PM4.0, PM10, number concentrations …), serial 0xD033 (Sensirion embedded-sps
  driver, checked 2026-11-07); 0–1000 µg/m³, ±10 µg/m³ (0–100) / ±10 % (100–1000) (docs/sensors.md).
"""
import struct
import time
from typing import Any, Dict, Optional

from drivers.gas.base import Descriptor, Driver, DriverError, Quantity

# --- Plantower PMS5003 -------------------------------------------------------------------------------------------
PMS_HEADER = b"\x42\x4d"


def find_pms_frame(data: bytes) -> Dict[str, float]:
    """:return: {pm1_0, pm2_5, pm10} (atmospheric, µg/m³) of the first checksum-valid frame; :raise DriverError:"""
    start = data.find(PMS_HEADER)
    while start != -1 and start + 32 <= len(data):
        frame = data[start:start + 32]
        words = struct.unpack(">16H", frame)
        if words[1] == 28 and sum(frame[:30]) & 0xFFFF == words[15]:
            return {"pm1_0": float(words[5]), "pm2_5": float(words[6]), "pm10": float(words[7])}
        start = data.find(PMS_HEADER, start + 1)
    raise DriverError("no valid PMS5003 frame")


class PMS5003(Driver):
    name = "pms5003"
    interface = "serial"
    bauds = (9600,)
    passive_header = PMS_HEADER

    @classmethod
    def probe(cls, port, address=None, options=None) -> Optional[Descriptor]:
        try:
            find_pms_frame(port.read(96))                      # ≥ 2 frames at the 1 s output rate
        except (OSError, DriverError):
            return None
        return Descriptor(cls.name, "Plantower PMS5003", getattr(port, "name", "?"), quantities=[
            Quantity("pm2_5", "pm2_5", "ug_m3", selectivity="selective",
                     source="Plantower PMS5003: range / accuracy not checked (docs/sensors.md) — unknown")])

    def read(self) -> Dict[str, Any]:
        self.bus.reset()
        return {"pm2_5": find_pms_frame(self.bus.read(96))["pm2_5"]}


# --- Nova SDS011 ---------------------------------------------------------------------------------------------------
SDS_HEADER = b"\xaa\xc0"


def find_sds_frame(data: bytes) -> Dict[str, Any]:
    start = data.find(SDS_HEADER)
    while start != -1 and start + 10 <= len(data):
        frame = data[start:start + 10]
        if frame[9] == 0xAB and sum(frame[2:8]) & 0xFF == frame[8]:
            return {"pm2_5": (frame[3] * 256 + frame[2]) / 10, "pm10": (frame[5] * 256 + frame[4]) / 10,
                    "id": f"{frame[6]:02x}{frame[7]:02x}"}
        start = data.find(SDS_HEADER, start + 1)
    raise DriverError("no valid SDS011 frame")


class SDS011(Driver):
    name = "sds011"
    interface = "serial"
    bauds = (9600,)
    passive_header = SDS_HEADER

    @classmethod
    def probe(cls, port, address=None, options=None) -> Optional[Descriptor]:
        try:
            frame = find_sds_frame(port.read(30))
        except (OSError, DriverError):
            return None
        return Descriptor(cls.name, "Nova Fitness SDS011", getattr(port, "name", "?"), serial=frame["id"],
                          quantities=[Quantity("pm2_5", "pm2_5", "ug_m3", selectivity="selective", range=[0.0, 999.9],
                                               resolution=0.3,
                                               accuracy={"relative": 0.15, "absolute": 10.0},
                                               source="Nova SDS011 datasheet copy (docs/sensors.md)")])

    def read(self) -> Dict[str, Any]:
        self.bus.reset()
        return {"pm2_5": find_sds_frame(self.bus.read(30))["pm2_5"]}


# --- Sensirion SPS30 -------------------------------------------------------------------------------------------------
SPS30_ADDRESS = 0x69
SPS_PRODUCT_TYPE, SPS_START, SPS_READY, SPS_READ, SPS_SERIAL = 0xD002, 0x0010, 0x0202, 0x0300, 0xD033


def crc8(data: bytes) -> int:
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def words(raw: bytes) -> bytes:
    """Strips and checks the CRC of every 2-byte word. :raise DriverError: a CRC mismatch"""
    out = bytearray()
    for i in range(0, len(raw) - 2, 3):
        chunk = raw[i:i + 2]
        if crc8(chunk) != raw[i + 2]:
            raise DriverError(f"SPS30 CRC mismatch at byte {i}")
        out += chunk
    return bytes(out)


def with_crc(*values: int) -> list:
    out = []
    for value in values:
        pair = [(value >> 8) & 0xFF, value & 0xFF]
        out += [*pair, crc8(bytes(pair))]
    return out


def cmd(code: int) -> list:
    return [(code >> 8) & 0xFF, code & 0xFF]


class SPS30(Driver):
    name = "sps30"
    interface = "i2c"
    addresses = (SPS30_ADDRESS,)

    @classmethod
    def probe(cls, bus, address=SPS30_ADDRESS, options=None) -> Optional[Descriptor]:
        try:
            product = words(bus.write_read(address, cmd(SPS_PRODUCT_TYPE), 12)).split(b"\0")[0].decode("ascii")
            if not product.startswith("00080000"):
                return None
            serial = words(bus.write_read(address, cmd(SPS_SERIAL), 48)).split(b"\0")[0].decode("ascii")
            bus.write(address, [*cmd(SPS_START), *with_crc(0x0300)])      # big-endian float output
        except (OSError, DriverError, UnicodeDecodeError):
            return None
        return Descriptor(cls.name, "Sensirion SPS30", getattr(bus, "name", "?"), address, serial=serial or None,
                          quantities=[Quantity("pm2_5", "pm2_5", "ug_m3", selectivity="selective", range=[0.0, 1000.0],
                                               accuracy={"relative": 0.10, "absolute": 10.0},
                                               source="Sensirion SPS30 datasheet copy / secondary table "
                                                      "(docs/sensors.md)")])

    def read(self) -> Dict[str, Any]:
        address = self.descriptor.address
        ready = words(self.bus.write_read(address, cmd(SPS_READY), 3))
        if ready[1] != 1:
            time.sleep(0.2)
        values = struct.unpack(">10f", words(self.bus.write_read(address, cmd(SPS_READ), 60)))
        return {"pm2_5": round(float(values[1]), 2)}
