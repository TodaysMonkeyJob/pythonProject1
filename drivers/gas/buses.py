"""
Bus access for the gas drivers (A7 / TOD-48). Drivers talk to these small interfaces only, so tests replace them
with recorded frames (``tests/test_gas_drivers.py``) and the real ones import ``smbus2`` / ``pyserial`` lazily (on the
Raspberry Pi: ``sudo raspi-config`` → Interface Options → I2C / Serial Port, user in the ``i2c`` / ``dialout`` groups).

* ``I2CBus``: ``write(address, data)``, ``read(address, n)``, ``write_read(address, data, n)`` (repeated start),
  ``present(address)`` (an ACK to a zero-length / one-byte read);
* ``SerialPort``: ``write(data)``, ``read(n)`` (returns what arrived within ``timeout``), ``reset()``, ``close()``.
"""
import glob
import os
import time
from typing import List, Optional, Sequence


class I2CBus:
    name = "i2c"

    def write(self, address: int, data: Sequence[int]) -> None:
        raise NotImplementedError

    def read(self, address: int, n: int) -> bytes:
        raise NotImplementedError

    def write_read(self, address: int, data: Sequence[int], n: int, delay: float = 0.0) -> bytes:
        self.write(address, data)
        if delay:
            time.sleep(delay)
        return self.read(address, n)

    def present(self, address: int) -> bool:
        try:
            self.read(address, 1)
            return True
        except OSError:
            return False


class SMBus2I2C(I2CBus):
    """``/dev/i2c-<number>`` through smbus2 (raw i2c_msg transfers: the sensors use command words, not SMBus)."""

    def __init__(self, number: int = 1):
        from smbus2 import SMBus

        self.number, self.name = number, f"i2c-{number}"
        self.smbus = SMBus(number)

    def write(self, address: int, data: Sequence[int]) -> None:
        from smbus2 import i2c_msg

        self.smbus.i2c_rdwr(i2c_msg.write(address, list(data)))

    def read(self, address: int, n: int) -> bytes:
        from smbus2 import i2c_msg

        msg = i2c_msg.read(address, n)
        self.smbus.i2c_rdwr(msg)
        return bytes(msg)

    def close(self) -> None:
        self.smbus.close()


class SerialPort:
    name = "serial"
    timeout = 1.0

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def read(self, n: int) -> bytes:
        raise NotImplementedError

    def reset(self) -> None:
        """Drops what is buffered."""

    def close(self) -> None:
        pass

    def readline(self, limit: int = 4096) -> bytes:
        out = bytearray()
        while len(out) < limit:
            chunk = self.read(1)
            if not chunk:
                break
            out += chunk
            if chunk == b"\n":
                break
        return bytes(out)


class PySerialPort(SerialPort):
    def __init__(self, path: str, baud: int, timeout: float = 1.0):
        import serial

        self.name, self.baud, self.timeout = path, baud, timeout
        self.port = serial.Serial(path, baudrate=baud, timeout=timeout)

    def write(self, data: bytes) -> None:
        self.port.write(data)
        self.port.flush()

    def read(self, n: int) -> bytes:
        return self.port.read(n)

    def reset(self) -> None:
        self.port.reset_input_buffer()

    def close(self) -> None:
        self.port.close()


SERIAL_PATTERNS = ("/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0", "/dev/ttyUSB*", "/dev/ttyACM*")


def serial_ports(patterns: Optional[Sequence[str]] = None) -> List[str]:
    """:return: candidate serial ports (Pi UART + USB adapters), symlinks resolved, each once"""
    seen, out = set(), []
    for pattern in patterns or SERIAL_PATTERNS:
        for path in sorted(glob.glob(pattern)):
            real = os.path.realpath(path)
            if real not in seen:
                seen.add(real)
                out.append(path)
    return out


def i2c_buses() -> List[int]:
    return sorted(int(p.rsplit("-", 1)[1]) for p in glob.glob("/dev/i2c-*") if p.rsplit("-", 1)[1].isdigit())
