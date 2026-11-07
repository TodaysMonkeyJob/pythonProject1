"""
Gas sensor discovery (A7 / TOD-48): which sensors are on this Raspberry Pi, what they measure and in which unit.

1. **I2C**: every bus (``/dev/i2c-*`` or ``buses`` in ``config/sensors.yaml``), addresses 0x08–0x77 checked for an
   ACK; for each address that answers, every driver listing it is asked to identify the device (an ID register, a
   product-type command, a checksum-valid answer). Exactly one driver → the device; none or several → **unknown**
   (several drivers share addresses: DFRobot gravity 0x74–0x77 vs others; an ADS1115 ADC can't identify the MQ
   sensor behind it).
2. **Serial / USB** (``/dev/serial0``, ``/dev/ttyAMA0``, ``/dev/ttyUSB*``, ``/dev/ttyACM*``): per baud rate of the
   drivers, first listen (sensors that send frames on their own: PMS5003 ``42 4D``, SDS011 ``AA C0``), then ask (our
   graphene board's ``ID?`` handshake, the DFRobot UART query). Nothing recognised → unknown port (only listed when
   something was received).
3. ``config/sensors.yaml``: ``devices`` declared by hand (driver + bus + address / port + driver options) are probed
   with that driver only; ``choices`` = drivers the user picked for unknown devices on the dashboard (key
   ``i2c-1:0x48`` / the port path) — saved by ``save_choice``.

Result: ``{"sensors": [descriptor …], "unknown": [{key, bus, address | port, candidates, reason}], "errors": […]}``.
"""
import datetime as dt
import json
import os
from typing import Any, Callable, Dict, List, Optional

import yaml

from drivers.gas import DRIVERS
from drivers.gas.base import Descriptor, Driver, DriverError
from drivers.gas.buses import I2CBus, i2c_buses, serial_ports
from system_setup.logger_setup import init_logger

logger = init_logger()

I2C_RANGE = range(0x08, 0x78)
DEFAULT_CONFIG = "config/sensors.yaml"


def key_of(bus: str, address: Optional[int] = None) -> str:
    return f"{bus}:{address:#04x}" if address is not None else bus


def parse_address(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value, 0) if isinstance(value, str) else int(value)


def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {"devices": [], "choices": {}}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {"devices": list(data.get("devices") or []), "choices": dict(data.get("choices") or {}),
            **{k: v for k, v in data.items() if k not in ("devices", "choices")}}


def save_choice(key: str, driver: str, options: Optional[Dict[str, Any]] = None, path: str = DEFAULT_CONFIG,
                by: str = "dashboard") -> Dict[str, Any]:
    """The user's driver for an unknown device → ``choices`` of config/sensors.yaml (comments are not kept)."""
    if driver not in DRIVERS:
        raise ValueError(f"unknown driver {driver!r} (one of {', '.join(DRIVERS)})")
    data = {"devices": [], "choices": {}}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or data
    data.setdefault("choices", {})
    data["choices"][key] = {"driver": driver, **({"options": options} if options else {}),
                            "by": by, "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# Gas sensors (TOD-48): devices declared by hand + drivers chosen for unknown devices "
                "(docs/sensors.md)\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)
    return data["choices"][key]


class Listening:
    """A serial port that remembers whether anything was received (an unknown device vs an empty port)."""

    def __init__(self, port):
        self.port, self.heard = port, False
        self.name = getattr(port, "name", "?")

    def read(self, n: int) -> bytes:
        data = self.port.read(n)
        self.heard = self.heard or bool(data)
        return data

    def readline(self, limit: int = 4096) -> bytes:
        data = self.port.readline(limit)
        self.heard = self.heard or bool(data)
        return data

    def write(self, data: bytes) -> None:
        self.port.write(data)

    def reset(self) -> None:
        self.port.reset()

    def close(self) -> None:
        self.port.close()


class Discovery:
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 open_i2c: Callable[[int], I2CBus] = None, open_serial: Callable[[str, int], Any] = None,
                 i2c_numbers: Optional[List[int]] = None, ports: Optional[List[str]] = None,
                 driver_options: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        :param config: the content of config/sensors.yaml (``load_config``)
        :param open_i2c: bus number → I2CBus (default smbus2); tests pass fakes
        :param open_serial: (port, baud) → SerialPort (default pyserial)
        :param driver_options: driver name → options for every device of that driver (e.g. model paths)
        """
        self.config = config or {"devices": [], "choices": {}}
        self.open_i2c = open_i2c or _smbus
        self.open_serial = open_serial or _pyserial
        self.i2c_numbers = i2c_numbers if i2c_numbers is not None else self.config.get("buses") or i2c_buses()
        self.ports = ports if ports is not None else self.config.get("ports") or serial_ports()
        self.driver_options = driver_options or {}
        self.handles: Dict[str, Any] = {}                      # bus / port name → open handle for the drivers

    def options(self, driver: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {**self.driver_options.get(driver, {}), **(extra or {})}

    # --- I2C ------------------------------------------------------------------------------------------------------
    def scan_i2c(self, result: Dict[str, Any], taken: set) -> None:
        for number in self.i2c_numbers:
            try:
                bus = self.open_i2c(number)
            except Exception as e:                          # no /dev/i2c-N, no permission
                result["errors"].append(f"i2c-{number}: {e}")
                continue
            name = getattr(bus, "name", f"i2c-{number}")
            self.handles[name] = bus
            for address in I2C_RANGE:
                key = key_of(name, address)
                if key in taken or not bus.present(address):
                    continue
                choice = self.config["choices"].get(key)
                if choice:
                    self._with_driver(result, choice["driver"], bus, address, choice.get("options"), key)
                    continue
                candidates = [d for d in DRIVERS.values() if d.interface == "i2c" and address in d.addresses]
                found = [(d, desc) for d in candidates if d.identifies
                         for desc in [self._probe(d, bus, address, None, result)] if desc is not None]
                if len(found) == 1:
                    result["sensors"].append(found[0][1])
                else:
                    reason = ("several drivers answer" if found else "no driver identified it" if candidates
                              else "no gas driver uses this address")
                    result["unknown"].append({"key": key, "bus": name, "address": address,
                                              "candidates": [d.name for d in candidates], "reason": reason})

    # --- serial ---------------------------------------------------------------------------------------------------
    def scan_serial(self, result: Dict[str, Any], taken: set) -> None:
        bauds = sorted({b for d in DRIVERS.values() if d.interface == "serial" for b in d.bauds})
        for path in self.ports:
            if path in taken:
                continue
            choice = self.config["choices"].get(path)
            if choice:
                driver = DRIVERS.get(choice["driver"])
                baud = int((choice.get("options") or {}).get("baud", driver.bauds[0] if driver and driver.bauds
                                                            else 9600))
                self._serial_with(result, choice["driver"], path, baud, choice.get("options"), path)
                continue
            heard, found = False, None
            for baud in bauds:
                drivers = [d for d in DRIVERS.values() if d.interface == "serial" and baud in d.bauds]
                ordered = sorted(drivers, key=lambda d: d.passive_header is None)    # listen first, then ask
                try:
                    port = Listening(self.open_serial(path, baud))
                except Exception as e:
                    result["errors"].append(f"{path}: {e}")
                    break
                for driver in ordered:
                    desc = self._probe(driver, port, None, None, result)
                    if desc is not None:
                        found = desc
                        self.handles[path] = port.port
                        break
                heard = heard or port.heard
                if found:
                    break
                port.close()
            if found:
                result["sensors"].append(found)
            elif heard:
                result["unknown"].append({"key": path, "bus": path, "port": path, "candidates": [
                    d.name for d in DRIVERS.values() if d.interface == "serial"], "reason": "data, but no driver "
                                                                                             "recognised it"})

    # --- declared / chosen ----------------------------------------------------------------------------------------
    def _probe(self, driver: type, bus, address, options, result) -> Optional[Descriptor]:
        try:
            return driver.probe(bus, address, self.options(driver.name, options))
        except DriverError as e:
            result["errors"].append(f"{driver.name} at {key_of(getattr(bus, 'name', '?'), address)}: {e}")
            return None

    def _with_driver(self, result, name: str, bus, address, options, key) -> None:
        driver = DRIVERS.get(name)
        desc = self._probe(driver, bus, address, options, result) if driver else None
        if desc is None:
            result["unknown"].append({"key": key, "bus": getattr(bus, "name", "?"), "address": address,
                                      "candidates": [name], "reason": f"chosen driver {name} did not answer"})
        else:
            desc.details["declared"] = True
            desc.details["options"] = dict(options or {})          # the driver instance reads them (curve, R0 …)
            result["sensors"].append(desc)

    def _serial_with(self, result, name, path, baud, options, key) -> None:
        try:
            port = self.open_serial(path, baud)
        except Exception as e:
            result["errors"].append(f"{path}: {e}")
            return
        self.handles[path] = port
        self._with_driver(result, name, port, None, options, key)

    def declared(self, result: Dict[str, Any], taken: set) -> None:
        for entry in self.config["devices"]:
            name, bus = entry.get("driver"), str(entry.get("bus", ""))
            driver = DRIVERS.get(name)
            if driver is None:
                result["errors"].append(f"config/sensors.yaml: unknown driver {name!r}")
                continue
            options = {k: v for k, v in entry.items() if k not in ("driver", "bus", "address", "port")}
            if driver.interface == "i2c":
                address = parse_address(entry.get("address"))
                number = int(bus.rsplit("-", 1)[-1]) if bus.startswith("i2c-") else 1
                handle = self.handles.get(f"i2c-{number}")
                if handle is None:
                    try:
                        handle = self.handles[f"i2c-{number}"] = self.open_i2c(number)
                    except Exception as e:
                        result["errors"].append(f"i2c-{number}: {e}")
                        continue
                taken.add(key_of(getattr(handle, "name", f"i2c-{number}"), address))
                self._with_driver(result, name, handle, address, options, key_of(handle.name, address))
            else:
                port = str(entry.get("port") or bus)
                taken.add(port)
                self._serial_with(result, name, port, int(options.pop("baud", driver.bauds[0])), options, port)

    def run(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"sensors": [], "unknown": [], "errors": [],
                                  "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        taken: set = set()
        self.declared(result, taken)
        self.scan_i2c(result, taken)
        self.scan_serial(result, taken)
        return result

    def drivers(self, result: Dict[str, Any]) -> List[Driver]:
        """:return: one driver instance per detected sensor (sharing the open bus / port handles)"""
        out = []
        for desc in result["sensors"]:
            handle = self.handles.get(desc.bus)
            options = {**self.options(desc.driver), **desc.details.get("options", {})}
            out.append(DRIVERS[desc.driver](desc, handle, options))
        return out


def report(result: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-ready (dashboard / ``gas.sensors_state``)."""
    return {**result, "sensors": [d.to_dict() for d in result["sensors"]],
            "unknown": [{**u, **({"address_hex": f"{u['address']:#04x}"} if u.get("address") is not None else {})}
                        for u in result["unknown"]],
            "drivers": [{"name": n, "interface": d.interface, "identifies": d.identifies} for n, d in DRIVERS.items()]}


def write_state(result: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report(result), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _smbus(number: int):
    from drivers.gas.buses import SMBus2I2C

    return SMBus2I2C(number)


def _pyserial(path: str, baud: int):
    from drivers.gas.buses import PySerialPort

    return PySerialPort(path, baud, timeout=1.5)


def main(argv=None) -> int:
    """``python -m drivers.gas.discovery`` — print what is connected (run on the Pi)."""
    import argparse

    parser = argparse.ArgumentParser(description="Detect gas sensors on I2C / serial")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = Discovery(load_config(args.config)).run()
    print(json.dumps(report(result), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
