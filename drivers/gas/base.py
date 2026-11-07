"""
Gas sensor driver interface (A7 / TOD-48).

A driver identifies one device on a bus and reads it in the sensor's **native unit** — the unit its model or its
factory calibration works in (vol % for our graphene FET, ppm for electrochemical modules, µg/m³ for particle
sensors, an index for index-only sensors). Nothing is converted here; ``modules/hfs`` compares substances by
c / C_crit and ``modules/hfs/gas_model_builder`` builds the per-sensor fuzzy inputs from the descriptors.

* ``probe(bus, address)`` (I2C) / ``probe(port)`` (serial) → ``Descriptor`` or None — only an answer that proves the
  device (an ID, a checksum-valid frame) counts; several drivers may share an I2C address;
* ``read()`` → ``{quantity key: value}`` (``needs_model`` drivers return the raw curve under ``"curve"``).

Descriptor::

    {driver, device_id, model, bus, address | port, quantities: [{key, gas, unit, kind, range, resolution,
     accuracy {relative, absolute} | null, sample_period, source}], needs_model, details}
"""
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

UNITS = ("vol_pct", "ppm", "mg_m3", "ug_m3", "index", "ohm", "ppb")
CONCENTRATION_UNITS = ("vol_pct", "ppm", "mg_m3", "ug_m3")


@dataclass
class Quantity:
    key: str                                # what read() returns it under
    gas: str                                # substance (config/gases.yaml) or the index name
    unit: str
    kind: str = "concentration"             # concentration | index
    range: Optional[List[float]] = None     # [low, high] in unit; None = not published / not checked
    resolution: Optional[float] = None
    accuracy: Optional[Dict[str, float]] = None   # {relative: share of reading, absolute: in unit}; None = unknown
    sample_period: float = 1.0              # s
    source: str = ""                        # where range / accuracy come from

    def __post_init__(self):
        if self.unit not in UNITS:
            raise ValueError(f"{self.key}: unit {self.unit!r} not in {UNITS}")
        if self.kind == "concentration" and self.unit not in CONCENTRATION_UNITS:
            raise ValueError(f"{self.key}: a concentration needs a concentration unit, got {self.unit}")


@dataclass
class Descriptor:
    driver: str
    model: str
    bus: str                                # "i2c-1" | serial port path
    address: Optional[int] = None           # I2C address
    serial: Optional[str] = None            # the device's own serial number, when it reports one
    quantities: List[Quantity] = field(default_factory=list)
    needs_model: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def device_id(self) -> str:
        """Stable: driver + the device's serial, else its bus position."""
        if self.serial:
            where = f"{self.model}:{self.serial}"
        elif self.address is not None:
            where = f"{self.bus}:{self.address:#04x}"
        else:
            where = self.bus
        return f"gas-{self.driver}-{hashlib.sha1(str(where).encode()).hexdigest()[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "device_id": self.device_id}


class Driver:
    """One sensor family. Subclasses set ``name``, ``interface`` and ``addresses`` (I2C) / ``bauds`` (serial)."""
    name = "driver"
    interface = "i2c"                       # i2c | serial
    addresses: tuple = ()
    bauds: tuple = ()
    passive_header: Optional[bytes] = None  # serial: frames the device sends on its own (listen first)
    identifies = True                       # False: cannot prove the device — only via config/sensors.yaml

    def __init__(self, descriptor: Descriptor, bus: Any, options: Optional[Dict[str, Any]] = None):
        self.descriptor, self.bus, self.options = descriptor, bus, options or {}

    @classmethod
    def probe(cls, bus: Any, address: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Optional[Descriptor]:
        raise NotImplementedError

    def read(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        close = getattr(self.bus, "close", None)
        if self.interface == "serial" and close is not None:
            close()


class DriverError(RuntimeError):
    """A device that answered wrongly (bad checksum, wrong header) or not at all."""
