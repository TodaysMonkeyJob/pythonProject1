"""
MOS sensors with an analog output (Winsen MQ series) through an ADS1115 ADC on I2C (A7 / TOD-48) — native unit
**ppm**, approximate.

The Pi has no ADC, and an ADS1115 has no ID register: it cannot say which MQ sensor is wired to which channel. So
discovery lists an ADS1115 address as an **unknown** device; the user declares each channel in
``config/sensors.yaml`` (or picks it on the dashboard)::

    {driver: mq_ads1115, bus: i2c-1, address: 0x48, channel: 0, model: MQ-137, gas: ammonia, range: [5, 500],
     curve: {a: …, b: …}, r0_ohm: …, rl_ohm: 10000, vc: 5.0, divider: 1.0}

Reading: single-shot conversion (±4.096 V, 128 SPS) → V_out (× ``divider`` when a resistor divider scales the 5 V
module output down to the ADC) → R_s = R_L · (V_c − V_out) / V_out → ppm = a · (R_s / R_0)^b — the log-log curve of
the datasheet plot, R_0 = R_s in clean air of that sensor (its own calibration). There are no default curve values:
the accuracy is unknown until the sensor is calibrated against a reference.
"""
import time
from typing import Any, Dict, Optional

from drivers.gas.base import Descriptor, Driver, DriverError, Quantity

ADS1115_ADDRESSES = (0x48, 0x49, 0x4A, 0x4B)
REG_CONVERSION, REG_CONFIG = 0x00, 0x01
FULL_SCALE_V = 4.096


def config_word(channel: int) -> int:
    """OS = start, MUX = AINx vs GND, PGA ±4.096 V, single shot, 128 SPS, comparator off."""
    if channel not in (0, 1, 2, 3):
        raise ValueError("ADS1115 channel must be 0..3")
    return (1 << 15) | ((0b100 + channel) << 12) | (0b001 << 9) | (1 << 8) | (0b100 << 5) | 0b11


def ppm_from_volts(v_out: float, options: Dict[str, Any]) -> float:
    if v_out <= 0:
        raise DriverError("MQ output 0 V (heater off or not wired)")
    vc, rl, r0 = float(options.get("vc", 5.0)), float(options.get("rl_ohm", 10_000.0)), float(options["r0_ohm"])
    rs = rl * (vc - v_out) / v_out
    curve = options["curve"]
    return max(0.0, float(curve["a"]) * (rs / r0) ** float(curve["b"]))


class MQAds1115(Driver):
    name = "mq_ads1115"
    interface = "i2c"
    addresses = ADS1115_ADDRESSES
    identifies = False                       # an ADC: the MQ sensor behind it must be declared

    @classmethod
    def probe(cls, bus, address=None, options=None) -> Optional[Descriptor]:
        """Only with a declaration (``options``): checks that the ADC answers and builds the descriptor."""
        if not options:
            return None
        missing = [k for k in ("gas", "curve", "r0_ohm") if k not in options]
        if missing:
            raise DriverError(f"mq_ads1115 needs {missing} in config/sensors.yaml")
        if not float(options["r0_ohm"]) > 0 or not float(options["curve"].get("a", 0)) > 0:
            raise DriverError("mq_ads1115: r0_ohm and curve.a must be > 0 (calibrate the sensor first)")
        try:
            bus.write_read(address, [REG_CONFIG], 2)
        except OSError:
            return None
        channel = int(options.get("channel", 0))
        model = str(options.get("model", "MQ"))
        return Descriptor(cls.name, f"{model} via ADS1115 AIN{channel}", getattr(bus, "name", "?"), address,
                          serial=f"{address:#04x}/ain{channel}", quantities=[Quantity(
                              "concentration", str(options["gas"]), "ppm", range=options.get("range"),
                              accuracy=options.get("accuracy"), sample_period=1.0,
                              source=f"{model} datasheet curve a={options['curve']['a']}, b={options['curve']['b']}, "
                                     f"R0={options['r0_ohm']} Ω (own calibration); accuracy "
                                     + ("declared" if options.get("accuracy") else "unknown"))],
                          details={"channel": channel})

    def read(self) -> Dict[str, Any]:
        address, channel = self.descriptor.address, self.descriptor.details["channel"]
        word = config_word(channel)
        self.bus.write(address, [REG_CONFIG, word >> 8, word & 0xFF])
        time.sleep(0.01)
        raw = self.bus.write_read(address, [REG_CONVERSION], 2)
        value = int.from_bytes(raw, "big", signed=True)
        v_out = value * FULL_SCALE_V / 32768 * float(self.options.get("divider", 1.0))
        return {"concentration": round(ppm_from_volts(v_out, self.options), 2), "volts": round(v_out, 4)}
