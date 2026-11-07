"""
Our graphene FET board (A7 / TOD-48) — native unit **vol %**: the trained TOD-19 classifier / regressor
(``modules/gas/inference.GasPredictor``) works on the R(V_G) curve and outputs gas type + concentration in vol %;
nothing is converted — C_crit is expressed in vol % instead (NH₃ 300 ppm = 0.03 vol %, ``gas_model_builder``).

Board protocol (serial / USB, 115200 baud, one JSON object per line — what the board firmware must implement):

* ``ID?\\n`` → ``{"device": "graphene_fet", "serial": "<id>", "gates": 16, "vg": [0.0, 1.5], "unit": "vol_pct"}``
* ``READ\\n`` → ``{"r": [16 resistances, Ω, for V_G = 0.0 … 1.5 V]}``

Ranges: the dataset is 0.25–14 vol % (docs/sensors.md); accuracy = the regressor's MAE per gas
(``config/uncertainty.yaml`` ``gas.concentration.sensors.graphene_fet``, measured).
"""
import json
from typing import Any, Dict, List, Optional

import yaml

from drivers.gas.base import Descriptor, Driver, DriverError, Quantity

HELLO, READ = b"ID?\n", b"READ\n"
GASES = ("ethanol", "ammonia", "acetone", "toluene")       # the TOD-19 classes (toluol = toluene)
RANGE_VOL_PCT = [0.25, 14.0]


def board_line(port, request: bytes) -> Dict[str, Any]:
    port.reset()
    port.write(request)
    line = port.readline()
    try:
        return json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise DriverError(f"graphene board: not a JSON line ({line[:40]!r})") from e


def measured_accuracy(path: str = "config/uncertainty.yaml") -> Dict[str, Dict[str, float]]:
    """:return: gas → {absolute: MAE vol %} of the regressor (TOD-45 measured values)"""
    try:
        with open(path, encoding="utf-8") as f:
            per_gas = yaml.safe_load(f)["gas"]["concentration"]["sensors"]["graphene_fet"]["per_gas"]
    except (OSError, KeyError, TypeError):
        return {}
    return {("toluene" if gas == "toluol" else gas): {"absolute": float(v["absolute"])} for gas, v in per_gas.items()}


class GrapheneFET(Driver):
    name = "graphene_fet"
    interface = "serial"
    bauds = (115200,)

    @classmethod
    def probe(cls, port, address=None, options=None) -> Optional[Descriptor]:
        try:
            hello = board_line(port, HELLO)
        except (OSError, DriverError):
            return None
        if hello.get("device") != "graphene_fet":
            return None
        if hello.get("unit", "vol_pct") != "vol_pct":
            raise DriverError(f"graphene board reports {hello.get('unit')}; the trained model expects vol_pct")
        accuracy = measured_accuracy((options or {}).get("uncertainty", "config/uncertainty.yaml"))
        return Descriptor(cls.name, "graphene FET board", getattr(port, "name", "?"), serial=hello.get("serial"),
                          needs_model=True, quantities=[
                              Quantity(gas, gas, "vol_pct", range=list(RANGE_VOL_PCT), accuracy=accuracy.get(gas),
                                       sample_period=1.0, source="dataset range data/gas/data.xls; accuracy = "
                                                                 "regressor MAE (reports/gas_metrics.json)")
                              for gas in GASES],
                          details={"gates": hello.get("gates", 16), "vg": hello.get("vg", [0.0, 1.5]),
                                   "model_unit": "vol_pct"})

    def __init__(self, descriptor, bus, options=None):
        super().__init__(descriptor, bus, options)
        self.predictor = None

    def curve(self) -> List[float]:
        values = board_line(self.bus, READ).get("r")
        if not isinstance(values, list) or len(values) != int(self.descriptor.details.get("gates", 16)):
            raise DriverError(f"graphene board: expected {self.descriptor.details.get('gates', 16)} resistances")
        return [float(v) for v in values]

    def load_model(self):
        """The trained model and the unit it expects (vol %); loaded once, only for a detected board."""
        if self.predictor is None:
            from modules.gas.inference import GasPredictor

            self.predictor = GasPredictor(self.options.get("classifier_path", "models/gas_classifier.joblib"),
                                          self.options.get("regressor_path", "models/gas_concentration.joblib"))
        return self.predictor, "vol_pct"

    def read(self) -> Dict[str, Any]:
        """:return: {curve, gas, gas_confidence, concentration (vol %, as the model outputs it)}"""
        curve = self.curve()
        predictor, unit = self.load_model()
        prediction = predictor.predict(curve)
        gas = "toluene" if prediction["gas"] == "toluol" else prediction["gas"]
        return {"curve": curve, "gas": gas, "gas_confidence": prediction["gas_confidence"],
                "concentration": prediction["concentration_pct"], "unit": unit}
