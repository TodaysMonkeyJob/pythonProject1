"""
Gas module, ``gas.mode: drivers`` (A7 / TOD-48): the sensors connected to this Raspberry Pi are found automatically
(``drivers.gas.discovery``) and each is read in its native unit.

* start-up: discovery → ``gas.sensors_state`` (JSON for the dashboard: sensors, unknown devices, errors) →
  ``modules.hfs.gas_model_builder`` → ``gas.generated_model`` (``config/fuzzy/gas_generated.yaml``);
* every sensor is polled at its quantities' ``sample_period``; one ``gas_reading`` per sensor and poll, device =
  the sensor (``Descriptor.device_id``)::

      {"readings": [{"gas": "ammonia", "concentration": 42.0, "unit": "ppm", "range": [0, 100],
                     "resolution": 1.0, "accuracy": {"relative": 0.1, "absolute": 5.0}, "saturated": false}],
       "driver": "dfrobot_gravity", "sensor_model": "DFRobot SEN0469 (NH3)", "sensor_id": "gas-dfrobot…"}

  the graphene board's reading keeps the TOD-19 fields (``gas``, ``gas_confidence``, ``concentration_pct`` in
  vol % as the trained model outputs it, ``curve``) plus ``unit: vol_pct`` and the same ``readings`` list;
* index-only quantities → ``air_quality_index`` events (never mixed with concentrations);
* a read error → ``alert`` ``sensor_error`` (warning) for that sensor, the others go on; ``gas_sensor_choice`` (the
  dashboard saved a driver for an unknown device) or ``rediscover_sec`` → discovery again.
"""
import time
from typing import Any, Callable, Dict, List, Optional

from system_setup.logger_setup import init_logger

logger = init_logger()


# C1 (TOD-54): calibration kind of each driver (config.yaml sensor_calibration.kinds)
CALIBRATION_KIND = {"graphene_fet": "graphene_fet", "mq_ads1115": "mos", "dfrobot_gravity": "electrochemical",
                    "dfrobot_gravity_uart": "electrochemical", "pms5003": "particle", "sds011": "particle",
                    "sps30": "particle", "ens160": "index_only"}


class DriverPoller:
    def __init__(self, gas_config: Dict[str, Any], emit: Callable[..., Any],
                 clock: Callable[[], float] = time.monotonic, discovery_factory: Optional[Callable[[], Any]] = None,
                 config: Optional[Dict[str, Any]] = None, store=None):
        """
        :param emit: ``BaseModule.emit`` (type, payload, severity=, confidence=, device_id=)
        :param discovery_factory: → a ``Discovery`` (tests pass one with fake buses)
        :param config: the full config (C1 / TOD-54 ``sensor_calibration``; without it calibration is off)
        :param store: a ``CalibrationStore`` (tests: in memory)
        """
        self.cfg, self.emit, self.clock = gas_config, emit, clock
        self.config, self.store = config or {}, store
        self.calibrators: Dict[str, Any] = {}           # device_id → calibrator
        self.discovery_factory = discovery_factory or self._discovery
        self.rediscover_sec = float(gas_config.get("rediscover_sec", 0) or 0)
        self.drivers: List[Any] = []
        self.due: Dict[str, float] = {}
        self.result: Dict[str, Any] = {"sensors": [], "unknown": [], "errors": []}
        self.discovered_at = float("-inf")
        self.discovery = None

    def _discovery(self):
        from drivers.gas.discovery import Discovery, load_config

        options = {"graphene_fet": {"classifier_path": self.cfg.get("classifier_path"),
                                    "regressor_path": self.cfg.get("regressor_path")}}
        return Discovery(load_config(self.cfg.get("sensors_config", "config/sensors.yaml")), driver_options=options)

    def discover(self) -> Dict[str, Any]:
        from drivers.gas.discovery import report, write_state
        from modules.hfs.gas_model_builder import build, write
        from modules.hfs.gas_normalize import GasTable

        self.close()
        self.discovery = self.discovery_factory()
        self.result = self.discovery.run()
        self.drivers = self.discovery.drivers(self.result)
        self.discovered_at = self.clock()
        state = self.cfg.get("sensors_state", "data/gas_sensors.json")
        if state:
            write_state(self.result, state)
        generated = self.cfg.get("generated_model", "config/fuzzy/gas_generated.yaml")
        if generated:
            write(build(report(self.result)["sensors"], GasTable.load(self.cfg.get("gases", "config/gases.yaml"))),
                  generated)
        self._calibrators()
        found = ", ".join(f"{d.descriptor.model} ({', '.join(sorted({q.unit for q in d.descriptor.quantities}))})"
                          for d in self.drivers)
        logger.info(f"gas sensors: {found or 'none'}; unknown: {len(self.result['unknown'])}")
        return self.result

    def close(self) -> None:
        for driver in self.drivers:
            try:
                driver.close()
            except Exception:
                pass
        self.drivers, self.due = [], {}

    def _calibrators(self) -> None:
        """C1 (TOD-54): one calibrator per identified device; uid = driver + serial + bus position + firmware"""
        from modules.calibration import Calibrator, GrapheneCalibrator, MosCalibrator, enabled

        self.calibrators = {}
        if not enabled(self.config) and self.store is None:
            return
        from core.sensor_calibration import device_uid

        for driver in self.drivers:
            desc = driver.descriptor
            kind = CALIBRATION_KIND.get(desc.driver, "factory")
            uid = device_uid(desc.driver, desc.serial or desc.model, f"{desc.bus}:{desc.address}",
                             desc.details.get("firmware"))
            device = desc.device_id

            def emit(payload, device=device):
                self.emit("calibration_status", payload, device_id=device)
            args = (self.config, uid, device, emit, self.clock, self.store)
            if kind == "graphene_fet":
                vg, gates = desc.details.get("vg") or [0.0, 1.5], int(desc.details.get("gates", 16))
                grid = [vg[0] + (vg[1] - vg[0]) * i / max(1, gates - 1) for i in range(gates)]
                calibrator = GrapheneCalibrator(*args, voltages=grid)
            elif kind == "mos":
                calibrator = MosCalibrator(*args, driver=driver)
            else:
                calibrator = Calibrator(*args, kind=kind)
            calibrator.start()
            self.calibrators[device] = calibrator

    def statuses(self) -> List[Dict[str, Any]]:
        return [c.status() for c in self.calibrators.values()]

    def handle_event(self, event) -> None:
        if event.type == "gas_sensor_choice":
            self.discover()
        elif event.type == "calibration_control":
            for calibrator in self.calibrators.values():
                calibrator.control(event.payload)

    def step(self) -> int:
        """:return: readings published"""
        now = self.clock()
        if self.rediscover_sec and now - self.discovered_at >= self.rediscover_sec:
            self.discover()
        count = 0
        for driver in self.drivers:
            desc = driver.descriptor
            if now < self.due.get(desc.device_id, 0.0):
                continue
            self.due[desc.device_id] = now + min((q.sample_period for q in desc.quantities), default=1.0)
            if self.poll(driver):
                count += 1
        return count

    def poll(self, driver) -> bool:
        desc = driver.descriptor
        try:
            values = driver.read()
        except Exception as e:                       # one broken sensor never stops the others
            logger.warning(f"gas sensor {desc.model} ({desc.device_id}): {e}")
            self.emit("alert", {"reason": "sensor_error", "sensor_id": desc.device_id, "driver": desc.driver,
                                "error": str(e)}, severity="warning", device_id=desc.device_id)
            return False
        calibrator = self.calibrators.get(desc.device_id)
        if calibrator is not None:
            calibrator.state.tick()
            if calibrator.kind == "mos":
                calibrator.observe(values.get("volts"))
            elif calibrator.kind == "graphene_fet":
                calibrator.observe(values["curve"], values["gas"])
        if desc.needs_model:
            return self._model_reading(desc, values)
        readings, indices = [], []
        for q in desc.quantities:
            value = values.get(q.key)
            if value is None:
                continue
            if q.kind == "index":
                indices.append({"quantity": q.key, "value": value, "unit": q.unit, "range": q.range})
                continue
            readings.append(self._item(q, q.gas, value))
        base = {"driver": desc.driver, "sensor_model": desc.model, "sensor_id": desc.device_id}
        if readings:
            self.emit("gas_reading", {"readings": readings, **base}, device_id=desc.device_id)
        if indices:
            self.emit("air_quality_index", {"indices": indices, **base,
                                            **({"validity": values["validity"]} if "validity" in values else {})},
                      device_id=desc.device_id)
        return bool(readings or indices)

    @staticmethod
    def _item(q, gas: str, value: float) -> Dict[str, Any]:
        from modules.hfs.gas_model_builder import saturated

        from modules.hfs.gas_normalize import flat_reading

        item = {"gas": gas, "concentration": value, "unit": q.unit, "range": q.range, "resolution": q.resolution,
                "accuracy": q.accuracy, "saturated": saturated(value, q.range, q.resolution),
                "selectivity": q.selectivity}
        flat = flat_reading({"readings": [item]})                   # TOD-53: «% від критичного рівня»
        return {**item, **({"pct_of_crit": flat["pct_of_crit"]} if flat.get("ratio") else {})}

    def _model_reading(self, desc, values: Dict[str, Any]) -> bool:
        """Graphene board: the trained model's gas + vol % as it outputs them (no conversion of its input)."""
        from modules.gas.worker import curve_summary

        gas = values["gas"]
        quantity = next((q for q in desc.quantities if q.gas == gas), None)
        payload = {"gas": gas, "gas_confidence": values["gas_confidence"], "concentration_pct": values["concentration"],
                   "unit": "vol_pct", "driver": desc.driver, "sensor_model": "graphene_fet",
                   "sensor_id": desc.device_id,
                   "readings": [] if quantity is None else [self._item(quantity, gas, values["concentration"])]}
        voltages = desc.details.get("vg") or [0.0, 1.5]
        gates = len(values["curve"])
        grid = [voltages[0] + (voltages[1] - voltages[0]) * i / max(1, gates - 1) for i in range(gates)]
        payload["curve"] = curve_summary(values["curve"], grid)
        if quantity is not None:                                        # TOD-53: the model's range starts ≥ C_crit
            from modules.gas.worker import model_note
            from modules.hfs.gas_normalize import default_table, flat_reading

            substance = default_table().get(gas)
            flat = flat_reading(payload)
            payload.update(ratio=flat["ratio"], pct_of_crit=flat["pct_of_crit"], detection_floor=quantity.range[0],
                           below_model_range=0 < values["concentration"] < quantity.range[0])
            if substance is not None:
                payload["model_note"] = model_note(substance, quantity.range[0])
        self.emit("gas_reading", payload, confidence=values["gas_confidence"], device_id=desc.device_id)
        return True
