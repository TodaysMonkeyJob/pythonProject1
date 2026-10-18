"""
Locations module (F12 / TOD-23, own process): feeds every bus event into ``LocationStateEngine``, publishes
``location_state_changed`` on transitions, notifies (dashboard / browser / Telegram) and pushes ``location_summary``
objects to the dashboard every ``summary_interval_sec`` while something changes (and every 10 s anyway).
``notification_acked`` events (the dashboard's Acknowledge) clear a location's alerts.
"""
import time
from typing import Any, Dict, Optional

from core.events import Event, make_event
from core.module import BaseModule
from modules.locations.state import ATTENTION, DANGER, OFFLINE, LocationStateEngine
from system_setup.logger_setup import init_logger

logger = init_logger()

TITLES = {DANGER: "Небезпека", ATTENTION: "Увага", OFFLINE: "Немає зв'язку"}


class LocationsModule(BaseModule):
    name = "locations"
    source = "system"

    def __init__(self, event_queue, control_queue, config: Dict[str, Any], engine: Optional[LocationStateEngine] = None,
                 notifications=None, clock=time.time):
        super().__init__(event_queue, control_queue, config)
        self.cfg = config.get("locations", {})
        self.engine = engine
        self.notifications = notifications
        self.clock = clock
        self.locations = None
        self.summary_interval = float(self.cfg.get("summary_interval_sec", 1.5))
        self._devices = []
        self._devices_at = 0.0
        self._next_tick = 0.0
        self._next_summary = 0.0
        self._next_full = 0.0

    def devices(self):
        """Device records (cached 2 s; device_* events refresh at once)."""
        if self.locations is not None and self.clock() - self._devices_at > 2:
            self._devices, self._devices_at = self.locations.devices.list_devices(), self.clock()
        return self._devices

    def setup(self) -> None:
        if self.engine is None:
            from core.locations import LocationRegistry

            network = self.config.get("network", {})
            self.locations = LocationRegistry(network.get("devices_path", "data/devices.db"), publish=self.publish,
                                              seed_file=self.cfg.get("seed_file", "config/locations.yaml"))
            ignore = (network.get("hub_location") or "hub", "unknown")
            self.engine = LocationStateEngine(self.locations.name, self.locations.rules, self.devices, self.cfg,
                                              ignore=ignore)
        if self.notifications is None:
            from modules.notify.service import NotificationService

            self.notifications = NotificationService.from_config(self.cfg, publish=self.publish)

    def teardown(self) -> None:
        if self.notifications is not None:
            self.notifications.close()
        if self.locations is not None:
            self.locations.close()

    def handle_control(self, message: Any) -> None:
        if not (isinstance(message, str) and message.startswith("{")):
            super().handle_control(message)
            return
        event = Event.from_json(message)
        now = self.clock()
        if event.type.startswith("device_"):
            self._devices_at = 0.0
        if event.type == "notification_acked" and event.payload.get("location_id"):
            self.transitions(self.engine.acknowledge(event.payload["location_id"], now))
            return
        self.transitions(self.engine.on_event(event, now))

    def transitions(self, changes) -> None:
        for change in changes:
            self.emit("location_state_changed", {k: change[k] for k in ("location_id", "from", "to", "reason",
                                                                         "event_ids")},
                      severity="critical" if change["to"] == DANGER else "warning" if change["to"] == ATTENTION
                      else "info", location_id=change["location_id"])
            logger.warning(f"{change['name']}: {change['from']} → {change['to']} ({change['reason']})")
            if change["to"] in TITLES and self.notifications is not None:
                self.notifications.notify(change["location_id"], change["to"],
                                          f"{TITLES[change['to']]}: {change['name']}", change["reason"],
                                          f"{change['to']}:{change['reason']}", change["event_ids"])

    def step(self) -> None:
        now = self.clock()
        if now >= self._next_tick:
            self._next_tick = now + 0.5
            self.transitions(self.engine.evaluate(now))       # cooldown de-escalation, devices going offline
        if now >= self._next_summary:
            self._next_summary = now + self.summary_interval
            full = now >= self._next_full
            if full:
                self._next_full = now + 10
            names = list(self.engine.live) if full else self.engine.take_dirty()
            if full:
                self.engine.take_dirty()
            for location_id in names:
                self.publish(make_event("system", "location_summary", self.engine.summary(location_id, now),
                                        device_id=self.device_id, location_id=location_id))
        time.sleep(0.05)


def run(event_queue, control_queue, config):
    LocationsModule(event_queue, control_queue, config).run()
