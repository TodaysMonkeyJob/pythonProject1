"""
Locations (F12 / TOD-23): rooms / territories with their order on the dashboard, an icon and per-location rules.

Stored in the ``locations`` table of the device registry database (``data/devices.db``): devices reference their
location there, so deleting a location and moving its devices to ``unassigned`` is one transaction. The first start
seeds it from ``config/locations.yaml`` (optional)::

    locations:
      - id: warehouse
        name: "Warehouse"
        order: 1
        icon: "📦"
        rules:
          unknown_person_after: "20:00"   # an unknown person only counts as a warning after this time (until 06:00)
          gas_thresholds: {acetone: 3.0}  # a reading above → critical in this location

Changes publish ``device_updated`` for moved devices (so modules follow) and are picked up by readers via ``poll()``.
"""
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from core.devices import slug
from core.events import UNASSIGNED, Event

COLUMNS = {"order_index": "INTEGER NOT NULL DEFAULT 50", "icon": "TEXT", "rules_json": "TEXT"}
RULE_KEYS = ("unknown_person_after", "gas_thresholds", "attention_window_min", "cooldown_min")


class LocationRegistry:
    def __init__(self, path: str = "data/devices.db", publish: Optional[Callable[[Event], None]] = None,
                 seed_file: Optional[str] = None, clock: Callable[[], float] = time.time):
        """
        :param path: The device registry database (shares it with ``DeviceRegistry``)
        :param publish: For ``device_updated`` of devices moved by a delete
        :param seed_file: YAML with ``locations:`` used when only ``unassigned`` exists yet
        """
        from core.devices import DeviceRegistry

        self.devices = DeviceRegistry(path, publish=publish, clock=clock)  # creates the tables
        self.db = self.devices.db
        self._lock = self.devices._lock
        self.clock = clock
        with self._lock:
            existing = {row[1] for row in self.db.execute("PRAGMA table_info(locations)")}
            for column, kind in COLUMNS.items():
                if column not in existing:  # databases created before F12
                    self.db.execute(f"ALTER TABLE locations ADD COLUMN {column} {kind}")
            self.db.execute("UPDATE locations SET order_index = 99 WHERE location_id = ? AND order_index = 50",
                            (UNASSIGNED,))
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._version: Optional[int] = None
        if seed_file and os.path.isfile(seed_file) and len(self.list()) <= 1:
            self.seed(seed_file)

    def close(self) -> None:
        self.devices.close()

    # --- reads ------------------------------------------------------------------------------------------------------

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.db.execute("SELECT location_id, name, order_index, icon, rules_json FROM locations "
                                   "ORDER BY order_index, name").fetchall()
        return [{"location_id": r[0], "name": r[1], "order": r[2], "icon": r[3] or "",
                 "rules": json.loads(r[4] or "{}")} for r in rows]

    def get(self, location_id: str) -> Optional[Dict[str, Any]]:
        self.poll()
        return self._cache.get(location_id)

    def name(self, location_id: str) -> str:
        location = self.get(location_id)
        return location["name"] if location else location_id

    def rules(self, location_id: str) -> Dict[str, Any]:
        location = self.get(location_id)
        return location["rules"] if location else {}

    def poll(self) -> bool:
        """Refreshes the cache when another connection changed the database. :return: refreshed"""
        with self._lock:
            version = self.db.execute("PRAGMA data_version").fetchone()[0]
        if version == self._version and self._cache:
            return False
        self._cache = {loc["location_id"]: loc for loc in self.list()}
        self._version = version
        return True

    # --- writes -----------------------------------------------------------------------------------------------------

    def create(self, name: str, order: Optional[int] = None, icon: str = "", rules: Optional[Dict] = None,
               location_id: Optional[str] = None) -> Dict[str, Any]:
        if not name.strip():
            raise ValueError("Empty location name")
        location_id = location_id or slug(name)
        with self._lock:
            if self.db.execute("SELECT 1 FROM locations WHERE location_id = ?", (location_id,)).fetchone():
                raise ValueError(f"Location {location_id!r} already exists")
            if order is None:
                order = (self.db.execute("SELECT MAX(order_index) FROM locations WHERE location_id != ?",
                                         (UNASSIGNED,)).fetchone()[0] or 0) + 1
            self.db.execute("INSERT INTO locations (location_id, name, created_at, order_index, icon, rules_json) "
                            "VALUES (?, ?, ?, ?, ?, ?)", (location_id, name.strip()[:64], self.clock(), int(order),
                                                          icon[:8], json.dumps(clean_rules(rules))))
        self._version = None
        return self.get(location_id)

    def update(self, location_id: str, name: Optional[str] = None, order: Optional[int] = None,
               icon: Optional[str] = None, rules: Optional[Dict] = None) -> Dict[str, Any]:
        if self.get(location_id) is None:
            raise KeyError(f"Unknown location {location_id!r}")
        sets, args = [], []
        for column, value in (("name", name and name.strip()[:64]), ("order_index", order), ("icon", icon),
                              ("rules_json", None if rules is None else json.dumps(clean_rules(rules)))):
            if value is not None:
                sets.append(f"{column} = ?")
                args.append(value)
        if sets:
            with self._lock:
                self.db.execute(f"UPDATE locations SET {', '.join(sets)} WHERE location_id = ?", (*args, location_id))
        self._version = None
        return self.get(location_id)

    def reorder(self, location_ids: List[str]) -> List[Dict[str, Any]]:
        """Dashboard drag & drop: the given order (1, 2, …); ``unassigned`` stays last."""
        with self._lock:
            for index, location_id in enumerate(i for i in location_ids if i != UNASSIGNED):
                self.db.execute("UPDATE locations SET order_index = ? WHERE location_id = ?", (index + 1, location_id))
        self._version = None
        return self.list()

    def delete(self, location_id: str) -> List[str]:
        """Deletes a location; its devices move to ``unassigned``. :return: moved device ids"""
        if location_id == UNASSIGNED:
            raise ValueError("'unassigned' cannot be deleted")
        if self.get(location_id) is None:
            raise KeyError(f"Unknown location {location_id!r}")
        moved = [d["device_id"] for d in self.devices.list_devices() if d["location_id"] == location_id]
        for device_id in moved:
            self.devices.update(device_id, location_id=UNASSIGNED)   # publishes device_updated → modules follow
        with self._lock:
            self.db.execute("DELETE FROM locations WHERE location_id = ?", (location_id,))
        self._version = None
        return moved

    def seed(self, path: str) -> int:
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        count = 0
        for item in data.get("locations", []):
            location_id = item.get("id") or slug(item.get("name", ""))
            if not location_id:
                continue
            if self.get(location_id) is None:
                self.create(item.get("name", location_id), item.get("order"), item.get("icon", ""),
                            item.get("rules"), location_id=location_id)
            else:
                self.update(location_id, item.get("name"), item.get("order"), item.get("icon"), item.get("rules"))
            count += 1
        return count


def clean_rules(rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Only known rule keys; gas thresholds as floats."""
    rules = {k: v for k, v in (rules or {}).items() if k in RULE_KEYS and v not in (None, "")}
    if "gas_thresholds" in rules:
        rules["gas_thresholds"] = {str(k): float(v) for k, v in (rules["gas_thresholds"] or {}).items()}
    return rules

