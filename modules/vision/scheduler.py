"""
Fair frame scheduling across cameras for the pooled F2 / F3 consumers (F11 / TOD-22).

One actions process and one identity process serve every camera. Each camera has its own budget: ``active_fps`` while
it sees motion or people, ``idle_fps`` (1) otherwise. Among the cameras that are due, active ones go first and ties
are served round-robin (least recently served first), so a busy camera never starves the others.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class CameraSlot:
    device_id: str
    location_id: str
    buffer_name: str
    buffer: Any = None              # SharedFrameBuffer, attached lazily
    last_frame_id: int = 0
    active: bool = True             # motion / scene / persons in its last frame
    next_due: float = 0.0
    last_served: float = float("-inf")
    served: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class FairScheduler:
    def __init__(self, active_fps: float = 5.0, idle_fps: float = 1.0):
        self.active_interval = 1.0 / max(0.01, float(active_fps))
        self.idle_interval = 1.0 / max(0.01, float(idle_fps))
        self.slots: Dict[str, CameraSlot] = {}

    def add(self, slot: CameraSlot) -> None:
        self.slots.setdefault(slot.device_id, slot)

    def remove(self, device_id: str) -> Optional[CameraSlot]:
        return self.slots.pop(device_id, None)

    def pick(self, now: float) -> Optional[CameraSlot]:
        """:return: The camera to serve now (active first, then least recently served), or None if none is due"""
        due = [s for s in self.slots.values() if s.next_due <= now]
        if not due:
            return None
        return min(due, key=lambda s: (not s.active, s.last_served))

    def served(self, slot: CameraSlot, now: float, active: Optional[bool] = None) -> None:
        if active is not None:
            slot.active = active
        slot.last_served, slot.served = now, slot.served + 1
        slot.next_due = now + (self.active_interval if slot.active else self.idle_interval)

    def next_due(self) -> Optional[float]:
        return min((s.next_due for s in self.slots.values()), default=None)

    def __iter__(self) -> Iterable[CameraSlot]:
        return iter(list(self.slots.values()))

    def __len__(self) -> int:
        return len(self.slots)


def is_active(meta: Dict[str, Any]) -> bool:
    return bool(meta.get("motion") or meta.get("scene_active") or meta.get("persons"))


def online_cameras(registry, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Enabled, online cameras of this node from the device registry."""
    return [d for d in registry.list_devices(kind="camera", status="online", node_id=node_id) if d["enabled"]]
