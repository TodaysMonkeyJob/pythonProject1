"""
Event bus: thin wrapper around ``multiprocessing.Queue`` carrying events as JSON strings.

Keeping the wire format a plain JSON string means the queue can later be replaced by MQTT
without touching the modules.
"""
import multiprocessing as mp
import queue
from typing import Optional

from core.events import Event

CONTROL_STOP = "stop"


def create_queue(maxsize: int = 0):
    """
    :param maxsize: Max number of queued items, 0 for unbounded
    :return: A process-safe queue usable as event or control queue
    """
    return mp.Queue(maxsize=maxsize)


def publish(event_queue, event: Event, block: bool = True, timeout: Optional[float] = None) -> None:
    """
    Puts an event on the bus.

    :param event_queue: Queue from create_queue()
    :param event: Event to send
    :param block: Block if the queue is full
    :param timeout: Max seconds to block
    """
    event_queue.put(event.to_json(), block=block, timeout=timeout)


def receive(event_queue, timeout: Optional[float] = None) -> Optional[Event]:
    """
    Takes the next event from the bus.

    :param event_queue: Queue from create_queue()
    :param timeout: Seconds to wait; None blocks forever
    :return: Event, or None when the timeout expires
    """
    try:
        data = event_queue.get(timeout=timeout)
    except queue.Empty:
        return None
    return Event.from_json(data)
