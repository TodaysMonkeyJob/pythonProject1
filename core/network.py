"""
Network reachability for the automatic fallback of the multi-agent mode (M1 / TOD-52): a TCP connect to
``agents.network_probe.host:port`` at most every ``every_sec``. Only used while some location is multiagent — mode
autonomous makes no network calls.
"""
import socket
import time
from typing import Callable, Optional


class NetworkProbe:
    def __init__(self, host: str = "1.1.1.1", port: int = 53, every_sec: float = 30.0, timeout_sec: float = 2.0,
                 clock: Callable[[], float] = time.monotonic, connect: Optional[Callable[[], None]] = None):
        """:param connect: raises OSError when the network is down (tests inject one)"""
        self.host, self.port, self.every, self.timeout, self.clock = host, int(port), float(every_sec), \
            float(timeout_sec), clock
        self.connect = connect or self._connect
        self.state: Optional[bool] = None
        self.checked_at = float("-inf")

    def _connect(self) -> None:
        with socket.create_connection((self.host, self.port), timeout=self.timeout):
            pass

    def ok(self) -> Optional[bool]:
        """:return: True / False (cached for ``every_sec``)"""
        now = self.clock()
        if now - self.checked_at >= self.every:
            self.checked_at = now
            try:
                self.connect()
                self.state = True
            except OSError:
                self.state = False
        return self.state
