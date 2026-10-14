"""
JSON Lines event logger: ``<base_dir>/YYYY-MM-DD/<module>.jsonl``.

A new day directory starts at UTC midnight. When the total size of ``base_dir`` exceeds
``max_total_bytes``, whole day directories are deleted, oldest first; the current day is never deleted.
"""
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Callable, Optional

from core.events import Event
from system_setup.logger_setup import init_logger

logger = init_logger()

DAY_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class JsonlLogger:
    """
    Appends events of one module to a daily JSONL file and enforces a total size cap.
    """

    def __init__(self,
                 module: str,
                 base_dir: str = "logs",
                 max_total_mb: float = 500,
                 check_every: int = 100,
                 clock: Callable[[], datetime] = _utc_now):
        """
        :param module: File name stem, e.g. "gas" → gas.jsonl
        :param base_dir: Root log directory
        :param max_total_mb: Size cap for the whole base_dir, in MB
        :param check_every: Re-check the size cap after this many writes (and on every day change)
        :param clock: Returns the current UTC datetime; injectable for tests
        """
        self.module = module
        self.base_dir = base_dir
        self.max_total_bytes = int(max_total_mb * 1024 * 1024)
        self.check_every = max(1, check_every)
        self._clock = clock
        self._current_day: Optional[str] = None
        self._file = None
        self._writes_since_check = 0

    @property
    def current_path(self) -> Optional[str]:
        """
        :return: Path of the file currently written to, None before the first write
        """
        return self._file.name if self._file else None

    def write(self, event: Event) -> None:
        """
        :param event: Event to append as one JSON line
        """
        day = self._clock().strftime("%Y-%m-%d")
        if day != self._current_day:
            self._open_day(day)
            self.enforce_size_limit()
        self._file.write(event.to_json() + "\n")
        self._file.flush()

        self._writes_since_check += 1
        if self._writes_since_check >= self.check_every:
            self.enforce_size_limit()

    def enforce_size_limit(self) -> None:
        """
        Deletes the oldest day directories until base_dir fits into max_total_bytes.
        """
        self._writes_since_check = 0
        if not os.path.isdir(self.base_dir):
            return
        today = self._current_day or self._clock().strftime("%Y-%m-%d")
        days = sorted(d for d in os.listdir(self.base_dir)
                      if DAY_DIR_PATTERN.match(d) and os.path.isdir(os.path.join(self.base_dir, d)))
        total = _dir_size(self.base_dir)
        for day in days:
            if total <= self.max_total_bytes or day >= today:
                break
            path = os.path.join(self.base_dir, day)
            size = _dir_size(path)
            shutil.rmtree(path, ignore_errors=True)
            total -= size
            logger.info(f"Log size cap reached, removed {path} ({size} bytes)")
        if total > self.max_total_bytes:
            logger.warning(f"Logs in {self.base_dir} exceed the cap ({total} > {self.max_total_bytes} bytes) "
                           f"with only the current day left")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def _open_day(self, day: str) -> None:
        self.close()
        day_dir = os.path.join(self.base_dir, day)
        os.makedirs(day_dir, exist_ok=True)
        self._file = open(os.path.join(day_dir, f"{self.module}.jsonl"), "a", encoding="utf-8")
        self._current_day = day

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
