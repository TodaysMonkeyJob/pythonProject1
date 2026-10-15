"""
Shared-memory frame buffer: the vision process writes, the dashboard (F7) and action/identity modules
(F2, F3) read by name.

One ``SharedMemory`` block holds a header, the latest annotated JPEG, the latest raw BGR frame and a JSON
metadata blob (person boxes etc.). There is a single writer, so consistency uses a sequence counter
(seqlock): the writer makes it odd while writing and even when done; readers retry if it changed.
"""
import json
import struct
import time
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Dict, Optional, Tuple

import numpy as np

# seq, frame_id, timestamp, jpeg_len, meta_len, height, width, channels
HEADER = struct.Struct("<QQdIIIII")
HEADER_SIZE = 64
DEFAULT_MAX_JPEG = 512 * 1024
DEFAULT_MAX_META = 64 * 1024


class SharedFrameBuffer:
    def __init__(self, shm: shared_memory.SharedMemory, width: int, height: int,
                 max_jpeg: int, max_meta: int, owner: bool):
        self.shm = shm
        self.width, self.height = width, height
        self.max_jpeg, self.max_meta = max_jpeg, max_meta
        self.owner = owner
        self.raw_size = width * height * 3
        self.jpeg_offset = HEADER_SIZE
        self.meta_offset = self.jpeg_offset + max_jpeg
        self.raw_offset = self.meta_offset + max_meta
        self._frame_id = 0

    @staticmethod
    def size_for(width: int, height: int, max_jpeg: int = DEFAULT_MAX_JPEG, max_meta: int = DEFAULT_MAX_META) -> int:
        return HEADER_SIZE + max_jpeg + max_meta + width * height * 3

    @classmethod
    def create(cls, name: str, width: int, height: int, max_jpeg: int = DEFAULT_MAX_JPEG,
               max_meta: int = DEFAULT_MAX_META) -> "SharedFrameBuffer":
        """
        Creates the block, or reuses an existing one of the right size (e.g. after a vision restart).
        The block outlives the creating process so readers keep a valid mapping across vision restarts;
        the orchestrator calls unlink() on shutdown.
        """
        size = cls.size_for(width, height, max_jpeg, max_meta)
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            _untrack(shm)
            shm.buf[:HEADER_SIZE] = bytes(HEADER_SIZE)
        except FileExistsError:
            shm = _attach(name)
            if shm.size < size:
                shm.close()
                raise ValueError(f"Shared memory {name!r} exists with a smaller size ({shm.size} < {size})") from None
        return cls(shm, width, height, max_jpeg, max_meta, owner=True)

    @classmethod
    def attach(cls, name: str, width: int, height: int, max_jpeg: int = DEFAULT_MAX_JPEG,
               max_meta: int = DEFAULT_MAX_META) -> "SharedFrameBuffer":
        """Attaches a reader to an existing block."""
        return cls(_attach(name), width, height, max_jpeg, max_meta, owner=False)

    def write(self, jpeg: bytes, raw: np.ndarray, meta: Dict[str, Any], timestamp: Optional[float] = None) -> int:
        """
        :param jpeg: Annotated frame, JPEG-encoded
        :param raw: Raw BGR frame (height, width, 3)
        :param meta: JSON-serialisable metadata (e.g. {"persons": [...]})
        :return: frame_id written
        """
        meta_bytes = json.dumps(meta).encode("utf-8")
        if len(jpeg) > self.max_jpeg:
            raise ValueError(f"JPEG too large ({len(jpeg)} > {self.max_jpeg} bytes)")
        if len(meta_bytes) > self.max_meta:
            raise ValueError(f"Metadata too large ({len(meta_bytes)} > {self.max_meta} bytes)")
        if raw.shape != (self.height, self.width, 3) or raw.dtype != np.uint8:
            raise ValueError(f"Raw frame must be uint8 {(self.height, self.width, 3)}, got {raw.dtype} {raw.shape}")

        buf = self.shm.buf
        seq = HEADER.unpack_from(buf, 0)[0]
        self._frame_id += 1
        _write_seq(buf, seq + 1)  # odd: write in progress
        buf[self.jpeg_offset:self.jpeg_offset + len(jpeg)] = jpeg
        buf[self.meta_offset:self.meta_offset + len(meta_bytes)] = meta_bytes
        buf[self.raw_offset:self.raw_offset + self.raw_size] = raw.tobytes()
        HEADER.pack_into(buf, 0, seq + 1, self._frame_id, timestamp or time.time(), len(jpeg), len(meta_bytes),
                         self.height, self.width, 3)
        _write_seq(buf, seq + 2)  # even: consistent
        return self._frame_id

    def read_annotated(self) -> Optional[Tuple[int, float, bytes]]:
        """
        :return: (frame_id, timestamp, jpeg) or None if nothing was written yet
        """
        result = self._read(raw=False)
        return None if result is None else result[:3]

    def read_raw(self) -> Optional[Tuple[int, float, np.ndarray, Dict[str, Any]]]:
        """
        :return: (frame_id, timestamp, raw frame copy, metadata) or None if nothing was written yet
        """
        result = self._read(raw=True)
        return None if result is None else (result[0], result[1], result[3], result[4])

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        """Removes the block; only the creator (orchestrator) should call this."""
        # unlink() unregisters from the resource tracker; register first because _untrack() removed it,
        # otherwise the tracker process prints a KeyError
        try:
            resource_tracker.register(self.shm._name, "shared_memory")
        except Exception:
            pass
        self.shm.unlink()

    def _read(self, raw: bool, retries: int = 50):
        buf = self.shm.buf
        for _ in range(retries):
            seq, frame_id, timestamp, jpeg_len, meta_len, _, _, _ = HEADER.unpack_from(buf, 0)
            if frame_id == 0:
                return None
            if seq % 2:
                time.sleep(0.001)
                continue
            jpeg = bytes(buf[self.jpeg_offset:self.jpeg_offset + jpeg_len])
            meta = bytes(buf[self.meta_offset:self.meta_offset + meta_len])
            frame = None
            if raw:
                frame = np.frombuffer(buf[self.raw_offset:self.raw_offset + self.raw_size], dtype=np.uint8)
                frame = frame.reshape(self.height, self.width, 3).copy()
            if HEADER.unpack_from(buf, 0)[0] == seq:
                return frame_id, timestamp, jpeg, frame, json.loads(meta)
        return None


def _write_seq(buf, seq: int) -> None:
    struct.pack_into("<Q", buf, 0, seq)


def _untrack(shm: shared_memory.SharedMemory) -> None:
    # Python < 3.13 registers every block with the resource tracker, which unlinks it when the process
    # exits and breaks the other processes using it
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


def _attach(name: str) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(name=name)
    _untrack(shm)
    return shm
