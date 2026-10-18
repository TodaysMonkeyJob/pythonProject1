"""
Automatic discovery of local cameras and microphones (F11 / TOD-22), without extra programs.

* macOS cameras: ``system_profiler SPCameraDataType -json`` — names + unique ids in AVFoundation order, which is also
  the OpenCV index order (``network.camera_index_overrides`` fixes a mismatch). An iPhone used as a Continuity Camera
  shows up here with an ``iPhone…`` model id → transport ``continuity``.
* Linux / Raspberry Pi cameras: ``/sys/class/video4linux/video*`` (``index`` 0 = capture node; metadata nodes skipped)
  and CSI camera modules via ``picamera2.Picamera2.global_camera_info()`` when picamera2 is installed.
* Microphones: ``sounddevice.query_devices()`` (input channels > 0), PortAudio re-initialised so hot-plugged devices
  appear; ``network.ignore_mics`` patterns skip virtual devices.

``diff()`` compares two scans by hardware key → added / removed / changed (e.g. an OpenCV index shifted after another
camera was unplugged); the orchestrator starts / stops / restarts pipelines from it.
"""
import glob
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from system_setup.logger_setup import init_logger

logger = init_logger()

DEFAULT_IGNORE_MICS = ("Steam Streaming", "BlackHole", "ZoomAudioDevice", "Microsoft Teams Audio", "Soundflower",
                       "Loopback", "Aggregate", "Multi-Output")


@dataclass(frozen=True)
class FoundDevice:
    kind: str                     # camera | mic
    transport: str                # local | continuity | picamera
    name: str
    hw_key: str                   # stable across scans (unique id / sysfs path / name)
    params: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)   # how to open it

    def same_params(self, other: "FoundDevice") -> bool:
        return self.params == other.params


@dataclass
class Diff:
    added: List[FoundDevice] = field(default_factory=list)
    removed: List[FoundDevice] = field(default_factory=list)
    changed: List[FoundDevice] = field(default_factory=list)     # same device, new params (restart its pipeline)

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.changed)


def diff(previous: Iterable[FoundDevice], current: Iterable[FoundDevice]) -> Diff:
    before = {d.hw_key: d for d in previous}
    after = {d.hw_key: d for d in current}
    return Diff(added=[after[k] for k in after if k not in before],
                removed=[before[k] for k in before if k not in after],
                changed=[after[k] for k in after if k in before and not before[k].same_params(after[k])])


# --- cameras ----------------------------------------------------------------------------------------------------------

def parse_system_profiler(output: str) -> List[FoundDevice]:
    """:param output: ``system_profiler SPCameraDataType -json`` (warnings before the JSON are ignored)"""
    start = output.find("{")
    if start < 0:
        return []
    cameras = json.loads(output[start:]).get("SPCameraDataType", [])
    found = []
    for index, camera in enumerate(cameras):
        name = camera.get("_name") or f"Camera {index}"
        model = camera.get("spcamera_model-id", "")
        transport = "continuity" if model.startswith(("iPhone", "iPad")) else "local"
        hw_key = f"avfoundation:{camera.get('spcamera_unique-id') or name}"
        found.append(FoundDevice("camera", transport, name, hw_key, {"source": index, "model": model}))
    return found


def macos_cameras(run: Callable[..., str] = None) -> List[FoundDevice]:
    run = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout)
    try:
        return parse_system_profiler(run(["system_profiler", "SPCameraDataType", "-json"]))
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        logger.warning(f"Camera discovery failed: {e}")
        return []


def linux_cameras(sysfs: str = "/sys/class/video4linux", picamera_info: Optional[Callable[[], list]] = None) \
        -> List[FoundDevice]:
    found = []
    for path in sorted(glob.glob(os.path.join(sysfs, "video*")), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        node = os.path.basename(path)

        def read(name, default="", path=path):
            try:
                with open(os.path.join(path, name)) as f:
                    return f.read().strip()
            except OSError:
                return default
        if read("index", "0") != "0":
            continue  # second node of the same camera = metadata, not frames
        name = read("name") or node
        if re.search(r"bcm2835|codec|isp|hevc|unicam|rp1-cfe", name, re.I):
            continue  # Pi hardware codec / ISP nodes, not cameras (CSI cameras come from picamera2)
        device_link = os.path.realpath(os.path.join(path, "device"))
        hw_key = f"v4l2:{device_link if device_link != os.path.join(path, 'device') else name}"
        found.append(FoundDevice("camera", "local", name, hw_key, {"source": f"/dev/{node}"}))
    if picamera_info is None:
        try:
            from picamera2 import Picamera2  # Raspberry Pi OS only

            picamera_info = Picamera2.global_camera_info
        except ImportError:
            picamera_info = list
    for index, info in enumerate(picamera_info()):
        name = info.get("Model") or f"Pi camera {index}"
        found.append(FoundDevice("camera", "picamera", f"Pi camera {name}", f"picamera:{info.get('Id', index)}",
                                 {"source": f"picamera2:{index}"}))
    return found


def cameras() -> List[FoundDevice]:
    system = platform.system()
    if system == "Darwin":
        return macos_cameras()
    if system == "Linux":
        return linux_cameras()
    return []


# --- microphones ------------------------------------------------------------------------------------------------------

def parse_input_devices(devices: Sequence[Dict[str, Any]], ignore: Sequence[str] = DEFAULT_IGNORE_MICS) \
        -> List[FoundDevice]:
    """:param devices: ``sounddevice.query_devices()`` entries"""
    found, seen = [], set()
    for device in devices:
        name = str(device.get("name", ""))
        if int(device.get("max_input_channels", 0)) <= 0 or any(p.lower() in name.lower() for p in ignore):
            continue
        if name in seen:
            continue  # the same microphone listed by several host APIs
        seen.add(name)
        transport = "continuity" if re.search(r"iphone|ipad", name, re.I) else "local"
        found.append(FoundDevice("mic", transport, name, f"mic:{name}", {"device": name}))
    return found


def microphones(ignore: Sequence[str] = DEFAULT_IGNORE_MICS) -> List[FoundDevice]:
    try:
        import sounddevice as sd

        sd._terminate()  # PortAudio only sees hot-plugged devices after a re-init
        sd._initialize()
        return parse_input_devices(sd.query_devices(), ignore)
    except Exception as e:
        logger.warning(f"Microphone discovery failed: {e}")
        return []


def scan(cfg: Dict[str, Any]) -> List[FoundDevice]:
    """
    :param cfg: The ``network`` config section (``discover_cameras``, ``discover_mics``, ``ignore_mics``,
                ``camera_index_overrides`` {name: index})
    :return: Cameras + microphones found now
    """
    found: List[FoundDevice] = []
    continuity_names: List[str] = []
    if cfg.get("discover_cameras", True):
        overrides = cfg.get("camera_index_overrides") or {}
        for camera in cameras():
            if camera.name in overrides:
                camera = FoundDevice(camera.kind, camera.transport, camera.name, camera.hw_key,
                                     {**camera.params, "source": overrides[camera.name]})
            found.append(camera)
            if camera.transport == "continuity":
                continuity_names.append(re.sub(r"\s*camera$", "", camera.name, flags=re.I))
    if cfg.get("discover_mics", True):
        for mic in microphones(tuple(cfg.get("ignore_mics") or DEFAULT_IGNORE_MICS)):
            if any(base and mic.name.startswith(base) for base in continuity_names):
                mic = FoundDevice(mic.kind, "continuity", mic.name, mic.hw_key, mic.params)  # "Minnor Microphone"
            found.append(mic)
    return found
