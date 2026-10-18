"""
Self-signed HTTPS certificate for the hub (F11): browsers allow camera / microphone only on HTTPS, so phones open
``https://<hub-ip>:8443/node``. Made with the system ``openssl`` (part of macOS and Raspberry Pi OS — nothing to
install); the browser shows a one-time warning for a self-signed certificate. Re-created when the hub's LAN address
changes.
"""
import ipaddress
import os
import socket
import subprocess
from typing import List, Optional, Tuple

from system_setup.logger_setup import init_logger

logger = init_logger()


def lan_ip() -> str:
    """The address other devices on the LAN reach this machine at (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def subject_names() -> List[str]:
    host = socket.gethostname()
    names = {"localhost", host, host.split(".")[0] + ".local", "127.0.0.1", lan_ip()}
    return sorted(names)


def ensure_certificate(folder: str = "data/tls", names: Optional[List[str]] = None,
                       openssl: str = "openssl") -> Tuple[str, str]:
    """
    :return: (certificate path, key path); created (or re-created for new names) with openssl
    """
    names = names or subject_names()
    os.makedirs(folder, exist_ok=True)
    cert, key, stamp = (os.path.join(folder, f) for f in ("hub.crt", "hub.key", "names.txt"))
    wanted = "\n".join(names)
    if os.path.isfile(cert) and os.path.isfile(key) and os.path.isfile(stamp):
        with open(stamp, encoding="utf-8") as f:
            if f.read() == wanted:
                return cert, key
    san = ",".join(f"IP:{n}" if _is_ip(n) else f"DNS:{n}" for n in names)
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "825",
                    "-subj", "/CN=IoT Guard hub", "-addext", f"subjectAltName={san}",
                    "-keyout", key, "-out", cert], check=True, capture_output=True, timeout=60)
    os.chmod(key, 0o600)
    with open(stamp, "w", encoding="utf-8") as f:
        f.write(wanted)
    logger.warning(f"Created a self-signed HTTPS certificate for {', '.join(names)} ({cert})")
    return cert, key


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False
