"""
ONVIF WS-Discovery (F11): finds IP cameras on the LAN with one UDP multicast probe (standard library only).

Cameras answer with their service address (``XAddrs``) and scopes (name, hardware). The RTSP stream URL itself needs
the camera's credentials (ONVIF GetStreamUri), so the dashboard shows what was found and the user enters the RTSP URL
(e.g. ``rtsp://user:pass@192.168.0.50:554/stream1``).
"""
import re
import socket
import time
import uuid
from typing import Dict, List
from urllib.parse import unquote, urlparse

MULTICAST = ("239.255.255.250", 3702)
PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header><w:MessageID>uuid:{id}</w:MessageID>
 <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
 <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
 <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>"""


def parse_probe_match(xml: str) -> Dict[str, str]:
    """:return: {"address", "host", "name", "hardware"} from one ProbeMatch answer ({} if not a camera answer)"""
    xaddrs = re.search(r"<[^>]*XAddrs>([^<]+)</", xml)
    if not xaddrs:
        return {}
    address = xaddrs.group(1).split()[0]
    scopes = re.search(r"<[^>]*Scopes>([^<]*)</", xml)
    scope_list = scopes.group(1).split() if scopes else []

    def scope(kind: str) -> str:
        values = [unquote(s.rsplit("/", 1)[-1]) for s in scope_list if f"/{kind}/" in s]
        return values[0] if values else ""
    return {"address": address, "host": urlparse(address).hostname or "", "name": scope("name"),
            "hardware": scope("hardware")}


def discover(timeout_sec: float = 3.0) -> List[Dict[str, str]]:
    """:return: ONVIF devices that answered within ``timeout_sec`` (unique by address)"""
    found: Dict[str, Dict[str, str]] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(0.3)
        sock.sendto(PROBE.format(id=uuid.uuid4()).encode(), MULTICAST)
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            match = parse_probe_match(data.decode("utf-8", "replace"))
            if match:
                found.setdefault(match["address"], match)
    return list(found.values())
