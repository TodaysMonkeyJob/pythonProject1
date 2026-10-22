"""F23 (TOD-34): trust registry — priors, update math, forgetting, health checks, explanations, failure / recovery."""
from pathlib import Path

import numpy as np
import pytest

from core.events import make_event
from core.fuzzy import load_yaml
from modules.audio.attribution import iso
from modules.trust import TrustScorer
from modules.trust.evidence import edit_ratio
from modules.trust.health import CameraHealth, GasHealth, HealthPublisher, MicHealth, frame_checks
from modules.trust.registry import TrustRegistry

REPO = Path(__file__).resolve().parents[1]
CFG = load_yaml(str(REPO / "config" / "trust_registry.yaml"))
T0 = 1_760_000_000.0
DAY = 86400.0


def registry():
    return TrustRegistry(CFG)


def ev(source, type, payload, t=T0, location="lab", device="cam-1", severity="info"):
    return make_event(source, type, payload, severity=severity, device_id=device,
                      location_id=location).model_copy(update={"timestamp": iso(t)})


@pytest.fixture
def scorer():
    s = TrustScorer.from_config(str(REPO / "config" / "fuzzy.yaml"))
    for device, kind in (("cam-1", "camera"), ("mic-1", "mic"), ("gas-1", "gas")):
        s.apply(ev("system", "device_online", {"device_id": device, "kind": kind, "location_id": "lab"}, T0 - 60,
                   device=device))
    return s


# --- channels and priors ----------------------------------------------------------------------------------------

def test_every_channel_of_every_device_kind_exists_with_a_measured_or_unmeasured_prior():
    reg = registry()
    for device, kind in (("cam-1", "camera"), ("mic-1", "mic"), ("gas-1", "gas"), ("identity", "identity")):
        reg.ensure_device(device, kind, "lab", T0)
    expected = {f"{d}/{c}" for d, kind in (("cam-1", "camera"), ("mic-1", "mic"), ("gas-1", "gas"),
                                           ("identity", "identity")) for c in CFG["device_channels"][kind]}
    assert set(reg.channels) == expected and len(expected) == 14
    for ch in reg.snapshot(T0):
        assert ch["prior_status"] in ("measured", "unmeasured")
        if ch["prior_status"] == "unmeasured":                 # nothing known → τ 0.5, u 1, said so
            assert ch["r"] == ch["s"] == 0 and ch["tau"] == 0.5 and ch["u"] == 1.0
            assert "не виміряно" in ch["explanation"]
        else:
            assert ch["prior"]["source"] and ch["r"] + ch["s"] > 0
    gas = reg.get("gas-1/gas_class", T0)
    assert gas["r"] == 29 and gas["s"] == 0 and gas["tau"] == pytest.approx(30 / 31, abs=1e-3)


def test_update_math_and_per_kind_forgetting():
    reg = registry()
    reg.ensure_device("cam-1", "camera", "lab", T0)
    reg.ensure_device("mic-1", "mic", "lab", T0)
    cid = "cam-1/objects"                                          # r0 = s0 = 0, γ 0.95
    reg.confirm(cid, 3, "test", T0)
    reg.refute(cid, 1, "test", T0)
    ch = reg.get(cid, T0)
    assert (ch["r"], ch["s"]) == (3, 1) and ch["tau"] == pytest.approx(4 / 6, abs=1e-4)
    assert ch["u"] == pytest.approx(2 / 6, abs=1e-4)
    later = reg.get(cid, T0 + 2 * DAY)
    assert later["r"] == pytest.approx(3 * 0.95 ** 2) and later["s"] == pytest.approx(0.95 ** 2)
    asr, gas_like = reg.channels["mic-1/asr"], reg.channels["cam-1/face_identity"]
    assert asr.gamma < gas_like.gamma                              # ASR forgets faster than identity
    r_asr = reg.get("mic-1/asr", T0 + 10 * DAY)["r"] / reg.get("mic-1/asr", T0)["r"]
    r_face = reg.get("cam-1/face_identity", T0 + 10 * DAY)["r"] / reg.get("cam-1/face_identity", T0)["r"]
    assert r_asr == pytest.approx(0.8 ** 10, rel=1e-3) and r_face == pytest.approx(0.97 ** 10, rel=1e-3)
    assert r_asr < r_face


def test_history_reproduces_every_past_tau(tmp_path):
    reg = TrustRegistry(CFG, str(tmp_path / "trust.db"))
    reg.ensure_device("cam-1", "camera", "lab", T0)
    reg.confirm("cam-1/objects", 2, "a", T0 + 10)
    reg.set_health("cam-1", {"lens_covered": {"detail": "σ 1.0"}}, T0 + 20)
    reg.refute("cam-1/objects", 1, "b", T0 + DAY)
    rows = reg.history("cam-1/objects")
    assert [r["cause_type"] for r in rows] == ["created", "evidence", "health", "evidence"]
    for row in rows:
        assert row["tau"] == pytest.approx((row["r"] + 1) / (row["r"] + row["s"] + 2) * row["h"])
    reopened = TrustRegistry(CFG, str(tmp_path / "trust.db"))           # state survives a restart
    assert reopened.get("cam-1/objects", T0 + DAY)["tau"] == pytest.approx(reg.get("cam-1/objects", T0 + DAY)["tau"])
    reg.close()
    reopened.close()


# --- health checks on synthetic inputs ----------------------------------------------------------------------------

def test_camera_checks_on_synthetic_frames():
    rng = np.random.default_rng(0)
    scene = rng.integers(40, 220, (240, 320, 3)).astype(np.uint8)
    cfg = CFG["camera"]
    assert frame_checks(scene[..., 0].astype(np.float32), cfg) == {}
    assert "lens_covered" in frame_checks(np.full((120, 160), 128.0, np.float32), cfg)
    dark = rng.integers(0, 30, (120, 160)).astype(np.float32)
    assert "too_dark" in frame_checks(dark, cfg)
    bright = rng.integers(230, 256, (120, 160)).astype(np.float32)
    assert "overexposed" in frame_checks(bright, cfg)
    blurred = np.tile(np.linspace(60, 200, 160, dtype=np.float32), (120, 1))
    assert "blurred" in frame_checks(blurred, cfg)
    health = CameraHealth({**cfg, "target_fps": 0})
    results = [health.observe(scene, 5.0 * i) for i in range(10)]   # the same frame for 45 s → frozen
    assert "frozen_frame" not in results[1] and "frozen_frame" in results[-1]
    halves = np.zeros((240, 320, 3), np.uint8)
    halves[:, :160] = 220
    halves += rng.integers(0, 20, halves.shape).astype(np.uint8)
    moved = CameraHealth({**cfg, "target_fps": 0})
    moved.observe(halves, 0)
    assert "camera_moved" in moved.observe(halves[:, ::-1].copy(), 5, motion=False)
    still = CameraHealth({**cfg, "target_fps": 0})
    still.observe(halves, 0)
    assert "camera_moved" not in still.observe(halves[:, ::-1].copy(), 5, motion=True)   # real motion: not moved
    offline = CameraHealth(cfg)
    assert offline.no_frame(0) is None and "no_frames" in offline.no_frame(10)


def test_mic_checks_on_synthetic_audio():
    cfg = CFG["mic"]
    rng = np.random.default_rng(1)
    clipped = MicHealth(cfg)
    loud = np.clip(rng.normal(0, 20000, 480), -32768, 32767).astype(np.int16)
    failing = None
    for i in range(400):
        failing = clipped.observe(loud, i * 0.03) or failing
    assert "clipping" in failing
    silent = MicHealth(cfg)
    for i in range(400):
        failing = silent.observe(np.zeros(480, np.int16), i * 0.03) or failing
    assert "digital_silence" in failing
    assert "wrong_sample_rate" in (MicHealth({**cfg, "interval_sec": 0}, 16000, 44100).observe(
        rng.normal(0, 300, 480).astype(np.int16), 1.0) or {})


def test_gas_checks_on_synthetic_readings():
    health = GasHealth.from_dataset(CFG["gas"])
    curve = list(health.clean * 1.0)
    failing = {}
    for _ in range(31):
        failing = health.observe(curve, "clean_air")
    assert "flat_readings" in failing
    assert "out_of_range" in health.observe(list(health.hi * 3), "ethanol")
    assert health.baseline(curve)["ok"] is True
    assert health.baseline(list(health.clean * 1.6))["ok"] is False


# --- explanations -------------------------------------------------------------------------------------------------

def test_explanation_templates_name_the_cause():
    reg = registry()
    for device, kind in (("cam-1", "camera"), ("mic-1", "mic"), ("gas-1", "gas")):
        reg.ensure_device(device, kind, "lab", T0)
    reg.set_health("cam-1", {"lens_covered": {"detail": "σ 1.2"}}, T0)
    text = reg.get("cam-1/objects", T0 + 240)["explanation"]
    assert text.startswith("Довіра 0,15: камера cam-1") and "закрито об'єктив" in text and "h = 0,30" in text
    for _ in range(6):
        reg.refute("mic-1/asr", 1, "виправлено", T0, source="operator")
    reg.set_health("mic-1", {"clipping": {"detail": "18 % відліків"}}, T0)
    text = reg.get("mic-1/asr", T0)["explanation"]
    assert "мікрофон mic-1" in text and "кліпінг" in text and "6 транскриптів виправлено" in text
    reg.refute("gas-1/gas_class", 1, "самоперевірка не пройдена", T0, source="selftest")
    reg.set_health("gas-1", {"baseline_drift": {"detail": "R_peak 40 %"}}, T0)
    assert "дрейф базової лінії" in reg.get("gas-1/gas_class", T0)["explanation"]
    assert "не виміряно" in reg.get("cam-1/motion", T0)["explanation"] or "закрито" in \
        reg.get("cam-1/motion", T0)["explanation"]


# --- failure and recovery per device type ---------------------------------------------------------------------

def test_camera_lens_covered_drops_every_channel_then_recovers_gradually(scorer):
    covered = {"lens_covered": {"value": 1.0, "detail": "σ 1.0"}}
    scorer.apply(ev("camera", "device_health", {"kind": "camera", "failing": covered}, T0))
    channels = [c for c in scorer.registry.snapshot(T0) if c["device_id"] == "cam-1"]
    assert len(channels) == 6 and all(c["tau"] < 0.4 for c in channels)
    assert all("закрито об'єктив" in c["explanation"] for c in channels)
    p = scorer.apply(ev("camera", "person_identified", {"track_id": 1, "person_id": "p", "known": False}, T0 + 1))
    assert p is True
    scorer.apply(ev("camera", "device_health", {"kind": "camera", "failing": {}}, T0 + 10))
    face = [scorer.registry.tau("cam-1/face_identity", T0 + 10 + m * 60) for m in (0, 2, 4, 8)]
    assert face == sorted(face) and face[0] < 0.4 and face[1] < face[-1]      # gradual, not instant
    assert face[-1] == pytest.approx(scorer.registry.tau("cam-1/face_identity", T0 + DAY / 100), abs=0.02)


def test_microphone_clipping_or_unplugging_drops_its_channels(scorer):
    before = {k: scorer.registry.tau(f"mic-1/{k}", T0) for k in ("vad", "asr", "speaker_identity")}
    scorer.apply(ev("audio", "device_health", {"kind": "mic", "failing": {"clipping": {"detail": "18 % відліків"}}},
                    T0, device="mic-1"))
    for k, tau in before.items():
        assert scorer.registry.tau(f"mic-1/{k}", T0) < tau * 0.6
        assert "кліпінг" in scorer.registry.get(f"mic-1/{k}", T0)["explanation"]
    assert scorer.registry.tau("cam-1/objects", T0) == 0.5                      # other devices untouched
    scorer.apply(ev("system", "device_offline", {"device_id": "mic-1", "kind": "mic", "location_id": "lab"}, T0 + 5,
                    device="mic-1"))
    assert scorer.registry.tau("mic-1/vad", T0 + 5) == 0.0
    assert "офлайн" in scorer.registry.get("mic-1/vad", T0 + 5)["explanation"]


def test_stuck_silence_only_counts_while_people_are_seen(scorer):
    silence = {"kind": "mic", "failing": {"digital_silence": {"value": -120, "factor": 1.0}}}
    scorer.apply(ev("camera", "object_detected", {"track_id": 1, "class": "person", "state": "appeared"}, T0))
    scorer.apply(ev("audio", "device_health", silence, T0, device="mic-1"))
    assert scorer.registry.get("mic-1/vad", T0)["h"] == 1.0
    scorer.apply(ev("camera", "person_identified", {"track_id": 1, "person_id": "p", "known": True,
                                                    "similarity": 0.7}, T0 + 590))
    scorer.apply(ev("audio", "device_health", silence, T0 + 601, device="mic-1"))
    assert scorer.registry.get("mic-1/vad", T0 + 601)["h"] == pytest.approx(0.3)


def test_gas_baseline_drift_drops_gas_channels_below_0_3_then_recovers(scorer):
    drift = {"kind": "gas", "failing": {"baseline_drift": {"value": 0.4, "detail": "R_peak 40 % від еталону"}},
             "baseline_ok": False, "baseline_detail": "R_peak 40 % від еталону"}
    scorer.apply(ev("gas", "device_health", drift, T0, device="gas-1"))
    for k in ("gas_class", "gas_concentration", "gas_trend"):
        assert scorer.registry.tau(f"gas-1/{k}", T0) < 0.3
    assert "1 самоперевірку не пройдено" in scorer.registry.get("gas-1/gas_class", T0)["explanation"]
    ok = {"kind": "gas", "failing": {}, "baseline_ok": True, "baseline_detail": "R_peak 2 %"}
    scorer.apply(ev("gas", "device_health", ok, T0 + 3600, device="gas-1"))   # recalibrated
    steps = [scorer.registry.tau("gas-1/gas_class", T0 + 3600 + m * 60) for m in (0, 3, 10)]
    assert steps[0] < 0.3 < steps[-1] and steps[0] < steps[1] < steps[2]


# --- evidence and operator feedback -------------------------------------------------------------------------------

def test_operator_feedback_changes_only_that_channel(scorer):
    before = {cid: scorer.registry.tau(cid, T0) for cid in scorer.registry.channels}
    scorer.apply(ev("system", "trust_feedback", {"channel_id": "cam-1/pose_action", "verdict": "refute",
                                                    "reason": "не лежить"}, T0, device="dashboard"))
    after = {cid: scorer.registry.tau(cid, T0) for cid in scorer.registry.channels}
    changed = {cid for cid in before if abs(before[cid] - after[cid]) > 1e-9}
    assert changed == {"cam-1/pose_action"} and after["cam-1/pose_action"] < before["cam-1/pose_action"]
    scorer.apply(ev("system", "trust_feedback", {"channel_id": "mic-1/asr", "original": "відкрий двері складу",
                                                    "corrected": "закрий двері на складі"}, T0, device="dashboard"))
    assert scorer.registry.get("mic-1/asr", T0)["stats"]["operator_refute"] == 1
    scorer.apply(ev("system", "trust_feedback", {"channel_id": "mic-1/asr", "original": "привіт олег",
                                                    "corrected": "привіт, Олег"}, T0, device="dashboard"))
    assert scorer.registry.get("mic-1/asr", T0)["stats"]["operator_confirm"] == 1      # a small edit confirms
    assert edit_ratio("abc", "abc") == 0 and edit_ratio("abc", "xyz") == 1


def test_stream_evidence_confirms_and_refutes_the_right_channels(scorer):
    obj = "cam-1/objects"
    scorer.apply(ev("camera", "object_detected", {"track_id": 1, "class": "person", "state": "appeared"}, T0))
    scorer.apply(ev("camera", "object_detected", {"track_id": 1, "class": "person", "state": "disappeared"}, T0 + 0.5))
    assert scorer.registry.get(obj, T0 + 1)["s"] == 1                              # flicker
    scorer.apply(ev("camera", "object_detected", {"track_id": 2, "class": "person", "state": "appeared"}, T0 + 2))
    scorer.apply(ev("camera", "motion", {"state": "start", "area_ratio": 0.1}, T0 + 6))
    assert scorer.registry.get(obj, T0 + 6)["r"] == 1                              # lived ≥ flicker_sec
    assert scorer.registry.get("cam-1/motion", T0 + 6)["r"] == 1                   # motion with an object
    pose = scorer.registry.get("cam-1/pose_action", T0)
    scorer.apply(ev("camera", "action", {"track_id": 2, "action": "standing"}, T0 + 7))
    scorer.apply(ev("camera", "action", {"track_id": 2, "action": "sitting"}, T0 + 8))      # flip
    scorer.apply(ev("camera", "action", {"track_id": 2, "action": "lying"}, T0 + 20))       # held 12 s
    after = scorer.registry.get("cam-1/pose_action", T0 + 20)
    assert after["s"] == pytest.approx(pose["s"] + 1) and after["r"] == pytest.approx(pose["r"] + 1, abs=1e-3)
    scorer.apply(ev("audio", "speech", {"text": "привіт", "start": iso(T0 + 20), "end": iso(T0 + 21)}, T0 + 21,
                    device="mic-1"))
    assert scorer.registry.get("mic-1/vad", T0 + 21)["r"] > scorer.registry.get("mic-1/vad", T0)["r"] - 1e-6
    scorer.apply(ev("audio", "trust_evidence", {"channel": "vad", "kind": "mic", "verdict": "refute",
                                                "reason": "Whisper не почув мови"}, T0 + 22, device="mic-1"))
    assert scorer.registry.get("mic-1/vad", T0 + 22)["stats"]["evidence_refute"] == 1
    scorer.apply(ev("system", "identity_merged", {"person_id": "p2", "source_id": "p1"}, T0 + 30, device="x"))
    assert scorer.registry.get("identity/person_registry", T0 + 30)["s"] == 1


def test_scored_events_carry_registry_source_and_channel_id(scorer):
    e = ev("camera", "object_detected", {"track_id": 9, "class": "person", "state": "appeared"}, T0)
    scorer.apply(e)
    assert e.payload["trust_source"] == "registry" and e.payload["channel_id"] == "cam-1/objects"
    g = ev("gas", "gas_reading", {"gas": "acetone", "concentration_pct": 0.1, "gas_confidence": 0.9}, T0,
           device="gas-1")
    scorer.apply(g)
    assert set(g.payload["channel_trust"]) == {"gas-1/gas_class", "gas-1/gas_concentration", "gas-1/gas_trend"}


def test_unknown_storm_lowers_face_identity_health_only(scorer):
    for i in range(12):
        scorer.apply(ev("system", "identity_created", {"person_id": f"u{i}", "display_name": f"unknown_{i}"},
                        T0 + i, device="x"))
    assert scorer.registry.get("cam-1/face_identity", T0 + 12)["h"] == pytest.approx(0.5)
    assert scorer.registry.get("cam-1/objects", T0 + 12)["h"] == 1.0


def test_health_publisher_emits_on_change_and_refresh():
    sent = []
    publisher = HealthPublisher(lambda *a, **k: sent.append(a), "camera", refresh_sec=60)
    assert publisher.publish({}, 0) and not publisher.publish({}, 10)
    assert publisher.publish({"blurred": {}}, 11) and not publisher.publish({"blurred": {}}, 20)
    assert publisher.publish({"blurred": {}}, 80) and len(sent) == 3
