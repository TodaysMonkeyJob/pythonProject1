"""A5 (TOD-45): measured input errors → interval type-2 FOU → corridor [y_l, y_r]."""
import os

import numpy as np
import pytest
import yaml

from core.events import make_event
from core.fuzzy import FuzzyEngine, load_yaml
from core.fuzzy.engine2 import FuzzyEngine2
from core.fuzzy.it2 import brute_force_cos, cos_interval, gauss_sigma_interval, term_interval
from core.fuzzy.membership import parse_term
from modules.hfs.gas import GasSubsystem
from modules.hfs.gas_normalize import GasTable
from modules.hfs.uncertainty import Uncertainty
from modules.hfs.worker import HierarchicalFuzzySystem

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = ("gas", "camera", "audio", "top")


def engines(name):
    engine = FuzzyEngine.from_dict(load_yaml(os.path.join(REPO, "config", "fuzzy", f"{name}.yaml")))
    return engine, FuzzyEngine2(engine, area_weighted=True)


def random_inputs(engine, rng):
    return {v: float(rng.uniform(engine.variables[v].low, engine.variables[v].high))
            for v in engine.variables if v != engine.output}


# --- the IT2 core -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", LEVELS)
def test_delta_zero_reproduces_type1(name):
    engine, it2 = engines(name)
    rng = np.random.default_rng(1)
    for _ in range(150):
        x = random_inputs(engine, rng)
        crisp = engine.infer(x)["crisp"]                          # rounded to 4 decimals
        y_l, y_r = it2.infer(x, deltas={v: 0.0 for v in x}, explain=False)["interval"]
        assert abs(y_l - crisp) <= 1e-4 and abs(y_r - crisp) <= 1e-4


@pytest.mark.parametrize("name", LEVELS)
def test_corridor_width_grows_with_delta(name):
    engine, it2 = engines(name)
    rng = np.random.default_rng(2)
    for _ in range(40):
        x = random_inputs(engine, rng)
        span = {v: engine.variables[v].high - engine.variables[v].low for v in x}
        widths = []
        for scale in (0.0, 0.01, 0.03, 0.08, 0.15):
            result = it2.infer(x, deltas={v: scale * span[v] for v in x}, explain=False)
            widths.append(result["width"])
            assert result["interval"][0] - 1e-9 <= it2.type1(x)["crisp"] <= result["interval"][1] + 1e-9
        assert all(b >= a - 1e-9 for a, b in zip(widths, widths[1:], strict=False)), widths


def test_karnik_mendel_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(1, 9))
        ys = rng.uniform(0, 1, n)
        lo = rng.uniform(0, 1, n)
        hi = lo + rng.uniform(0, 1, n)
        order = np.argsort(ys)
        km, brute = cos_interval(ys, lo, hi), brute_force_cos(list(ys[order]), list(lo[order]), list(hi[order]))
        assert km is not None and brute is not None
        assert abs(km[0] - brute[0]) < 1e-9 and abs(km[1] - brute[1]) < 1e-9


def test_it2_triangle_and_gaussian_uncertain_sigma():
    tri = parse_term(["tri", 0.2, 0.5, 0.8])
    for x in np.linspace(0, 1, 41):
        lo, hi = term_interval(tri, float(x), 0.05, 0, 1)
        t1 = term_interval(tri, float(x), 0.0, 0, 1)
        assert lo <= t1[0] + 1e-12 and t1[1] <= hi + 1e-12 and t1[0] == t1[1]
    shoulder = parse_term(["trap", 0, 0, 0.2, 0.4])                 # the universe edge stays fixed
    assert term_interval(shoulder, 0.0, 0.1, 0, 1) == (1.0, 1.0)
    gauss = parse_term(["gauss", 0.5, 0.1])
    for x in np.linspace(0, 1, 41):
        lo, hi = term_interval(gauss, float(x), 0.0, 0, 1, sigma_k=0.2)
        assert 0 <= lo <= hi <= 1
        t1 = term_interval(gauss, float(x), 0.0, 0, 1, sigma_k=0.0)
        assert t1[0] == pytest.approx(t1[1]) and lo <= t1[0] + 1e-12 <= hi + 1e-12
    assert gauss_sigma_interval(0.5, 0.5, 0.1, 0.3) == (1.0, 1.0)  # the peak is certain, the width is not
    wide, narrow = gauss_sigma_interval(0.7, 0.5, 0.1, 0.4), gauss_sigma_interval(0.7, 0.5, 0.1, 0.1)
    assert wide[0] < narrow[0] < narrow[1] < wide[1]                # larger k → wider FOU


# --- measured errors --------------------------------------------------------------------------------------------------

def test_uncertainty_file_names_a_source_for_every_input():
    data = yaml.safe_load(open(os.path.join(REPO, "config", "uncertainty.yaml"), encoding="utf-8"))
    entries = [*data["gas"]["concentration"]["sensors"].values(), data["gas"]["dynamics"],
               *data["camera"].values(), *data["audio"].values(), data["top"]["risk"]]
    for entry in entries:
        assert entry.get("source") and entry.get("status") in (
            "measured", "datasheet", "derived", "exact", "unmeasured"), entry
        if entry.get("kind") in ("absolute", "decision") and entry.get("status") != "exact":
            assert entry.get("date"), entry                         # measured values carry their date
    assert data["audio"]["keyword_danger"]["n"] > 0 and 0 < data["audio"]["keyword_danger"]["delta"] < 1
    assert data["camera"]["unknown_share"]["p"] > 0.99


def test_fit_uncertainty_reproduces_the_committed_file(tmp_path):
    from tools.fit_uncertainty import main

    assert main(["--out", str(tmp_path / "u.yaml")]) == 0
    fresh = yaml.safe_load(open(tmp_path / "u.yaml", encoding="utf-8"))
    committed = yaml.safe_load(open(os.path.join(REPO, "config", "uncertainty.yaml"), encoding="utf-8"))
    for section, name in (("camera", "unknown_share"), ("camera", "motion_intensity"), ("audio", "keyword_danger"),
                          ("audio", "speaker_unknown")):
        assert fresh[section][name] == committed[section][name]
    assert fresh["gas"]["concentration"] == committed["gas"]["concentration"]


def test_gas_reading_errors_propagate_to_c_norm_and_dynamics():
    u = Uncertainty.load(os.path.join(REPO, "config", "uncertainty.yaml"), GasTable.load())
    assert u.reading_delta("ammonia", 30, "ppm") == 5.0                # DFRobot: ±5 % FS floor …
    assert u.reading_delta("ammonia", 90, "ppm") == pytest.approx(9.0)   # … or ±10 % of the reading
    assert u.norm_delta("ammonia", 90, "ppm") == pytest.approx(9.0 / 300)
    graphene = u.norm_delta("toluene", 0.03, "vol_pct")               # classifier name "toluol" in the file
    assert graphene == pytest.approx(0.0074 / 0.05, rel=1e-3)
    assert u.reading_delta("carbon_monoxide", 100, "ppm") == 0.0       # SEN0466 accuracy unpublished: unmeasured
    times = list(range(0, 61, 2))
    expected = 60 * 0.05 / np.sqrt(sum((t - np.mean(times)) ** 2 for t in times))
    assert u.dynamics_delta(0.05, times) == pytest.approx(expected) and u.dynamics_delta(0.05, [0, 1]) == 0


def test_subsystem_corridor_contains_the_risk_and_vanishes_without_errors():
    noisy = GasSubsystem({})
    exact = GasSubsystem({"hfs": {"it2": False}})
    for gas in (noisy, exact):
        gas.observe(make_event("gas", "gas_reading", {"gas": "ammonia", "concentration": 150, "unit": "ppm"},
                               location_id="lab", device_id="nh3"), 1000.0)
    a, b = noisy.evaluate("lab", 1000.0, True), exact.evaluate("lab", 1000.0, True)
    assert a["risk"] == b["risk"] and b["interval"] == [b["risk"], b["risk"]] and b["corridor_width"] == 0
    assert a["interval"][0] < a["risk"] < a["interval"][1] and a["deltas"]["max_norm"] == pytest.approx(0.05)
    assert a["widened_by"][0]["variable"] in ("max_norm", "mixture_index")


# --- device health -----------------------------------------------------------------------------------------------

def test_stuck_device_is_excluded_with_an_explanation_and_comes_back():
    devices = [{"device_id": "nh3", "kind": "gas", "location_id": "lab", "status": "online", "enabled": True},
               {"device_id": "co", "kind": "gas", "location_id": "lab", "status": "online", "enabled": True}]
    clock = [1000.0]
    system = HierarchicalFuzzySystem({}, devices=lambda: devices, clock=lambda: clock[0])
    reading = lambda dev, gas, c: make_event("gas", "gas_reading", {"gas": gas, "concentration": c, "unit": "ppm"},  # noqa: E731
                                             location_id="lab", device_id=dev)
    system.observe(reading("nh3", "ammonia", 290))
    system.observe(reading("co", "carbon_monoxide", 50))
    [first] = system.tick()
    assert first["level"] == "critical" and first["excluded"] == []
    system.observe(make_event("gas", "device_health", {"kind": "gas", "failing": {
        "flat_readings": {"value": 30, "detail": "30 однакових показів"}, "too_dark": {"value": 1}}},
        location_id="lab", device_id="nh3"))
    system.observe(reading("nh3", "ammonia", 290))                         # ignored while excluded
    clock[0] += 3
    [second] = system.tick()
    assert second["level"] == "low" and second["partial"] is True
    assert second["excluded"] == [{"device_id": "nh3", "kind": "gas", "location_id": "lab",
                                   "reasons": ["flat_readings (30 однакових показів)"]}]
    assert second["subsystems"]["gas"]["excluded"][0]["device_id"] == "nh3"
    system.observe(make_event("gas", "device_health", {"kind": "gas", "failing": {}}, location_id="lab",
                              device_id="nh3"))
    system.observe(reading("nh3", "ammonia", 290))
    clock[0] += 3
    [third] = system.tick()
    assert third["level"] == "critical" and third["excluded"] == []


def test_offline_device_is_excluded_until_online():
    system = HierarchicalFuzzySystem({})
    system.observe(make_event("system", "device_offline", {"device_id": "cam", "kind": "camera",
                                                           "location_id": "lab"}, location_id="lab"))
    assert system.excluded["cam"]["reasons"] == ["offline"]
    system.observe(make_event("system", "device_online", {"device_id": "cam", "kind": "camera",
                                                          "location_id": "lab"}, location_id="lab"))
    assert "cam" not in system.excluded
