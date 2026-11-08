"""Guard: tests must never touch the real registries (identity, devices) or logs (biometric data, user history)."""
import glob
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARDED = ("data/identity.db*", "data/devices.db*", "data/trust_registry.db*", "data/rule_suggestions.db*",
           "data/notifications.db*", "data/tls/*", "data/rules/changelog.jsonl", "data/rules/versions/*",
           "data/rules/firing.db*", "data/agents_state.json", "data/mode.json", "data/mode_audit.jsonl",
           "data/gas_sensors.json", "data/calibration/devices/*", "config/sensors.yaml", "config/fuzzy.yaml",
           "config/fuzzy/*.yaml",
           "logs/*/*", "reports/identity_metrics.json")


def _state():
    files = [p for pattern in GUARDED for p in glob.glob(os.path.join(REPO, pattern))]
    return {p: os.stat(p).st_mtime_ns for p in files if os.path.isfile(p)}


def pytest_configure(config):
    config.addinivalue_line("markers", "safety_violation_expected: the test provokes a gas safety invariant "
                                       "violation on purpose (G1 / TOD-53)")


@pytest.fixture(autouse=True)
def gas_safety_invariant_holds(request):
    """G1 (TOD-53): a component that gives a gas at / above C_crit less than critical fails the test."""
    from modules.hfs.gas_safety import VIOLATIONS

    before = len(VIOLATIONS)
    yield
    new = VIOLATIONS[before:]
    if request.node.get_closest_marker("safety_violation_expected") is None:
        assert not new, f"gas safety invariant violated: {new}"


@pytest.fixture(autouse=True, scope="session")
def real_data_untouched():
    before = _state()
    yield
    after = _state()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    assert not changed, f"tests modified real data / logs (use tmp_path): {changed}"
