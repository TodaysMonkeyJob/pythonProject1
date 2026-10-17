"""Guard: tests must never touch the real identity registry or logs (biometric data / the user's history)."""
import glob
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARDED = ("data/identity.db*", "logs/*/*")


def _state():
    files = [p for pattern in GUARDED for p in glob.glob(os.path.join(REPO, pattern))]
    return {p: os.stat(p).st_mtime_ns for p in files if os.path.isfile(p)}


@pytest.fixture(autouse=True, scope="session")
def real_data_untouched():
    before = _state()
    yield
    after = _state()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    assert not changed, f"tests modified real data / logs (use tmp_path): {changed}"
