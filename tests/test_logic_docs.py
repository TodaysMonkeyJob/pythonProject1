"""D1 (TOD-55): the human-readable logic documentation is generated from the live configuration, follows its rules."""

import copy
import os
import re

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from tools.logic_docs.build import build

    out = tmp_path_factory.mktemp("logic")
    return out, build(str(out), REPO)


def test_document_builds_and_follows_the_writing_rules(built):
    from tools.logic_docs.build import check

    out, result = built
    assert check(result) == []
    for name in ("README.md", "part1_autonomous.md", "part2_multiagent.md", "references.md"):
        assert (out / name).is_file()
    assert len(result["figures"]) >= 30 and all((out / "figures" / f"{f}.png").is_file() for f in result["figures"])
    part1 = result["texts"]["part1_autonomous.md"]
    for section in (
        "Сенсори для Raspberry Pi",
        "Підсистема «газ»",
        "Підсистема «камера»",
        "Підсистема «мікрофон»",
        "Верхній рівень",
        "Невизначеність і нечіткий коридор",
        "Калібрування",
        "Відповіді на запитання керівника",
    ):
        assert section in part1, section
    for slide in ("Слайд 8, п. 1", "Слайди 9–11", "Слайд 14", "Слайд 22", "Слайд 25", "Слайд 26"):
        assert slide in part1, slide  # every supervisor question answered
    assert "заплановано" in result["texts"]["part2_multiagent.md"].lower()


def test_checker_catches_code_hand_tables_and_missing_sources(built):
    from tools.logic_docs.build import check

    _, result = built
    broken = copy.deepcopy(result)
    broken["texts"]["part1_autonomous.md"] += "\n```\nprint(1)\n```\nДив. [99].\n"
    broken["missing"] = ["v:gas.unknown"]
    errors = check(broken)
    assert any("code" in e for e in errors) and any("[99]" in e for e in errors)
    assert any("unresolved" in e for e in errors)


def test_numbers_come_from_the_configuration(built):
    from modules.hfs.gas_normalize import GasTable
    from tools.logic_docs.data import fmt

    _, result = built
    text = result["texts"]["part1_autonomous.md"]
    table = GasTable.load(os.path.join(REPO, "config/gases.yaml"))
    for name in ("ammonia", "acetone", "ethanol", "toluene"):
        s = table.substances[name]
        assert f"| {fmt(s.c_crit)} ppm | {fmt(s.c_crit_in('vol_pct'), 4)} |" in text, name
    rules = {
        n: len(yaml.safe_load(open(os.path.join(REPO, f"config/fuzzy/{n}.yaml"), encoding="utf-8"))["rules"])
        for n in ("gas", "camera", "audio", "top")
    }
    assert f"газ — {rules['gas']} правил" in text and f"верхній рівень — {rules['top']}" in text


def test_every_fuzzy_variable_is_explained_in_ukrainian():
    from tools.logic_docs.data import SOURCE_UK, TERM_UK, VAR_UK

    for name in ("gas", "camera", "audio", "top"):
        base = yaml.safe_load(open(os.path.join(REPO, f"config/fuzzy/{name}.yaml"), encoding="utf-8"))
        for var, spec in base["variables"].items():
            assert f"{name}.{var}" in SOURCE_UK and var in VAR_UK, f"{name}.{var}"
            assert set(spec["terms"]) <= set(TERM_UK[var]), f"{name}.{var}"


def test_sensor_catalog_agrees_with_the_drivers():
    catalog = yaml.safe_load(open(os.path.join(REPO, "config/sensor_catalog.yaml"), encoding="utf-8"))["sensors"]
    source = open(os.path.join(REPO, "drivers/gas/particle.py"), encoding="utf-8").read()
    by_driver = {c["driver"]: c for c in catalog if c.get("driver")}
    for driver, c in (("sds011", by_driver["sds011"]), ("sps30", by_driver["sps30"])):
        assert re.search(rf"range=\[0\.0, {c['range'][1]:g}(\.0)?\]", source), driver
        acc = c["accuracy"]
        assert f'"relative": {acc["relative"]:.2f}' in source and f'"absolute": {acc["absolute"]}' in source, driver
