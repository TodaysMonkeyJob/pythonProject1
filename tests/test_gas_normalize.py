"""A2 (TOD-42) critical concentrations and unit conversions."""
import pytest

from modules.hfs.gas_normalize import GasTable, UnitError, convert, readings_of, slope_per_minute


@pytest.fixture(scope="module")
def table():
    return GasTable.load()


def test_gases_table_has_required_substances_with_sources(table):
    for name, c_crit, unit in (("ammonia", 300, "ppm"), ("carbon_monoxide", 1200, "ppm"), ("pm2_5", 225.5, "ug_m3")):
        substance = table.get(name)
        assert (substance.c_crit, substance.unit) == (c_crit, unit) and substance.source
    assert table.get("NH3").name == "ammonia" and table.get("toluol").name == "toluene"
    assert table.get("PM2.5").name == "pm2_5" and table.get("unobtainium") is None


def test_unit_conversions(table):
    assert convert(1200, "ppm", "mg_m3", 28.01) == pytest.approx(1380, rel=0.01)       # CO
    assert convert(300, "ppm", "mg_m3", 17.03) == pytest.approx(210, rel=0.01)         # NH₃
    assert convert(1, "vol_pct", "ppm") == 10_000
    assert convert(1380, "mg_m3", "ppm", 28.01) == pytest.approx(1200, rel=0.01)
    assert convert(225.5, "ug_m3", "mg_m3") == pytest.approx(0.2255)
    assert table.get("ammonia").c_crit_in("vol_pct") == pytest.approx(0.03)         # the graphene model's unit
    with pytest.raises(UnitError):
        convert(10, "ppm", "ug_m3")                     # no molar mass
    with pytest.raises(UnitError):
        convert(10, "ppb", "ppm")


def test_norm_is_clipped(table):
    ammonia = table.get("ammonia")
    assert ammonia.norm(150, "ppm") == 0.5 and ammonia.norm(10.8, "vol_pct") == 1.2 and ammonia.norm(-3, "ppm") == 0


def test_reading_formats():
    assert readings_of({"gas": "ammonia", "concentration_pct": 0.5}) == [("ammonia", 0.5, "vol_pct")]
    assert readings_of({"gas": "co", "concentration": 30, "unit": "ppm"}) == [("co", 30.0, "ppm")]
    assert readings_of({"readings": [{"gas": "pm2_5", "concentration": 12, "unit": "ug_m3"}]}) == \
        [("pm2_5", 12.0, "ug_m3")]


def test_slope_per_minute():
    assert slope_per_minute([(0, 0.0), (30, 0.25), (60, 0.5)]) == pytest.approx(0.5)
    assert slope_per_minute([(0, 0.0), (1, 1.0)]) == 0                    # too few samples
    assert slope_per_minute([(0, 0.0), (1, 0.1), (2, 0.2)], min_span_sec=10) == 0   # span too short
