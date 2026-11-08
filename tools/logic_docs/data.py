"""
Every number and table of the logic documentation, read from the live configuration (D1 / TOD-55): config/gases.yaml,
config/fuzzy/*.yaml, config/uncertainty.yaml, config.yaml, the gas drivers, the agents' eval files and the latest
Article 1 / Article 2 reports. Nothing here is typed by hand except the Ukrainian words for the terms and the short
Ukrainian explanation of each source kind.
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional

import yaml

from core.fuzzy import load_yaml

FUZZY_DIR = os.path.join("config", "fuzzy")
LEVEL_UK = {"low": "низький", "medium": "середній", "high": "високий", "critical": "критичний"}
TERM_UK = {
    "max_norm": {"safe": "безпечно", "moderate": "помірно", "dangerous": "небезпечно", "critical": "критично"},
    "second_norm": {"none": "немає", "low": "низький", "medium": "середній", "high": "високий"},
    "group_index": {"below": "менше 1", "reached": "досягнуто"},
    "third_norm": {"below": "нижче", "high": "високий"},
    "dynamics": {"falling": "спадає", "stable": "стабільно", "rising": "зростає", "sharp_rise": "різко зростає"},
    "unknown_share": {"none": "немає", "part": "частина", "most": "більшість", "all": "усі"},
    "motion_intensity": {"none": "немає", "low": "слабкий", "medium": "помірний", "high": "сильний"},
    "activity_time": {"working": "робочий час", "transition": "перехід", "off_hours": "неробочий час"},
    "intensity": {"low": "тихо", "medium": "помірно", "high": "гучно"},
    "sound_type_danger": {"safe": "безпечний", "suspicious": "підозрілий", "dangerous": "небезпечний"},
    "keyword_danger": {"none": "немає", "alarming": "тривожні", "dangerous": "небезпечні", "critical": "критичні"},
    "speaker_unknown": {"known": "відомий", "uncertain": "невизначений", "unknown": "невідомий"},
    "max_risk": dict(LEVEL_UK),
    "second_risk": {"none": "немає", "low": "низький", "medium": "середній", "high": "високий"},
    "third_risk": {"none": "немає", "active": "активна"},
    "risk": dict(LEVEL_UK),
}
VAR_UK = {
    "max_norm": "найвищий газ (частка критичного рівня)",
    "second_norm": "другий активний газ",
    "group_index": "індекс суміші в групі",
    "third_norm": "третій активний газ",
    "dynamics": "швидкість зміни",
    "unknown_share": "частка незнайомих людей",
    "motion_intensity": "інтенсивність руху",
    "activity_time": "час відносно розкладу",
    "intensity": "гучність над фоном",
    "sound_type_danger": "небезпечність типу звуку",
    "keyword_danger": "небезпечні слова",
    "speaker_unknown": "чий голос",
    "max_risk": "найвищий ризик підсистем",
    "second_risk": "друга активна підсистема",
    "third_risk": "третя активна підсистема",
    "risk": "ризик (вихід)",
}
# what each input / output means and where its universe and terms come from — a Ukrainian rendering of the
# ``source`` texts of config/fuzzy/*.yaml (tests/test_logic_docs.py checks that every variable has one)
SOURCE_UK = {
    "gas.max_norm": "частка критичного рівня (IDLH NIOSH [1], EPA [3]); межі слів — частки 0,1 / 0,3 / 0,6 / 1 "
    "критичного рівня, «критично» з 1",
    "gas.second_norm": "той самий масштаб для другого активного газу; «немає» — спокійна смуга до 5 %",
    "gas.group_index": "сума часток у групі дії; «досягнуто» з 1 — правило сумішей OSHA [4]",
    "gas.third_norm": "третій активний газ; «високий» — від вершини «небезпечно» найвищого газу",
    "gas.dynamics": "±0,5 за хвилину = від нуля до критичного за 2 хв; межі слів — експертна оцінка (розділ 3.5)",
    "gas.risk": "ризик 0…1; слова однакової площі — для монотонності",
    "camera.unknown_share": "частка незнайомих (вимога керівника, слайди 9–11); межі — третини частки",
    "camera.motion_intensity": "частка кадру, що рухається; «немає» — нижче хибного руху порожніх сцен (виміряно), "
    "вершина «помірного» — одна людина, що йде",
    "camera.activity_time": "години до межі розкладу (±12); «перехід» — ±1 год навколо межі (вимога керівника)",
    "camera.risk": "як у газу",
    "audio.intensity": "дБ над фоном цього мікрофона; межі за рівнями мовлення ANSI S3.5 [11]",
    "audio.sound_type_danger": "ступінь небезпеки типу звуку; приклади керівника (клавіатура 0, крик 0,8)",
    "audio.keyword_danger": "найвищий ступінь почутих слів; вершини = ступені словника",
    "audio.speaker_unknown": "результат впізнавання голосу: відомий 0, невизначений 0,5, невідомий 1",
    "audio.risk": "як у газу",
    "top.max_risk": "ризик підсистеми; вершини = центри виходів підсистем, щоб рівень переходив без змін",
    "top.second_risk": "друга активна підсистема (нижня межа коридору вище спокійної смуги)",
    "top.third_risk": "чи активна третя підсистема",
    "top.risk": "як у підсистем",
}
UNIT_UK = {"/min": "за хвилину", "h": "год", "dB": "дБ"}
KIND_UK = {
    "normative": "норматив",
    "datasheet": "паспорт сенсора",
    "measured": "виміряно",
    "fitted": "підібрано",
    "requirement": "вимога",
    "design": "конструктивне рішення",
    "expert": "експертна оцінка",
}
SUBSYSTEMS = {"gas": "газ", "camera": "камера", "audio": "мікрофон", "top": "верхній рівень"}


def fmt(x: Any, digits: int = 3) -> str:
    """Ukrainian number: decimal comma, no trailing zeros, thin thousands separator"""
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer() and abs(x) >= 1):
        n = int(round(x))
        return f"{n:,}".replace(",", " ") if abs(n) >= 10000 else str(n)
    if isinstance(x, float):
        text = f"{x:.{digits}g}" if abs(x) < 1 else f"{x:.{digits}f}".rstrip("0").rstrip(".")
        if "e" in text:
            text = f"{x:.6f}".rstrip("0").rstrip(".")
        return text.replace(".", ",")
    return str(x)


def term_text(parts) -> str:
    shape, *p = parts[0] if isinstance(parts[0], (list, tuple)) else parts
    name = {"trap": "трапеція", "tri": "трикутник", "gauss": "гаус"}.get(shape, shape)
    return f"{name} ({'; '.join(fmt(float(v)) for v in p)})"


def latest(pattern: str) -> Optional[str]:
    found = sorted(glob.glob(pattern))
    return found[-1] if found else None


class Facts:
    """All values of the document, computed once."""

    def __init__(self, root: str = "."):
        self.root = root
        self.config = yaml.safe_load(open(os.path.join(root, "config.yaml"), encoding="utf-8"))
        self.gases_raw = yaml.safe_load(open(os.path.join(root, "config/gases.yaml"), encoding="utf-8"))
        self.uncertainty = yaml.safe_load(open(os.path.join(root, "config/uncertainty.yaml"), encoding="utf-8"))
        self.bases = {n: load_yaml(os.path.join(root, FUZZY_DIR, f"{n}.yaml")) for n in SUBSYSTEMS}
        from modules.hfs.gas_normalize import GasTable

        self.table = GasTable.load(os.path.join(root, "config/gases.yaml"))
        self.values: Dict[str, Any] = {}
        self.tables: Dict[str, str] = {}
        self._gases()
        self._fuzzy()
        self._uncertainty()
        self._sensors()
        self._audio_camera()
        self._calibration()
        self._agents()
        self._results()
        self._examples()

    # --- gases ----------------------------------------------------------------------------------------------------

    def _gases(self) -> None:
        from modules.gas.data import MODEL_RANGE_VOL_PCT

        rows = []
        for name, s in self.table.substances.items():
            raw = self.gases_raw["gases"][name]
            vol = None if s.unit == "ug_m3" else s.c_crit_in("vol_pct")
            ppm = None if s.unit == "ug_m3" else s.c_crit_in("ppm")
            mg = s.c_crit_in("mg_m3") if s.molar_mass else None
            v = {
                "name_uk": s.name_uk,
                "formula": s.formula,
                "c_crit": fmt(s.c_crit),
                "unit": s.unit,
                "ppm": fmt(ppm) if ppm else None,
                "vol_pct": fmt(vol, 4) if vol else None,
                "mg_m3": fmt(round(mg, 1)) if mg else None,
                "lel": fmt(s.lel_pct) if s.lel_pct else None,
                "group": (self.gases_raw.get("groups") or {}).get(s.group, {}).get("name_uk", s.group),
                "five_pct": fmt(round(5.0 / vol * 100)) if vol else None,
                "five_x": fmt(round(5.0 / vol)) if vol else None,
                "lel_x_crit": fmt(round(s.lel_pct / vol, 1)) if s.lel_pct and vol else None,
            }
            self.values[f"gas.{name}"] = v
            source = (
                "NIOSH IDLH [1]"
                if "NIOSH" in raw.get("source", "")
                else "EPA AQI 2024 «Небезпечний» [3]"
                if "EPA" in raw.get("source", "")
                else raw.get("source", "")
            )
            label = s.name_uk if s.formula in (s.name_uk, "") else f"{s.name_uk} ({s.formula})"
            rows.append(
                [
                    label,
                    f"{v['c_crit']} {UNIT_TEXT[s.unit]}",
                    v["vol_pct"] or "—",
                    v["mg_m3"] or "—",
                    f"{v['lel']} об.%" if v["lel"] else "—",
                    v["group"] or "—",
                    source + (" · НКМПР [2]" if s.lel_pct else ""),
                ]
            )
        self.tables["gases"] = md_table(
            ["Речовина", "Критичний рівень", "об.%", "мг/м³", "НКМПР (вибух)", "Група дії", "Джерело"], rows
        )
        self.values["calm_norm"] = fmt(float((self.gases_raw.get("activity") or {}).get("calm_norm", 0.05)))
        self.values["calm_pct"] = fmt(round(float((self.gases_raw.get("activity") or {}).get("calm_norm", 0.05)) * 100))
        groups = self.gases_raw.get("groups") or {}
        self.tables["groups"] = md_table(
            ["Група", "Речовини", "Чому разом"],
            [
                [
                    g.get("name_uk", k),
                    ", ".join(s.name_uk for s in self.table.substances.values() if s.group == k),
                    GROUP_WHY.get(k, ""),
                ]
                for k, g in groups.items()
            ],
        )
        rows = []
        for gas, (lo, hi) in MODEL_RANGE_VOL_PCT.items():
            s = self.table.get(gas)
            crit = s.c_crit_in("vol_pct")
            rows.append(
                [
                    s.name_uk,
                    f"{fmt(lo)} – {fmt(hi)} об.%",
                    f"{fmt(crit, 4)} об.%",
                    f"{fmt(round(lo / crit, 1))} × ({fmt(round(lo / crit * 100))} %)",
                ]
            )
            self.values[f"graphene.{s.name}"] = {
                "lo": fmt(lo),
                "hi": fmt(hi),
                "x": fmt(round(lo / crit, 1)),
                "pct": fmt(round(lo / crit * 100)),
            }
        self.tables["graphene_range"] = md_table(
            ["Газ", "Діапазон навчання моделі", "Критичний рівень", "Початок діапазону = критичного рівня"], rows
        )

    # --- fuzzy variables ----------------------------------------------------------------------------------------------

    def _fuzzy(self) -> None:
        for name, base in self.bases.items():
            self.values[f"rules.{name}"] = len(base["rules"])
            rows = []
            for var, spec in base["variables"].items():
                terms = "<br>".join(
                    f"{TERM_UK.get(var, {}).get(t, t)}: {term_text(p)}" for t, p in spec["terms"].items()
                )
                src = spec.get("source") or {}
                kinds = sorted({KIND_UK.get((src.get(k) or {}).get("kind"), "—") for k in ("universe", "terms")})
                unit = UNIT_UK.get(str(spec.get("unit", "")), spec.get("unit", ""))
                universe = spec["universe"]
                rows.append(
                    [
                        VAR_UK.get(var, var),
                        f"від {fmt(float(universe[0]))} до {fmt(float(universe[1]))}" + (f" {unit}" if unit else ""),
                        terms,
                        f"{', '.join(kinds)}: {SOURCE_UK.get(f'{name}.{var}', 'пояснення бракує')}",
                    ]
                )
                for t, p in spec["terms"].items():
                    first = p[0] if isinstance(p[0], (list, tuple)) else p
                    self.values[f"{name}.{var}.{t}"] = [fmt(float(v)) for v in first[1:]]
            self.tables[f"vars_{name}"] = md_table(
                ["Вхід / вихід", "Шкала", "Слова (терми) і їхні точки", "Звідки значення"], rows
            )
        risk = self.bases["gas"]["variables"]["risk"]["terms"]
        from core.fuzzy import FuzzyEngine

        engine = FuzzyEngine.from_dict(self.bases["gas"])
        self.values["mf04"] = {t: fmt(round(m, 2)) for t, m in engine.fuzzify("max_norm", 0.4).items() if m > 0}
        var = engine.variables["risk"]
        import numpy as np

        grid = var.grid
        centroids = {}
        for t in risk:
            mu = np.array([engine.fuzzify("risk", float(x))[t] for x in grid])
            centroids[t] = float((grid * mu).sum() / mu.sum())
        self.values["risk_centroid"] = {t: fmt(round(c, 2)) for t, c in centroids.items()}
        top = self.config["hfs"].get("top") or {}
        self.values["top.calm_risk"] = fmt(float(top.get("calm_risk", 0.15)))
        self.values["top.co_window"] = fmt(int(top.get("co_window_sec", 60)))
        self.values["hfs.gas.window"] = fmt(int(self.config["hfs"]["gas"].get("window_sec", 60)))
        self.values["hfs.gas.stale"] = fmt(int(self.config["hfs"]["gas"].get("stale_sec", 30)))
        cam = self.config["hfs"]["camera"]
        self.values["camera.hours"] = "–".join(cam["activity_hours"][0])
        self.values["camera.unidentified_after"] = fmt(int(cam.get("unidentified_after_sec", 10)))
        self.tables["schedules"] = md_table(
            ["Тип об'єкта", "Години активності", "Дні"],
            [
                [
                    OBJECT_UK.get(k, k),
                    ", ".join("–".join(h) for h in v["hours"]),
                    "усі" if "days" not in v else ", ".join(DAY_UK[d] for d in v["days"]),
                ]
                for k, v in cam["schedules"].items()
            ],
        )

    # --- uncertainty --------------------------------------------------------------------------------------------------

    def _uncertainty(self) -> None:
        u = self.uncertainty
        graphene = u["gas"]["concentration"]["sensors"]["graphene_fet"]["per_gas"]
        rows = []
        for gas, e in graphene.items():
            s = self.table.get(gas)
            rows.append(
                [
                    f"{s.name_uk}: графеновий сенсор",
                    f"±{fmt(e['absolute'])} об.% "
                    f"(±{fmt(round(e['absolute'] / s.c_crit_in('vol_pct') * 100))} % критичного рівня)",
                    f"виміряно на {e['n_test']} відкладених кривих [наші дані]",
                ]
            )
            self.values[f"delta.graphene.{s.name}"] = fmt(e["absolute"])
        sensors = u["gas"]["concentration"]["sensors"]
        for key, label in (
            ("dfrobot_sen0469", "аміак: DFRobot SEN0469"),
            ("sds011", "PM2.5: Nova SDS011"),
            ("sps30", "PM2.5: Sensirion SPS30"),
            ("dfrobot_sen0466", "CO: DFRobot SEN0466"),
        ):
            e = sensors.get(key) or {}
            if e.get("kind") == "unmeasured":
                rows.append([label, "не опубліковано", "паспорт не містить точності — у коридорі 0"])
                continue
            unit = UNIT_TEXT.get(e.get("unit"), e.get("unit"))
            rows.append(
                [
                    label,
                    f"±{fmt(round(e['relative'] * 100))} % показу або ±{fmt(e['absolute'])} {unit} (більше з двох)",
                    "паспорт сенсора [13]",
                ]
            )
        cam, au = u["camera"], u["audio"]
        rows += [
            [
                "частка незнайомих (розпізнавання облич)",
                f"ймовірність помилки ≤ {fmt(round(1 - cam['unknown_share']['p'], 4))} на одну людину",
                "виміряно на LFW [17] (650 рішень, нижня межа Вілсона [10])",
            ],
            [
                "інтенсивність руху",
                f"±{fmt(cam['motion_intensity']['delta'])} частки кадру",
                "виміряно: найбільший хибний рух на порожніх сценах (тест камери)",
            ],
            ["час відносно розкладу", "0", "годинник і розклад точні"],
            ["гучність над фоном", "половина розкиду фону (живий розрахунок)", "статистика самого мікрофона"],
            ["тип звуку", "—", "моделі типів звуку ще немає: тип звуку не є доказом (заплановано)"],
            [
                "небезпечні слова",
                f"±{fmt(au['keyword_danger']['delta'])} (оцінка ступеня)",
                f"виміряно: {au['keyword_danger']['n']} небезпечних фраз, точність "
                f"{fmt(round(au['keyword_danger']['accuracy'] * 100))} %, нижня межа "
                f"{fmt(round(au['keyword_danger']['accuracy_lower'] * 100))} % [10]",
            ],
            [
                "чий голос (впізнавання мовця)",
                f"±{fmt(au['speaker_unknown']['delta'])}",
                f"виміряно: EER {fmt(round(au['speaker_unknown']['eer'] * 100, 1))} % на "
                f"{au['speaker_unknown']['n']} парах голосів [15]",
            ],
        ]
        self.tables["uncertainty"] = md_table(["Вхід", "Похибка δ", "Джерело"], rows)
        self.values["delta.motion"] = fmt(cam["motion_intensity"]["delta"])
        self.values["delta.keyword"] = fmt(au["keyword_danger"]["delta"])
        self.values["keyword.accuracy"] = fmt(round(au["keyword_danger"]["accuracy"] * 100))
        self.values["keyword.lower"] = fmt(round(au["keyword_danger"]["accuracy_lower"] * 100))
        self.values["keyword.n"] = fmt(au["keyword_danger"]["n"])
        self.values["speaker.eer"] = fmt(round(au["speaker_unknown"]["eer"] * 100, 1))
        self.values["speaker.n"] = fmt(au["speaker_unknown"]["n"])
        self.values["identity.p"] = fmt(round(cam["unknown_share"]["p"] * 100, 2))

    # --- sensors ------------------------------------------------------------------------------------------------------

    def _sensors(self) -> None:
        from drivers.gas.electrochemical import GAS_TYPES
        from modules.gas.data import MODEL_RANGE_VOL_PCT

        graphene = self.uncertainty["gas"]["concentration"]["sensors"]["graphene_fet"]["per_gas"]
        lo = min(r[0] for r in MODEL_RANGE_VOL_PCT.values())
        hi = max(r[1] for r in MODEL_RANGE_VOL_PCT.values())
        errors = [e["absolute"] for e in graphene.values()]
        rows = [
            [
                "наш графеновий сенсор (rGO / пористий кремній)",
                ", ".join(self.table.get(g).name_uk for g in MODEL_RANGE_VOL_PCT),
                "об.%",
                f"{fmt(lo)}–{fmt(hi)} об.%: починається вище критичного рівня кожного газу",
                f"похибка моделі ±{fmt(min(errors))}…{fmt(max(errors))} об.% (виміряно)",
                "власна плата, USB / послідовний порт",
                "наші дані: тестова вибірка моделі",
            ]
        ]
        for gas, model, rng, res, acc, _source in GAS_TYPES.values():
            if rng is None:
                continue
            s = self.table.get(gas)
            crit = s.c_crit_in("ppm")
            accuracy = (
                f"±{fmt(round(acc['relative'] * 100))} % показу або ±{fmt(acc['absolute'])} ppm (більше)"
                if acc
                else "точність не опубліковано"
            )
            rows.append(
                [
                    model,
                    s.name_uk,
                    "ppm",
                    f"0–{fmt(rng[1])} ppm (до {fmt(round(rng[1] / crit, 2))} × критичного рівня)",
                    f"роздільність {fmt(res)} ppm; {accuracy}",
                    "I2C / UART",
                    "DFRobot wiki [13]" + ("; точність — відповідь DFRobot на форумі" if acc else ""),
                ]
            )
        catalog = yaml.safe_load(open(os.path.join(self.root, "config/sensor_catalog.yaml"), encoding="utf-8"))
        for c in catalog["sensors"]:
            names = [self.table.get(g).name_uk if self.table.get(g) else "індекс якості повітря" for g in c["measures"]]
            unit = UNIT_TEXT.get(c["unit"], "індекс" if c["unit"] == "index" else c["unit"])
            if c.get("range") is None:
                rng = "не перевірено"
            else:
                rng = f"{fmt(c['range'][0])}–{fmt(c['range'][1])} {unit}"
                s = self.table.get(c["measures"][0])
                if s is not None:
                    rng += f" (до {fmt(round(c['range'][1] / s.c_crit_in(c['unit']), 2))} × критичного рівня)"
            acc = c.get("accuracy")
            if isinstance(acc, dict):
                acc = f"±{fmt(round(acc['relative'] * 100))} % або ±{fmt(acc['absolute'])} {unit} (більше)"
                if c.get("resolution"):
                    acc = f"роздільність {fmt(c['resolution'])}; {acc}"
            rows.append([c["model"], ", ".join(names), unit, rng, acc or "не перевірено", c["interface"], c["source"]])
        for gas, _model, rng, *_ in GAS_TYPES.values():
            if rng and gas in ("ammonia", "carbon_monoxide"):
                self.values[f"sensor.{gas}_frac"] = fmt(round(rng[1] / self.table.get(gas).c_crit_in("ppm") * 100))
        self.tables["sensors"] = md_table(
            [
                "Сенсор",
                "Що вимірює",
                "Одиниця",
                "Діапазон",
                "Точність / похибка",
                "Підключення до Raspberry Pi",
                "Джерело",
            ],
            rows,
        )

    # --- audio / camera -----------------------------------------------------------------------------------------------

    def _audio_camera(self) -> None:
        words = yaml.safe_load(open(os.path.join(self.root, "config/danger_words.yaml"), encoding="utf-8"))
        grades = words["grades"]
        rows = []
        for grade, value in grades.items():
            listed = [w if isinstance(w, str) else w["word"] for w in words["words"].get(grade, [])]
            uk = [w for w in listed if any("а" <= ch <= "я" or ch in "іїєґ" for ch in w.lower())]
            rows.append(
                [
                    GRADE_UK[grade],
                    fmt(float(value)),
                    ", ".join(uk[:8]) + (" …" if len(uk) > 8 else ""),
                    str(len(listed)),
                ]
            )
            self.values[f"grade.{grade}"] = fmt(float(value))
        self.tables["danger_words"] = md_table(
            ["Ступінь", "Значення", "Приклади слів (українською)", "Слів і фраз"], rows
        )
        types = yaml.safe_load(open(os.path.join(self.root, "config/audio_types.yaml"), encoding="utf-8"))
        self.tables["sound_types"] = md_table(
            ["Тип звуку", "Небезпечність (0–1)"],
            [[t["label"], fmt(float(t["danger"]))] for t in types["types"].values()],
        )
        mic = (self.config.get("audio") or {}).get("mic") or {}
        self.values["audio.window_min"] = fmt(round(float(mic.get("background_window_sec", 300)) / 60))
        self.values["audio.percentile"] = fmt(int(mic.get("background_percentile", 20)))
        vision = self.config.get("vision") or {}
        self.values["vision.background_sec"] = fmt(float(vision.get("motion_background_sec", 20)))
        self.values["vision.grid"] = fmt(int(vision.get("motion_grid", 8)))
        self.values["actions.mode"] = (self.config.get("actions") or {}).get("mode", "per_location")

    # --- calibration ------------------------------------------------------------------------------------------------

    def _calibration(self) -> None:
        kinds = (self.config.get("sensor_calibration") or {}).get("kinds") or {}
        rows = []
        for kind, p in kinds.items():
            rows.append(
                [
                    CAL_KIND_UK.get(kind, kind),
                    CAL_STORED_UK.get(kind, "—"),
                    "заводська" if p.get("factory") else CAL_CHECK_UK.get(kind, "коротка перевірка"),
                    f"{fmt(int(p['warmup_sec']))} с" if p.get("warmup_sec") else "—",
                    f"{fmt(int(p['max_age_days']))} дн." if p.get("max_age_days") and not p.get("factory") else "—",
                    fmt(int(p["max_failures"])) if p.get("max_failures") and not p.get("factory") else "—",
                ]
            )
        self.tables["calibration"] = md_table(
            [
                "Сенсор",
                "Що зберігається",
                "Перевірка під час запуску",
                "Прогрів",
                "Найбільший вік",
                "Невдалих перевірок до перекалібрування",
            ],
            rows,
        )
        restart = os.path.join(self.root, "reports", "calibration_restart", "restart.json")
        if os.path.isfile(restart):
            r = json.load(open(restart, encoding="utf-8"))
            self.tables["ready_time"] = md_table(
                ["Сенсор", "Перший запуск, с", "Перезапуск через 1 хв, с"],
                [
                    [
                        CAL_KIND_UK.get(k, k),
                        fmt(v["first_start"]["ready_after_sec"]),
                        fmt(v["restart"]["ready_after_sec"]),
                    ]
                    for k, v in r["rows"].items()
                ],
            )

    # --- agents / modes -----------------------------------------------------------------------------------------------

    def _agents(self) -> None:
        rows = []
        for name in ("fuzzy", "video", "audio", "validation", "echo"):
            manifest = yaml.safe_load(open(os.path.join(self.root, "agents", name, "agent.yaml"), encoding="utf-8"))
            ev_path = os.path.join(self.root, "agents", name, "eval.yaml")
            weight = "—"
            if os.path.isfile(ev_path):
                ev = yaml.safe_load(open(ev_path, encoding="utf-8"))
                weight = f"{fmt(ev['accuracy'])} (n = {fmt(ev['n'])})"
                self.values[f"agent.{name}.weight"] = fmt(ev["accuracy"])
                self.values[f"agent.{name}.n"] = fmt(ev["n"])
            m = manifest.get("agent", manifest)
            rows.append(
                [AGENT_UK[name][0], AGENT_UK[name][1], AGENT_UK[name][2], weight, "так" if m.get("required") else "ні"]
            )
        thresholds = yaml.safe_load(open(os.path.join(self.root, "config/thresholds.yaml"), encoding="utf-8"))
        self.values["claim_theta"] = fmt(float(thresholds["values"]["claim_theta"]))
        self.values["cost_ratio"] = fmt(float((self.config.get("calibration") or {}).get("cost_ratio", 10)))
        self.tables["agents"] = md_table(
            ["Агент", "Що бачить", "Що каже", "Вага (виміряна точність)", "Обов'язковий"], rows
        )

    # --- worked examples -----------------------------------------------------------------------------------------

    def _examples(self) -> None:
        from experiments.hfs.run import run_scenario

        cfg = {
            "hfs": self.config["hfs"],
            "identity": self.config.get("identity", {}),
            "gas": self.config.get("gas", {}),
        }
        self.examples = {}
        for key, scenario in EXAMPLES.items():
            r = run_scenario(scenario, cfg, 42)
            self.examples[key] = r
            v = {
                "level": LEVEL_UK[r["level"]],
                "risk": fmt(round(r["risk"], 2)),
                "lo": fmt(round(r["interval"][0], 2)),
                "hi": fmt(round(r["interval"][1], 2)),
                "width": fmt(round(r["interval"][1] - r["interval"][0], 2)),
            }
            for i, g in enumerate(scenario.get("gas") or []):
                sub = self.table.get(g["gas"])
                ratio = sub.ratio(g["value"], g["unit"])
                v[f"g{i}"] = {
                    "name": sub.name_uk,
                    "value": fmt(g["value"]),
                    "unit": UNIT_TEXT[g["unit"]],
                    "share": fmt(round(ratio, 2)),
                    "pct": fmt(round(ratio * 100)),
                }
            for sub, (level, risk, interval) in r["subsystems"].items():
                v[sub] = {
                    "level": LEVEL_UK[level],
                    "risk": fmt(round(risk, 2)),
                    "lo": fmt(round(interval[0], 2)),
                    "hi": fmt(round(interval[1], 2)),
                }
            self.values[f"ex.{key}"] = v
        nh3 = self.table.get("ammonia")
        self.values["corridor"] = {
            "ppm": fmt(120),
            "delta_ppm": fmt(12),
            "share": fmt(round(120 / nh3.c_crit, 2)),
            "delta": fmt(round(12 / nh3.c_crit, 2)),
        }

    # --- results ------------------------------------------------------------------------------------------------------

    def _results(self) -> None:
        a1 = latest(os.path.join(self.root, "reports", "article1_*", "metrics.json"))
        if a1:
            m = json.load(open(a1, encoding="utf-8"))
            self.values["a1.report"] = os.path.basename(os.path.dirname(a1))
            self.values["a1.scenarios"] = fmt(m["manifest"]["scenarios"])
            for k, v in m["summary"].items():
                self.values[f"a1.{k}.f1"] = fmt(v["macro_f1"])
            mono = m.get("monotonicity") or {}
            rows = []
            for level, shapes in mono.items():
                rows.append(
                    [SUBSYSTEMS.get(level, level)]
                    + [
                        f"{fmt(round(shapes[s]['share'] * 100, 1))} % "
                        f"({fmt(shapes[s]['falls'])} з {fmt(shapes[s]['steps'])})"
                        for s in ("triangular", "gaussian_inputs", "gaussian_all")
                        if s in shapes
                    ]
                )
            self.tables["monotonicity"] = md_table(
                ["Рівень", "Трикутні / трапецієвидні", "Гаусові входи", "Усе гаусове"], rows
            )
        a2 = latest(os.path.join(self.root, "reports", "article2_*", "metrics.json"))
        if a2:
            m = json.load(open(a2, encoding="utf-8"))
            self.values["a2.report"] = os.path.basename(os.path.dirname(a2))
            rows = []
            for name, s in m["summary"].items():
                rows.append(
                    [
                        CONFIG_UK.get(name, name),
                        fmt(s["macro_f1"]),
                        fmt(round(s["attack_success_rate"] * 100)) + " %",
                        fmt(round(s["operator_load"] * 100, 1)) + " %",
                    ]
                )
            self.tables["article2"] = md_table(["Конфігурація", "Macro-F1", "Успішні атаки", "Запити оператору"], rows)


EXAMPLES = {  # worked examples, run through the real HFS (experiments.hfs.run.run_scenario, seed 42)
    "two_gases": {
        "id": "ex1",
        "group": "gas",
        "gas": [
            {"gas": "ammonia", "value": 45, "unit": "ppm"},
            {"gas": "carbon_monoxide", "value": 504, "unit": "ppm"},
        ],
    },
    "ammonia_120": {"id": "ex2", "group": "gas", "gas": [{"gas": "ammonia", "value": 120, "unit": "ppm"}]},
    "acetone_5": {"id": "ex3", "group": "gas", "gas": [{"gas": "acetone", "value": 5, "unit": "vol_pct"}]},
    "stranger_night": {
        "id": "ex4",
        "group": "camera",
        "camera": {"unknown": 1, "known": 1, "motion": 0.05, "time": "23:00"},
    },
    "help_unknown": {
        "id": "ex5",
        "group": "audio",
        "audio": {"intensity": 14, "speech": {"text": "допоможіть", "speaker": "unknown"}},
    },
    "gas_and_stranger": {
        "id": "ex6",
        "group": "combined",
        "gas": [{"gas": "ammonia", "value": 180, "unit": "ppm"}],
        "camera": {"unknown": 1, "motion": 0.0, "time": "23:00"},
    },
    "quiet_day": {"id": "ex7", "group": "combined", "camera": {"known": 2, "motion": 0.05}},
}


UNIT_TEXT = {"ppm": "ppm", "vol_pct": "об.%", "mg_m3": "мг/м³", "ug_m3": "мкг/м³"}
GROUP_WHY = {
    "asphyxiant": "заважають диханню / перенесенню кисню",
    "irritant": "подразнюють дихальні шляхи",
    "solvents": "діють на нервову систему (наркотична дія розчинників)",
    "particles": "частинки диму",
}
OBJECT_UK = {"apartment": "квартира", "office": "офіс", "warehouse": "склад", "factory": "завод"}
DAY_UK = {0: "пн", 1: "вт", 2: "ср", 3: "чт", 4: "пт", 5: "сб", 6: "нд"}
GRADE_UK = {"alarming": "тривожні", "dangerous": "небезпечні", "critical": "критичні"}
CAL_KIND_UK = {
    "graphene_fet": "графеновий сенсор",
    "mos": "MOS-сенсор (MQ)",
    "electrochemical": "електрохімічний",
    "particle": "сенсор частинок",
    "index_only": "ENS160",
    "mic": "мікрофон",
    "camera": "камера",
}
CAL_STORED_UK = {
    "graphene_fet": "крива чистого повітря, точка Дірака, пік",
    "mos": "R₀ у чистому повітрі",
    "mic": "рівень фону (дБ) і спектр шуму",
    "camera": "зображення фону і шум кожної комірки",
}
CAL_CHECK_UK = {
    "graphene_fet": "перший замір чистого повітря проти збереженого",
    "mos": "R_s / R₀ після прогріву",
    "mic": "3 с фону проти збереженого",
    "camera": "15 кадрів проти збереженого фону",
}
AGENT_UK = {
    "fuzzy": ("нечіткий", "результат нечіткої системи (статті 1)", "ризик і коридор"),
    "video": ("відео", "дії, людей, предмети", "дію з упевненістю, «ризик не менше ніж»"),
    "audio": ("аудіо", "мовлення і типи звуків", "зміст мови, небезпечні слова, звуки"),
    "validation": ("валідаційний", "заяви всіх агентів", "остаточне рішення"),
    "echo": ("зразок (echo)", "—", "шаблон для нових агентів, не впливає"),
}
CONFIG_UK = {
    "article1": "стаття 1 (лише нечітка система)",
    "multi_agent": "мультиагентна",
    "no_video": "без відеоагента",
    "no_audio": "без аудіоагента",
    "no_physical_guard": "без правила фізичного підтвердження",
}


def md_table(header: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("--" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows]
    return "\n".join(out)
