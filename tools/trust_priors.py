"""
Channel trust priors from validation results (F22 / TOD-33) → ``config/trust_priors.yaml``.

One entry per device × channel of the trust registry (TOD-34): camera objects / motion / pose_action / smoking /
scene_text, mic vad / asr / asr_local / speaker, gas gas_class / gas_concentration / gas_trend, identity face_known /
face_unknown. Measured: r0 = correct, s0 = wrong on held-out data → τ0 = (r0+1)/(r0+s0+2), u0 = 2/(r0+s0+2), Wilson
95 % lower bound, source file + sha256, n, a one-line Ukrainian explanation. A channel without labelled data is
``prior_status: unmeasured`` with r0 = s0 = 0 (τ0 0.5, u0 1) and what data it needs — never a typed constant.

``policy`` holds rules that are not channel priors (instructions never trusted, a degraded device, …) and the
``interim`` values ``config/fuzzy.yaml`` uses for unmeasured channels until the registry (TOD-34) takes over; every
one is ``measured: false`` with its reason, so the gaps stay visible.

    python -m tools.trust_priors     # after tools.calibrate_channels (asr / speaker / gas) + tools.measure_channels all
"""
import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow `python tools/trust_priors.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import now_iso, prior_entry, read_yaml, sha256_files, wer, write_yaml  # noqa: E402

OUT = "config/trust_priors.yaml"
HEADER = """F22 (TOD-33) channel trust priors per device x channel (TOD-34) — written by python -m tools.trust_priors.
tau0 = (r0+1)/(r0+s0+2) (expected trust), u0 = 2/(r0+s0+2) (uncertainty), wilson_lb = conservative 95 % bound.
prior_status unmeasured = no labelled data yet (r0 = s0 = 0). policy = rules / interim values, measured: false."""

# Channels that need labelled data nobody has recorded yet: what would measure them
UNMEASURED = {
    ("camera", "objects"): "короткі розмічені кліпи з кожної локації (людина / сумка / телефон …): правильні vs "
                           "хибні детекції на трек",
    ("camera", "motion"): "кліпи з реальним рухом і без нього (зокрема зміна освітлення): правильні vs хибні "
                          "події руху",
    ("camera", "scene_text"): "кадри з табличками / екранами і чи відповідав текст дійсності: узгоджені vs "
                              "суперечливі",
}

# Not channel priors: system rules, and interim values for unmeasured channels (used by config/fuzzy.yaml)
POLICY = {
    "interim_physical": (0.95, "тимчасово для camera.objects / camera.motion / scene_active, поки їх не виміряно "
                               "(TOD-34 замінить реєстром)"),
    "interim_scene_text": (0.1, "тимчасово для camera.scene_text: текст у кадрі контролює будь-хто — слабкий доказ"),
    "physical_degraded": (0.3, "пристрій офлайн / з помилкою: його показам майже не віримо"),
    "redundant_disagreement": (0.6, "два датчики газу однієї локації розходяться: межа θ, не вище"),
    "identity_unsettled": (0.8, "людину видно, обличчя ще не розпізнане: сам факт присутності — фізичне спостереження"),
    "deescalation_floor": (0.1, "заспокійлива заява при повністю тривожних сенсорах: майже не віримо"),
    "instruction": (0.05, "інструкція системі («скажи, що безпечно») довіри не отримує ніколи"),
    "gas_disagreement_pct": (1.0, "два датчики газу розходяться більше ніж на 1 % концентрації — вважаємо розбіжністю"),
}


def measured(correct: int, wrong: int, source: str, sha256: str, explanation: str, **meta) -> Dict[str, Any]:
    return {"prior_status": "measured", **prior_entry(correct, wrong, source=source, sha256=sha256, **meta),
            "explanation": explanation}


def unmeasured(needs: str) -> Dict[str, Any]:
    return {"prior_status": "unmeasured", **prior_entry(0, 0), "measured": False, "source": None, "sha256": None,
            "explanation": f"ще не виміряно — потрібні {needs}"}


def from_eval(path: str, explanation: str) -> Optional[Dict[str, Any]]:
    """An entry from ``reports/channel_eval/<channel>.json`` (tools.measure_channels)"""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    extra = {k: data[k] for k in ("unit", "tolerance_pct", "alternatives", "by_truth", "speech", "noise_only",
                                  "noise_false_by_level", "note") if k in data}
    return measured(data["correct"], data["wrong"], data["source"], data["sha256"],
                    explanation.format(**data, n=data["correct"] + data["wrong"]), eval_file=path,
                    measured_at=data.get("measured_at"), **extra)


def gas_class(calibration: Dict[str, Any]) -> Dict[str, Any]:
    deployed = calibration["deployed_test"]
    return measured(deployed["correct"], deployed["wrong"], deployed["path"], deployed["sha256"],
                    f"класифікатор газу: {deployed['correct']} з {deployed['correct'] + deployed['wrong']} "
                    "відкладених кривих розпізнано правильно",
                    pooled_heldout={"n": calibration["n"], "wrong": calibration["wrong"],
                                    "splits": calibration["splits"]})


def face(report_path: str, threshold: float) -> Dict[str, Dict[str, Any]]:
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    row = min(report["by_threshold"], key=lambda r: abs(r["threshold"] - threshold))
    known, strangers = int(report["known_test_images"]), int(report["stranger_images"])
    tp = int(round(row["known_accuracy"] * known))
    fp = int(round(row["false_accept_rate"] * strangers))
    sha = sha256_files([report_path])
    return {
        "face_known": measured(tp, fp, report_path, sha, f"«це відома людина»: {tp} правильних, {fp} хибно прийнятих "
                                                        f"незнайомців (LFW, поріг {row['threshold']})",
                               threshold=row["threshold"]),
        "face_unknown": measured(strangers - fp, known - tp, report_path, sha,
                                 f"«незнайома людина»: {strangers - fp} незнайомців відхилено, {known - tp} відомих "
                                 "пропущено (LFW)", threshold=row["threshold"]),
    }


def asr_xlsx(path: str, sheet: str, wer_max: float) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_excel(path, sheet_name=sheet)
    good = int(((1 - df["Matching Words"] / df["Sentence Length"]) <= wer_max + 1e-9).sum())
    return measured(good, len(df) - good, f"{path}#{sheet}", sha256_files([path]),
                    f"Whisper ({sheet}): {good} з {len(df)} фраз з WER ≤ {wer_max:.0%}")


def asr_csv(path: str, wer_max: float) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    good = sum(wer(r["reference"], r["hypothesis"]) <= wer_max + 1e-9 for r in rows)
    models = ", ".join(sorted({r.get("model", "") for r in rows}))
    return measured(good, len(rows) - good, path, sha256_files([path]),
                    f"{models}: {good} з {len(rows)} синтетичних фраз з WER ≤ {wer_max:.0%}")


def speaker(calibration: Dict[str, Any]) -> Dict[str, Any]:
    counts = calibration["at_voice_threshold"]
    return measured(counts["correct"], counts["wrong"], calibration["dataset"]["path"],
                    calibration["dataset"]["sha256"],
                    f"голос: {counts['correct']} з {counts['correct'] + counts['wrong']} рішень (відомий / "
                    f"незнайомий) правильні на відкладених TTS-голосах, поріг {counts['threshold']}",
                    threshold=counts["threshold"])


def build(cal_dir: str, eval_dir: str, face_report: str, face_threshold: float, xlsx: Optional[str],
          xlsx_sheet: str, wer_max: float) -> Dict[str, Any]:
    channels: Dict[str, Dict[str, Any]] = {"camera": {}, "mic": {}, "gas": {}, "identity": {}}

    def put(device: str, name: str, entry: Optional[Dict[str, Any]], needs: str) -> None:
        channels[device][name] = entry or unmeasured(needs)

    ev = lambda name: os.path.join(eval_dir, f"{name}.json")  # noqa: E731
    put("camera", "objects", None, UNMEASURED[("camera", "objects")])
    put("camera", "motion", None, UNMEASURED[("camera", "motion")])
    put("camera", "pose_action", from_eval(ev("pose_action"), "поза (стоїть / сидить / лежить / йде): {correct} з "
                                                              "{n} відео-сегментів правильні"),
        "розмічені відео-сегменти поз (python -m tools.measure_channels pose_action)")
    put("camera", "smoking", from_eval(ev("smoking"), "куріння (рука до рота): {correct} з {n} епізодів виявлено"),
        "розмічені епізоди куріння (python -m tools.measure_channels smoking)")
    put("camera", "scene_text", None, UNMEASURED[("camera", "scene_text")])
    put("mic", "vad", from_eval(ev("vad"), "VAD: {correct} з {n} кліпів правильно (мова → сегмент, лише шум → "
                                          "нічого)"), "аудіо з розміченою мовою / не-мовою")
    asr_needs = "фрази з еталонним текстом (python -m tools.calibrate_channels asr)"
    put("mic", "asr", asr_xlsx(xlsx, xlsx_sheet, wer_max) if xlsx and os.path.isfile(xlsx) else None, asr_needs)
    asr = read_yaml(os.path.join(cal_dir, "asr.yaml"))
    put("mic", "asr_local", asr_csv(asr["dataset"]["path"], wer_max) if asr else None, asr_needs)
    voice = read_yaml(os.path.join(cal_dir, "speaker.yaml"))
    put("mic", "speaker", speaker(voice) if voice else None,
        "пари голосів (python -m tools.calibrate_channels speaker)")
    gas = read_yaml(os.path.join(cal_dir, "gas.yaml"))
    put("gas", "gas_class", gas_class(gas) if gas else None,
        "відкладені криві (python -m tools.calibrate_channels gas)")
    put("gas", "gas_concentration", from_eval(ev("gas_concentration"),
                                              "концентрація: {correct} з {n} відкладених кривих у межах "
                                              "±{tolerance_pct} %"), "криві з відомою концентрацією")
    put("gas", "gas_trend", from_eval(ev("gas_trend"), "тренд газу: {correct} з {n} оцінок правильні на "
                                                       "відтворених рампах / утриманнях"),
        "послідовності з відомим ростом / спадом")
    if os.path.isfile(face_report):
        channels["identity"].update(face(face_report, face_threshold))
    else:
        for name in ("face_known", "face_unknown"):
            put("identity", name, None, "оцінка облич (python -m tools.eval_faces)")
    policy = {name: {"value": value, "measured": False, "source": "expert", "explanation": why}
              for name, (value, why) in POLICY.items()}
    return {"version": 2, "generated_at": now_iso(), "wer_max": wer_max, "channels": channels, "policy": policy}


def main(argv: Optional[List[str]] = None) -> None:
    from core.config import load_config

    parser = argparse.ArgumentParser(description="Channel trust priors from validation results")
    parser.add_argument("--calibration-dir", default="config/calibration")
    parser.add_argument("--eval-dir", default="reports/channel_eval")
    parser.add_argument("--face-report", default="reports/identity_eval_lfw.json")
    parser.add_argument("--asr-xlsx", default="multi_model_sentence_data.xlsx", help="read only")
    parser.add_argument("--asr-sheet", default="Whisper 1", help="the deployed Azure whisper-1 = Whisper 1")
    parser.add_argument("--wer-max", type=float, default=0.10)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args(argv)
    face_threshold = float(load_config(args.config).get("identity", {}).get("face_threshold", 0.45))
    data = build(args.calibration_dir, args.eval_dir, args.face_report, face_threshold, args.asr_xlsx,
                 args.asr_sheet, args.wer_max)
    write_yaml(args.out, data, HEADER)
    for device, entries in data["channels"].items():
        for name, e in entries.items():
            print(f"{device + '.' + name:<26} {e['prior_status']:<10} r0 {e['r0']:>4} s0 {e['s0']:>3}  "
                  f"τ0 {e['tau0']:.3f}  u0 {e['u0']:.3f}  LB {e['wilson_lb']:.3f}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
