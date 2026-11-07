"""
Parameter provenance of Article 1 (A6 / TOD-46) → ``docs/parameters.md``::

    python -m tools.params_doc [--out docs/parameters.md] [--check]

Every universe and term breakpoint of the four rule bases (``source:`` per variable in ``config/fuzzy/*.yaml``), the
critical concentrations (``config/gases.yaml``), the measured input errors (``config/uncertainty.yaml``) and the
thresholds (``config/parameter_sources.yaml``: every key of the ``hfs`` section of config.yaml plus the listed input
thresholds) with its source. Exit code 1 and a list when anything has no source (``--check``: nothing is written).

Source kinds: normative (a standard / regulation), datasheet, measured (our evaluation data), fitted (cost-based
fitting), requirement (the supervisor / the task), design (structural, no effect on the level, or chosen for a
property such as monotonicity), expert (a judgement with its rationale, no measurement yet).
"""
import argparse
import datetime as dt
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import yaml

from core.fuzzy import load_yaml

KINDS = ("normative", "datasheet", "measured", "fitted", "requirement", "design", "expert")
LEVELS = ("gas", "camera", "audio", "top")
TITLES = {"gas": "Subsystem 1 — gas and PM2.5 (risk 1)", "camera": "Subsystem 2 — camera (risk 2)",
          "audio": "Subsystem 3 — microphone (risk 3)", "top": "Top level — overall risk"}


def read(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def term_text(spec) -> str:
    parts = spec if spec and isinstance(spec[0], (list, tuple)) else [spec]
    return " ∪ ".join(f"{p[0]}({', '.join(f'{v:g}' for v in p[1:])})" for p in parts)


def check_source(source: Any, where: str, errors: List[str]) -> Tuple[str, str]:
    if not isinstance(source, dict) or source.get("kind") not in KINDS or not str(source.get("text") or "").strip():
        errors.append(f"{where}: no source (need {{kind: {'|'.join(KINDS)}, text: …}})")
        return "—", ""
    return source["kind"], str(source["text"])


def fuzzy_section(config_dir: str, errors: List[str], kinds: Counter) -> List[str]:
    lines = []
    for level in LEVELS:
        data = load_yaml(os.path.join(config_dir, f"{level}.yaml"))
        lines += [f"### {TITLES[level]}", "", f"Rule base `config/fuzzy/{level}.yaml`: {len(data.get('rules') or [])} "
                  f"rules, the complete table of `modules/hfs/policy.py` {level}() (every cell checked by "
                  "tests/test_hfs.py); operators "
                  f"{', '.join(f'{k} {v}' for k, v in (data.get('operators') or {}).items())}.", "",
                  "| Variable | Universe | Terms (breakpoints) | Source of the universe | Source of the terms |",
                  "| -- | -- | -- | -- | -- |"]
        for name, var in (data.get("variables") or {}).items():
            source = var.get("source") or {}
            u_kind, u_text = check_source(source.get("universe"), f"{level}.{name}.universe", errors)
            t_kind, t_text = check_source(source.get("terms"), f"{level}.{name}.terms", errors)
            kinds.update([u_kind, t_kind])
            terms = "<br>".join(f"{t}: {term_text(spec)}" for t, spec in (var.get("terms") or {}).items())
            unit = f" {var['unit']}" if var.get("unit") else ""
            universe = f"[{var['universe'][0]:g}, {var['universe'][1]:g}]{unit}"
            lines.append(f"| `{name}` | {universe} | {terms} | **{u_kind}** — {u_text} | **{t_kind}** — {t_text} |")
        lines.append("")
    return lines


def walk(node: Any, path: str, sources: Dict[str, Any], found: List[Tuple[str, Any]], missing: List[str]) -> None:
    if path in sources:
        found.append((path, node))
        return
    if isinstance(node, dict) and node:
        for key, value in node.items():
            walk(value, f"{path}.{key}", sources, found, missing)
    else:
        missing.append(path)


def lookup(config: Dict[str, Any], path: str) -> Any:
    node: Any = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(path)
        node = node[key]
    return node


def thresholds_section(config: Dict[str, Any], sources: Dict[str, Any], errors: List[str],
                       kinds: Counter) -> List[str]:
    found: List[Tuple[str, Any]] = []
    missing: List[str] = []
    walk(config.get("hfs") or {}, "hfs", sources, found, missing)
    errors += [f"config.yaml {path}: no source in config/parameter_sources.yaml" for path in missing]
    files = {"audio_types": read("config/audio_types.yaml"), "danger_words": read("config/danger_words.yaml")}
    for path in sources:
        if path.startswith("hfs"):
            continue
        root, _, rest = path.partition(".")
        try:
            if root == "audio_types" and rest == "danger":
                value = {k: v["danger"] for k, v in files["audio_types"]["types"].items()}
            elif root in files:
                value = lookup(files[root], rest)
            else:
                value = lookup(config, path)
        except KeyError:
            errors.append(f"config/parameter_sources.yaml {path}: no such parameter")
            continue
        found.append((path, value))
    lines = ["| Parameter | Value | Source |", "| -- | -- | -- |"]
    for path, value in found:
        kind, text = check_source(sources[path], path, errors)
        kinds[kind] += 1
        shown = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=1000).strip()
        lines.append(f"| `{path}` | {shown[:120]} | **{kind}** — {text} |")
    return lines


def gases_section(errors: List[str], kinds: Counter) -> List[str]:
    lines = ["| Substance | C_crit | Unit | Molar mass | Source |", "| -- | -- | -- | -- | -- |"]
    for name, entry in (read("config/gases.yaml").get("gases") or {}).items():
        if not str(entry.get("source") or "").strip():
            errors.append(f"config/gases.yaml {name}: no source")
        kinds["normative"] += 1
        lines.append(f"| {name} ({entry.get('formula', '')}) | {entry['c_crit']:g} | {entry['unit']} | "
                     f"{entry.get('molar_mass') or '—'} | **normative** — {entry.get('source', '')} |")
    return lines


def uncertainty_section(errors: List[str], kinds: Counter) -> List[str]:
    path = "config/uncertainty.yaml"
    if not os.path.isfile(path):
        errors.append(f"{path} missing (python -m tools.fit_uncertainty)")
        return []
    data = read(path)
    lines = ["| Input | δ / kind | Status | Source |", "| -- | -- | -- | -- |"]
    entries = [(f"gas sensor {k}", v) for k, v in data["gas"]["concentration"]["sensors"].items()]
    entries += [("gas dynamics", data["gas"]["dynamics"])]
    entries += [(f"{s} {k}", v) for s in ("camera", "audio") for k, v in data[s].items()]
    for name, entry in entries:
        if not str(entry.get("source") or "").strip():
            errors.append(f"{path} {name}: no source")
        status = entry.get("status", "—")
        kinds[{"datasheet": "datasheet", "measured": "measured"}.get(status, "design")] += 1
        value = entry.get("delta", entry.get("kind"))
        lines.append(f"| {name} | {value} | {status} | {str(entry.get('source', '')).replace('|', '/')} |")
    return lines


QUESTIONS = """## Questions of the supervisor

**Slide 25 — how are the x-axis values of a membership function defined (e.g. "degree of suspicion of a
played-back recording")?** In Article 1 every x-axis is a measurable quantity in its own unit, with the source in
the tables above: c / C_crit of a normative limit, a share of persons, a share of the frame, signed hours to a
schedule, dB above the microphone's background, a dictionary grade, a speaker outcome. The playback suspicion was
a *binary indicator* of the hybrid (GPT) trust layer (`config/fuzzy.yaml` → `trust.source.variables.playback`:
1 = speech in a location with working cameras and nobody visible) with no measurable degree in between — it is
**removed from Article 1**: the autonomous hierarchical system has no such input. If needed later, it would have to
be redefined through a measurable indicator (e.g. the share of a speech segment's duration during which the
location's cameras saw nobody, with the camera's detection recall as its error).

**Slide 26 — where does the threshold 0.42 come from?** 0.42 is `claim_theta` of the hybrid pipeline: the trust
degree below which a speech / scene-text claim ("all clear", "ignore the alarm") is not believed by the GPT risk
corridor. It was **fitted**, not chosen: `python -m tools.fit_thresholds` minimises the expected cost
C_miss · P_miss + C_fa · P_fa on the calibration scenarios with the cost ratio C_miss / C_fa = {cost_ratio}
(`config.yaml` calibration.cost_ratio) — 33 attack and 30 normal claims, expected cost 0.027
(`config/thresholds.yaml` fits.claim_theta). It belongs to Article 2 (the LLM pipeline) and is **not used in
Article 1** (no claims, no LLM). Another 0.42 in the code is unrelated: the face duplicate *review* threshold of the
identity dedup job (`modules/vision/dedup.py`), the lower edge of the band shown to the operator for a manual
decision.

**Slide 8, item 1 — do triangular membership functions give a more monotonic surface?** Yes, measured in
`reports/article1_<date>/` (`python -m experiments.hfs.run`): the triangular / trapezoidal rule bases (complete
tables on Ruspini partitions, product / sum inference) have no decreasing step on any grid, the same rule bases with
Gaussian terms do — see the monotonicity table there.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Article 1 parameter provenance → docs/parameters.md")
    parser.add_argument("--out", default=os.path.join("docs", "parameters.md"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fuzzy-dir", default=os.path.join("config", "fuzzy"))
    parser.add_argument("--sources", default=os.path.join("config", "parameter_sources.yaml"))
    parser.add_argument("--check", action="store_true", help="only check that every parameter has a source")
    args = parser.parse_args(argv)
    config = read(args.config)
    sources = read(args.sources).get("parameters") or {}
    errors: List[str] = []
    kinds: Counter = Counter()
    fuzzy = fuzzy_section(args.fuzzy_dir, errors, kinds)
    thresholds = thresholds_section(config, sources, errors, kinds)
    gases = gases_section(errors, kinds)
    uncertainty = uncertainty_section(errors, kinds)
    if errors:
        print("Parameters without a source:\n  - " + "\n  - ".join(errors), file=sys.stderr)
        return 1
    if args.check:
        print(f"ok: every parameter has a source ({sum(kinds.values())} sources: "
              f"{', '.join(f'{k} {v}' for k, v in kinds.most_common())})")
        return 0
    cost_ratio = (config.get("calibration") or {}).get("cost_ratio")
    lines = ["# Parameters of Article 1 and their sources", "",
             f"Generated by `python -m tools.params_doc` on {dt.date.today().isoformat()} from config/fuzzy/*.yaml, "
             "config/gases.yaml, config/uncertainty.yaml, config/parameter_sources.yaml and config.yaml; the "
             "command fails when a parameter has no source.", "",
             "Source kinds: **normative** (a standard / regulation), **datasheet**, **measured** (our evaluation "
             "data), **fitted** (cost-based fitting), **requirement** (the supervisor / the task), **design** "
             "(structural, no effect on the level, or chosen for a property such as monotonicity), **expert** (a "
             "judgement with its rationale — no measurement yet; these are the values to measure next).", "",
             "Counts: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()) + ".", "",
             QUESTIONS.replace("{cost_ratio}", str(cost_ratio)), "## Fuzzy variables", "", *fuzzy,
             "## Critical concentrations (config/gases.yaml)", "", *gases, "",
             "## Measured input errors (config/uncertainty.yaml) — the corridor", "", *uncertainty, "",
             "## Thresholds", "", *thresholds, ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"→ {args.out} ({sum(kinds.values())} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
