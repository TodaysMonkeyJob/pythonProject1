"""
Build the human-readable logic documentation (D1 / TOD-55) from the live configuration::

    python -m tools.logic_docs                 # → docs/logic/*.md + docs/logic/figures/*.png
    python -m tools.logic_docs --check         # build in a temp folder and verify the rules below
    python -m tools.logic_docs --export        # + docs/logic/export/logika_systemy.{docx,html,pdf (Chrome)}

The text lives in ``tools/logic_docs/text/*.md`` with placeholders that are filled from the configuration:

* ``{{v:key}}`` — a value of ``Facts.values`` (dotted path, Ukrainian number format);
* ``{{t:name}}`` — a generated table;  ``{{f:name|caption}}`` — a generated figure (numbered);
* ``{{F:id}}`` — a formula block from ``FORMULAS`` (numbered «Ф1 …», listed in the formula index).

Checks (``--check``, also tests/test_logic_docs.py): every placeholder resolves; the main parts contain no code
(no code fences, no backticks); no table is typed by hand in the text (every table comes from the data); every
formula has a source or «загальновідома» and is used; every citation [n] exists in the reference list and every
reference is cited.
"""

import argparse
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

TEXT_DIR = os.path.join(os.path.dirname(__file__), "text")
OUT_DIR = os.path.join("docs", "logic")
PARTS = [
    ("README.md", "README.md"),
    ("part1_autonomous.md", "part1_autonomous.md"),
    ("part2_multiagent.md", "part2_multiagent.md"),
    ("references.md", "references.md"),
]
MAIN_PARTS = ("part1_autonomous.md", "part2_multiagent.md")

# id → (formula in plain text, where it is used, source: [n] or «загальновідома»)
FORMULAS: Dict[str, Tuple[str, str, str]] = {
    "share": ("частка критичного рівня = c / C_крит", "газ: порівняння всіх речовин", "[1], [3] (значення C_крит)"),
    "ppm_mg": ("мг/м³ = ppm · M / 24,45 (25 °C, 1 атм)", "газ: переведення одиниць", "[2]"),
    "volpct": ("1 об.% = 10 000 ppm", "газ: переведення одиниць", "загальновідома"),
    "mixture": ("індекс суміші = Σ cᵢ / C_крит,ᵢ ≥ 1 → критично (лише в одній групі дії)", "газ: суміш газів", "[4]"),
    "slope": (
        "швидкість = нахил прямої найменших квадратів за останні 60 с, × 60 (за хвилину)",
        "газ: динаміка",
        "загальновідома",
    ),
    "tri": (
        "трикутна функція належності: μ(x) = max(0, min((x − a)/(b − a), (c − x)/(c − b)))",
        "усі входи й виходи",
        "загальновідома [5]",
    ),
    "trap": (
        "трапецієвидна: μ(x) = max(0, min((x − a)/(b − a), 1, (d − x)/(d − c)))",
        "усі входи й виходи",
        "загальновідома [5]",
    ),
    "and": ("сила правила = добуток належностей його умов", "усі бази правил", "[6], [7]"),
    "centroid": (
        "ризик = Σ wᵣ · cᵣ / Σ wᵣ (центр мас; wᵣ — сила правила, cᵣ — центр його вихідного слова)",
        "усі бази правил",
        "загальновідома [6]",
    ),
    "share_unknown": (
        "частка незнайомих = N_незнайомих / (N_незнайомих + N_відомих)",
        "камера",
        "вимога керівника (слайди 9–11)",
    ),
    "relative_db": (
        "гучність над фоном = рівень кадру − фон (20-й перцентиль медіан за 5 хв), дБ",
        "мікрофон",
        "загальновідома (перцентиль)",
    ),
    "fou": ("смуга (тип-2): верхня крива = ребра зсунуті назовні на δ, нижня — всередину на δ", "коридор", "[7]"),
    "km": ("коридор [y_l, y_r] — алгоритм Карніка–Менделя (центр множин)", "коридор", "[8]"),
    "wilson": (
        "нижня межа Вілсона: (p̂ + z²/2n − z·√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n), z = 1,96",
        "вага агентів, точність каналів",
        "[10]",
    ),
    "top_escalation": (
        "загальний рівень = найвищий + 1, якщо друга підсистема активна; критичний — лише з доказом",
        "верхній рівень",
        "вимога керівника (TOD-51), правило системи",
    ),
}


def load_texts() -> Dict[str, str]:
    return {name: open(os.path.join(TEXT_DIR, src), encoding="utf-8").read() for src, name in PARTS}


def resolve(
    text: str,
    values: Dict[str, Any],
    tables: Dict[str, str],
    figure_names: set,
    counters: Dict[str, int],
    used_formulas: List[str],
    missing: List[str],
    figures_prefix: str,
) -> str:
    def value(match):
        key = match.group(1)
        parts = key.split(".")
        node: Any = None
        for k in range(len(parts), 0, -1):  # the longest dotted prefix that is a key, then walk
            prefix = ".".join(parts[:k])
            if prefix in values:
                node = values[prefix]
                for part in parts[k:]:
                    if isinstance(node, dict):
                        node = node.get(part)
                    elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                        node = node[int(part)]
                    else:
                        node = None
                break
        if node is None or isinstance(node, (dict, list)):
            missing.append(f"v:{key}")
            return f"⟦{key}⟧"
        from tools.logic_docs.data import fmt

        return fmt(node) if isinstance(node, (int, float)) else str(node)

    def table(match):
        name = match.group(1)
        if name not in tables:
            missing.append(f"t:{name}")
            return f"⟦таблиця {name}⟧"
        counters["table"] += 1
        return f"*Таблиця {counters['table']}.*\n\n{tables[name]}"

    def figure(match):
        name, caption = match.group(1), match.group(2)
        if name not in figure_names:
            missing.append(f"f:{name}")
        counters["figure"] += 1
        return f"![{caption}]({figures_prefix}{name}.png)\n\n*Рисунок {counters['figure']}. {caption}*"

    def formula(match):
        fid = match.group(1)
        if fid not in FORMULAS:
            missing.append(f"F:{fid}")
            return f"⟦формула {fid}⟧"
        if fid not in used_formulas:
            used_formulas.append(fid)
        number = used_formulas.index(fid) + 1
        text, _, source = FORMULAS[fid]
        cite = "" if source.startswith("загальновідома") and "[" not in source else f" — джерело: {source}"
        return f"> **Ф{number}.** {text}{cite}"

    text = re.sub(r"\{\{v:([\w.]+)\}\}", value, text)
    text = re.sub(r"\{\{t:(\w+)\}\}", table, text)
    text = re.sub(r"\{\{f:(\w+)\|([^}]+)\}\}", figure, text)
    text = re.sub(r"\{\{F:(\w+)\}\}", formula, text)
    return text


def formula_index(used: List[str]) -> str:
    from tools.logic_docs.data import md_table

    return md_table(
        ["№", "Формула", "Де використовується", "Джерело"],
        [[f"Ф{i}", FORMULAS[f][0], FORMULAS[f][1], FORMULAS[f][2]] for i, f in enumerate(used, start=1)],
    )


def build(out_dir: str = OUT_DIR, root: str = ".") -> Dict[str, Any]:
    from tools.logic_docs.data import Facts
    from tools.logic_docs.figures import build_all

    facts = Facts(root)
    figures_dir = os.path.join(out_dir, "figures")
    if os.path.isdir(figures_dir):
        shutil.rmtree(figures_dir)
    names = set(build_all(facts, figures_dir))
    texts = load_texts()
    counters = {"table": 0, "figure": 0}
    used: List[str] = []
    missing: List[str] = []
    out: Dict[str, str] = {}
    for name in ("README.md", "part1_autonomous.md", "part2_multiagent.md"):
        out[name] = resolve(texts[name], facts.values, facts.tables, names, counters, used, missing, "figures/")
    tables = {**facts.tables, "formulas": formula_index(used)}
    out["references.md"] = resolve(
        texts["references.md"], facts.values, tables, names, counters, used, missing, "figures/"
    )
    os.makedirs(out_dir, exist_ok=True)
    header = (
        "<!-- Згенеровано: python -m tools.logic_docs (tools/logic_docs/text/*.md + config). Не редагуйте "
        "вручну — змініть шаблон і запустіть знову. -->\n\n"
    )
    for name, text in out.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(header + text)
    return {"texts": out, "missing": missing, "formulas": used, "figures": sorted(names)}


def check(result: Dict[str, Any]) -> List[str]:
    """:return: violations of the writing rules (empty = fine)"""
    errors = [f"unresolved placeholder {m}" for m in result["missing"]]
    templates = load_texts()
    for name in MAIN_PARTS:
        text = result["texts"][name]
        if "```" in text or re.search(r"`[^`]+`", text):
            errors.append(f"{name}: code / backticks in the main text")
        for i, line in enumerate(templates[name].splitlines(), start=1):
            if line.lstrip().startswith("|"):
                errors.append(f"{name}:{i}: a table typed by hand (generate it in tools/logic_docs/data.py)")
    unused = [f for f in FORMULAS if f not in result["formulas"]]
    errors += [f"formula {f} is defined but never used" for f in unused]
    refs = set(re.findall(r"^\[(\d+)\]", result["texts"]["references.md"], re.M))
    cited = set()
    for name in ("README.md", *MAIN_PARTS):
        cited |= set(re.findall(r"\[(\d+)\]", result["texts"][name]))
    cited |= {n for f in result["formulas"] for n in re.findall(r"\[(\d+)\]", FORMULAS[f][2])}
    errors += [f"citation [{n}] has no entry in references.md" for n in sorted(cited - refs, key=int)]
    errors += [f"reference [{n}] is never cited" for n in sorted(refs - cited, key=int)]
    return errors


def export(out_dir: str = OUT_DIR) -> List[str]:
    """DOCX (always) + HTML + PDF (headless Chrome, when installed) through the tools/fuzzy_docs renderer"""
    from tools.fuzzy_docs import render

    import datetime as dt

    blocks: List[tuple] = [
        (
            "title",
            "Логіка системи безпеки: автономна система і мультиагентний підхід",
            [
                "Частина 1 — автономна система (стаття 1); частина 2 — мультиагентний підхід (стаття 2)",
                f"Згенеровано з поточних налаштувань {dt.date.today():%d.%m.%Y}",
            ],
        ),
        ("toc",),
    ]
    for name in ("part1_autonomous.md", "part2_multiagent.md", "references.md"):
        blocks += markdown_blocks(open(os.path.join(out_dir, name), encoding="utf-8").read(), out_dir)
        blocks.append(("pagebreak",))
    target = os.path.join(out_dir, "export")
    os.makedirs(target, exist_ok=True)
    made = [render.to_docx(blocks, os.path.join(target, "logika_systemy.docx"), "Логіка системи безпеки")]
    html = os.path.join(target, "logika_systemy.html")
    with open(html, "w", encoding="utf-8") as f:
        f.write(render.to_html(blocks, "Логіка системи безпеки"))
    made.append(html)
    pdf = os.path.join(target, "logika_systemy.pdf")
    if render.html_to_pdf(html, pdf):
        made.append(pdf)
    return made


def markdown_blocks(text: str, base: str) -> List[tuple]:
    """The subset of Markdown the templates use → blocks of tools/fuzzy_docs/render.py"""
    blocks: List[tuple] = []
    lines = [ln for ln in text.splitlines() if not ln.startswith("<!--")]
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()  # the renderer numbers headings itself: drop our numbers
            title = re.sub(r"^Частина \d+\.\s*", "", title)
            title = re.sub(r"^\d+(\.\d+)*\.\s*", "", title)
            blocks.append(({1: "h1", 2: "h2"}.get(level, "h3"), title))
        elif line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            caption = m.group(1)  # the renderer adds «Рис. N.»
            blocks.append(("img", os.path.join(base, m.group(2)), caption, 16.5))
            i += 3
            continue
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip().replace("<br>", "; ") for c in lines[i].strip("|").split("|")]
                if not all(set(c) <= {"-", " "} for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows[0], rows[1:], None))
            continue
        elif line.startswith("> "):
            blocks.append(("formula", line[2:].replace("**", "")))
        elif re.match(r"^(\d+\.|-|\*) ", line) and not line.startswith("*Таблиця"):
            kind = "ol" if line[0].isdigit() else "ul"
            items = []
            while i < len(lines) and re.match(r"^(\d+\.|-|\*) ", lines[i]):
                items.append(re.sub(r"^(\d+\.|-|\*) ", "", lines[i]).strip())
                i += 1
            blocks.append((kind, items))
            continue
        elif not line.startswith("*Таблиця"):
            paragraph = [line]
            while (
                i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|!\[|\||> |\d+\. |- )", lines[i + 1])
            ):
                i += 1
                paragraph.append(lines[i].strip())
            blocks.append(("p", " ".join(paragraph)))
        i += 1
    return blocks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Human-readable logic documentation from the live configuration")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--check", action="store_true", help="build into a temp folder and verify the writing rules")
    parser.add_argument("--export", action="store_true", help="also DOCX / HTML / PDF in <out>/export")
    args = parser.parse_args(argv)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            errors = check(build(tmp))
        for e in errors:
            print(f"FAIL {e}")
        print("ok" if not errors else f"{len(errors)} problems")
        return 1 if errors else 0
    result = build(args.out)
    errors = check(result)
    for e in errors:
        print(f"warning: {e}")
    print(f"→ {args.out}: {len(result['figures'])} figures, {len(result['formulas'])} formulas")
    if args.export:
        for path in export(args.out):
            print(f"→ {path}")
    return 1 if errors else 0
