"""
Fuzzy-logic documentation generator: every rule base, membership function, formula and decision step of the system,
explained in plain Ukrainian with figures computed from the live configs.

    python -m tools.fuzzy_docs [--out reports/fuzzy_docs_<date>]   → fuzzy_logic.pdf, fuzzy_logic.docx, fuzzy_logic.html

The PDF is printed by headless Google Chrome (skipped when Chrome is missing); the DOCX is written with the standard
library only (``render.to_docx``) — no extra packages.
"""
