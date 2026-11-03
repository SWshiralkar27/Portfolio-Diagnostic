"""check_repo_safe.py - run this (F5 in IDLE) BEFORE every commit or upload to GitHub.

It scans the folder this script sits in and refuses (DO NOT COMMIT) if it finds anything that looks like your private data:
holdings or broker statements, evidence/cost/buy-date files, Excel and PDF files, carry-over files, report outputs, secrets,
and PAN numbers or e-mail addresses inside text files. Browser uploads to GitHub ignore .gitignore, so this check does not
rely on it: private files must simply not be in this folder. Only public_data/*.csv may be a .csv.
"""
from __future__ import annotations
import fnmatch
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".idea", ".vscode"}
BAD_NAMES = ["holdings*", "*statement*", "evidence.csv", "evidence_auto.csv", "buy_dates.csv", "cost_overrides.csv", "input_skips.csv",
             "verified_flags.csv", "last_run_snapshot.csv", "portfolio_diagnostic_*.xlsx", "portfolio_report_*.html",
             "broker_shape_report.txt", "carry_over*", "*.bak*", "secrets.toml", ".env", "*.pkl", "*.zip"]
BAD_DIRS = {"diagnostic_output", ".cache", "private", "my_data"}
BAD_SUFFIX = {".xlsx", ".xls", ".xlsm", ".pdf"}
TEXT_SUFFIX = {".csv", ".txt", ".md", ".json", ".toml", ".yaml", ".yml", ".py"}
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
OK_EMAIL_TAIL = ("example.com", "example.org", "example.invalid", "users.noreply.github.com")


def scan(root):
    root = Path(root).resolve()
    problems = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        parts = set(rel.parts)
        if parts & SKIP_DIRS:
            continue
        if p.is_dir():
            if p.name in BAD_DIRS:
                problems.append((str(rel) + "/", "folder that holds private or generated data"))
            continue
        if parts & BAD_DIRS:
            continue                                   # already reported at the folder level
        name = p.name.lower()
        if any(fnmatch.fnmatch(name, pat) for pat in BAD_NAMES):
            problems.append((str(rel), "file name looks private or generated"))
            continue
        if p.suffix.lower() in BAD_SUFFIX:
            problems.append((str(rel), "Excel/PDF files can be broker statements or reports"))
            continue
        if p.suffix.lower() == ".csv" and rel.parts[0] != "public_data":
            problems.append((str(rel), "a .csv outside public_data/"))
            continue
        if p.suffix.lower() in TEXT_SUFFIX:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if PAN.search(txt):
                problems.append((str(rel), "contains something shaped like a PAN number"))
            for m in EMAIL.finditer(txt):
                if not m.group(0).lower().endswith(OK_EMAIL_TAIL):
                    problems.append((str(rel), "contains an e-mail address"))
                    break
    return problems


def main():
    root = Path(__file__).resolve().parent
    problems = scan(root)
    print(f"Scanned {root}")
    if problems:
        print("\nDO NOT COMMIT. Move or delete these first:")
        for rel, why in problems:
            print(f"  {rel}  -  {why}")
        return 1
    print("\nSAFE TO COMMIT: no private-looking files or personal identifiers found.")
    return 0


if __name__ == "__main__":
    code = main()
    if code:
        sys.exit(code)
