"""make_upload_folder.py - builds a CLEAN folder with only your code files, ready to drag into GitHub.

Run from IDLE (F5) in your project folder. It copies a fixed list of code files into a new folder named portfolio_repo_upload
(next to your project folder, not inside it), then runs the privacy check on that folder. Your holdings, broker statements,
evidence, cost and buy-date files, reports and carry-over files are never on the list, so they cannot be uploaded by mistake.
GitHub's browser upload ignores .gitignore, which is why the list works the other way round: only named files go in.
Run it again after every step; it rebuilds the folder from scratch.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "portfolio_repo_upload"
FILES = ["portfolio_diagnostic.py", "scoring.py", "earnings_engine.py", "ownership_engine.py", "catalyst_engine.py", "leadership_engine.py",
         "input_assistant.py", "test_input_assistant.py", "step26_spike_app.py", "check_step26.py", "check_step27.py", "check_repo_safe.py",
         "broker_inspect.py", "make_upload_folder.py", "requirements.txt", ".gitignore", "README_REPO.md", "app.py"]
GLOBS = ["apply_step*.py", "check_step*.py"]                      # earlier step scripts are code too
DIRS = {"pdiag": ["*.py", "*.json"], "tests": ["**/*.py"]}
SINGLE = ["legacy/portfolio_diagnostic_v4_monolith.py"]


def main():
    chosen = []
    for f in FILES:
        if (HERE / f).is_file():
            chosen.append(HERE / f)
    for g in GLOBS:
        chosen += [p for p in HERE.glob(g) if p.is_file()]
    for d, pats in DIRS.items():
        for pat in pats:
            chosen += [p for p in (HERE / d).glob(pat) if p.is_file() and "__pycache__" not in p.parts]
    chosen += [HERE / s for s in SINGLE if (HERE / s).is_file()]
    chosen = sorted(set(chosen))
    if not chosen:
        print("No code files found here. Run this from your project folder.")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)
    for p in chosen:
        dst = OUT / p.relative_to(HERE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    print(f"Copied {len(chosen)} code files to {OUT}")

    sys.path.insert(0, str(HERE))
    from check_repo_safe import scan
    problems = scan(OUT)
    left = sorted(p.name for p in HERE.iterdir() if p.is_file() and not (OUT / p.name).exists() and not p.name.startswith("."))
    print("\nLeft out on purpose (stay on your PC only):", ", ".join(left) if left else "nothing")
    if problems:
        bad = OUT.with_name("portfolio_repo_upload_DO_NOT_UPLOAD")
        if bad.exists():
            shutil.rmtree(bad)
        OUT.rename(bad)
        print("\nDO NOT UPLOAD. The privacy check found:")
        for rel, why in problems:
            print(f"  {rel}  -  {why}")
        print(f"The folder was renamed {bad.name}. Fix the file(s) above in your project folder and run this again.")
        return 1
    n = len(chosen)
    print(f"\nREADY. Privacy check passed. Open this folder: {OUT}")
    print("In GitHub: your repository -> Add file -> Upload files -> select EVERYTHING inside that folder (Ctrl+A) and drag it in -> Commit changes.")
    if n > 95:
        print(f"Note: GitHub takes at most 100 files per upload and this folder has {n}: upload the pdiag and tests folders first, then the rest.")
    return 0


if __name__ == "__main__":
    code = main()
    if code:
        sys.exit(code)
