"""Proof helpers for check_step27.py: is the single-file version exactly the file the modules were cut from, and are the
functions marked 'moved unchanged' byte-for-byte identical in their modules?"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path


def segments(src: str) -> dict:
    """name -> exact source text of each top-level function, class or assignment."""
    lines = src.replace("\r\n", "\n").split("\n")
    out = {}
    for node in ast.parse("\n".join(lines)).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)): names = [node.name]
        elif isinstance(node, ast.Assign): names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        else: continue
        for n in names:
            out[n] = "\n".join(lines[node.lineno - 1: node.end_lineno])
    return out


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def legacy_mismatches(legacy_path, map_path):
    moved = json.loads(Path(map_path).read_text(encoding="utf-8"))["moved"]
    seg = segments(Path(legacy_path).read_text(encoding="utf-8"))
    missing = sorted(n for n in moved if n not in seg)
    changed = sorted(n for n in moved if n in seg and sha(seg[n]) != moved[n]["sha"])
    return missing, changed


def verbatim_mismatches(pdiag_dir, legacy_path, map_path):
    moved = json.loads(Path(map_path).read_text(encoding="utf-8"))["moved"]
    seg = segments(Path(legacy_path).read_text(encoding="utf-8"))
    bad = []
    for n, m in moved.items():
        if not m["verbatim"]: continue
        mod = Path(pdiag_dir) / f"{m['module']}.py"
        if n not in seg or not mod.exists() or seg[n] not in mod.read_text(encoding="utf-8").replace("\r\n", "\n"):
            bad.append(n)
    return sorted(bad)
