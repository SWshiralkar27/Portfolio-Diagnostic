"""broker_inspect.py - shows the SHAPE of your broker statement so the app can be built for it, without sharing the statement.

Run from IDLE (F5). A window asks you to pick the file (.xlsx, .xls, .csv or .pdf). It prints a report and saves it as
broker_shape_report.txt next to this script. The report keeps: sheet names, the header row, column types, and the text values of
columns with few distinct values (stock names, tickers, the 'source' column). It hides: every number (shown as a shape like
"<num 5 int digits, 2 dp>"), PAN, e-mail, phone and account-style digit runs, and the value after any "label: value" line.
READ THE REPORT BEFORE YOU SHARE IT: names or IDs written without a label cannot always be recognised.
A password-protected PDF asks for the password in a window; it is used once and never stored or printed.
Nothing is sent anywhere; this script only reads the file you pick.
"""
from __future__ import annotations
import datetime as dt
import re
import sys
from pathlib import Path

FILE = ""            # optionally put the full path of the statement here instead of using the window
MAX_BODY_ROWS = 12
HEADER_WORDS = ("name", "stock", "scrip", "instrument", "company", "security", "ticker", "symbol", "isin", "qty", "quantity", "units", "shares",
                "invest", "cost", "avg", "average", "buy", "value", "price", "ltp", "xirr", "irr", "return", "source", "type", "p&l", "pnl",
                "gain", "holding", "current", "market", "date")
PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
ISIN = re.compile(r"^IN[A-Z0-9]{10}$")
LONG_DIGITS = re.compile(r"\d{6,}")


def num_shape(x):
    try:
        v = float(x)
    except Exception:
        return "<num>"
    s = f"{abs(v):.10f}".rstrip("0").rstrip(".")
    ip, _, fp = s.partition(".")
    return f"<num {len(ip)} int digit{'s' if len(ip) != 1 else ''}" + (f", {len(fp)} dp" if fp else "") + (", negative" if v < 0 else "") + ">"


def mask_text(s, label_only=False):
    s = " ".join(str(s).split())
    if not s:
        return ""
    if ISIN.match(s):
        return s
    s = PAN.sub("<PAN>", s)
    s = EMAIL.sub("<email>", s)
    if ":" in s:
        head, _, tail = s.partition(":")
        if tail.strip():
            return f"{head.strip()[:40]}: <value hidden>"
    s = LONG_DIGITS.sub(lambda m: "9" * len(m.group(0)), s)
    if label_only:
        return f"<text, {len(s)} chars>"
    return s[:60]


def mask_cell(v, label_only=False):
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return num_shape(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return "<date>"
    return mask_text(v, label_only)


def header_score(row):
    cells = [str(c).strip().lower() for c in row if c is not None and str(c).strip() and not isinstance(c, (int, float))]
    return sum(1 for c in cells if any(w in c for w in HEADER_WORDS)), len(cells)


def table_profile(rows, header_idx, out):
    header = [("" if c is None else str(c).strip()) for c in rows[header_idx]]
    body = [r for r in rows[header_idx + 1:] if any(c is not None and str(c).strip() for c in r)]
    out.append(f"  header row (row {header_idx + 1}): {header}")
    out.append(f"  rows below the header with any content: {len(body)}")
    for j, h in enumerate(header):
        vals = [r[j] if j < len(r) else None for r in body]
        nonblank = [v for v in vals if v is not None and str(v).strip() != ""]
        nums = [v for v in nonblank if isinstance(v, (int, float)) and not isinstance(v, bool)]
        dates = [v for v in nonblank if isinstance(v, (dt.datetime, dt.date))]
        kind = "numbers" if len(nums) == len(nonblank) and nonblank else "dates" if len(dates) == len(nonblank) and nonblank else "text" if nonblank else "empty"
        line = f"  col {j + 1} '{h}': {kind}, {len(nonblank)} filled of {len(vals)}"
        if kind == "text":
            distinct = sorted({mask_text(v) for v in nonblank})
            if len(distinct) <= 15:
                line += f"; values: {distinct}"
            else:
                line += f"; {len(distinct)} distinct, e.g. {distinct[:5]}"
        elif kind == "numbers":
            line += "; e.g. " + ", ".join(num_shape(v) for v in nums[:3])
        out.append(line)
    out.append(f"  first {min(MAX_BODY_ROWS, len(body))} body rows (numbers hidden):")
    for r in body[:MAX_BODY_ROWS]:
        out.append("    " + " | ".join(mask_cell(c) for c in r))
    if len(body) > MAX_BODY_ROWS:
        out.append(f"  last rows (often totals): ")
        for r in body[-3:]:
            out.append("    " + " | ".join(mask_cell(c) for c in r))


def inspect_rows(rows, out):
    rows = [list(r) for r in rows]
    scores = [(header_score(r)[0], i) for i, r in enumerate(rows[:60])]
    best = max(scores, default=(0, 0))
    out.append(f"  rows read: {len(rows)}; widest row: {max((len(r) for r in rows), default=0)} columns")
    if best[0] < 3:
        out.append("  no clear header row found (fewer than 3 cells look like column titles). First rows, masked:")
        for r in rows[:15]:
            out.append("    " + " | ".join(mask_cell(c, label_only=True) for c in r))
        return
    h = best[1]
    if h > 0:
        out.append(f"  {h} row(s) above the header (account details, dates ...), masked:")
        for r in rows[:h]:
            txt = " | ".join(x for x in (mask_cell(c, label_only=True) for c in r) if x)
            if txt:
                out.append("    " + txt)
    table_profile(rows, h, out)


def inspect_excel(path, out):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
    out.append(f"workbook: {len(wb.sheetnames)} sheet(s): {wb.sheetnames}")
    for ws in wb.worksheets:
        out.append(f"\nsheet '{ws.title}': {ws.max_row} rows x {ws.max_column} columns, merged ranges: {len(ws.merged_cells.ranges)}, state: {ws.sheet_state}")
        inspect_rows(list(ws.iter_rows(values_only=True)), out)


def inspect_csv(path, out):
    import pandas as pd
    try:
        df = pd.read_csv(path, header=None, sep=None, engine="python", dtype=object, skip_blank_lines=False)
    except Exception:
        df = pd.read_csv(path, header=None, dtype=object)
    rows = []
    for r in df.itertuples(index=False):
        row = []
        for c in r:
            if c is None or (isinstance(c, float) and c != c):
                row.append(None)
                continue
            try:
                row.append(float(str(c).replace(",", "")))
            except ValueError:
                row.append(str(c))
        rows.append(row)
    out.append("csv file")
    inspect_rows(rows, out)


def ask_password():
    try:
        import tkinter
        from tkinter import simpledialog
        root = tkinter.Tk(); root.withdraw()
        pw = simpledialog.askstring("Password", "This PDF is password-protected. Password (used once, never stored):", show="*")
        root.destroy()
        return pw
    except Exception:
        return None


def inspect_pdf(path, out, password=None):
    try:
        import pdfplumber
    except ImportError:
        out.append("PDF support needs the pdfplumber library. In a Windows Command Prompt run:  pip install pdfplumber   then run this script again.")
        return
    try:
        pdf = pdfplumber.open(path, password=password)
    except Exception as e:
        if password is None and ("assword" in str(e) or "ncrypt" in str(e) or type(e).__name__ in ("PDFPasswordIncorrect", "PdfminerException")):
            pw = ask_password()
            if pw:
                return inspect_pdf(path, out, password=pw)
        out.append(f"could not open the PDF ({type(e).__name__}); wrong or missing password?")
        return
    with pdf:
        out.append(f"pdf: {len(pdf.pages)} page(s)")
        for pi, page in enumerate(pdf.pages[:4], 1):
            chars = len(page.chars)
            out.append(f"\npage {pi}: {chars} text characters" + ("  (no text layer: a scanned image would need OCR)" if chars == 0 else ""))
            if chars == 0:
                continue
            tables = page.extract_tables()
            out.append(f"  tables found by line detection: {len(tables)}")
            if not tables:
                tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
                out.append(f"  tables found by text alignment: {len(tables)}")
            for ti, t in enumerate(tables[:2], 1):
                out.append(f" table {ti}: {len(t)} rows x {max((len(r) for r in t), default=0)} columns")
                rows = []
                for r in t:
                    row = []
                    for c in r:
                        if c is None or not str(c).strip():
                            row.append(None)
                            continue
                        txt = " ".join(str(c).split())
                        try:
                            row.append(float(txt.replace(",", "")))
                        except ValueError:
                            row.append(txt)
                    rows.append(row)
                inspect_rows(rows, out)
            lines = (page.extract_text() or "").splitlines()
            out.append(f"  first text lines on page {pi}, masked:")
            for ln in lines[:15]:
                out.append("    " + mask_text(ln, label_only=True))


def inspect_file(path, password=None):
    path = Path(path)
    out = [f"BROKER FILE SHAPE REPORT  ({dt.date.today()})", f"file type: {path.suffix.lower()}   size: {path.stat().st_size:,} bytes",
           "Numbers are hidden; labelled values, PAN, e-mail and long digit runs are masked. Check before sharing.", ""]
    ext = path.suffix.lower()
    try:
        if ext in (".xlsx", ".xlsm"):
            inspect_excel(path, out)
        elif ext == ".xls":
            import pandas as pd
            x = pd.read_excel(path, sheet_name=None, header=None)
            out.append(f"workbook (.xls): sheets {list(x)}")
            for name, df in x.items():
                out.append(f"\nsheet '{name}'")
                inspect_rows([[None if (isinstance(c, float) and c != c) else c for c in r] for r in df.itertuples(index=False)], out)
        elif ext == ".csv":
            inspect_csv(path, out)
        elif ext == ".pdf":
            inspect_pdf(path, out, password)
        else:
            out.append("unsupported file type; expected .xlsx, .xls, .csv or .pdf")
    except Exception as e:
        out.append(f"could not read the file: {type(e).__name__}: {e}")
    return "\n".join(out)


def main():
    path = FILE
    if not path:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="Pick your broker statement", filetypes=[("Statements", "*.xlsx *.xls *.csv *.pdf"), ("All files", "*.*")])
        root.destroy()
    if not path:
        print("No file chosen.")
        return
    report = inspect_file(path)
    print(report)
    target = Path(__file__).resolve().parent / "broker_shape_report.txt"
    target.write_text(report, encoding="utf-8")
    print(f"\nSaved as {target}. Read it, then paste it into the chat.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        FILE = sys.argv[1]
    main()
