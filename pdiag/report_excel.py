"""Excel workbook writers: bytes for the app, a file for the command line. Empty sheets are not written."""
from __future__ import annotations
import io
import pandas as pd


def _write(target, sheets):
    with pd.ExcelWriter(target, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if len(df) > 0:
                df.to_excel(xw, sheet_name=name[:31], index=False)


def build_xlsx_bytes(sheets) -> bytes:
    buf = io.BytesIO()
    _write(buf, sheets)
    return buf.getvalue()


def write_xlsx(path, sheets):
    try:
        _write(path, sheets)
        print(f"Excel report: {path}")
    except Exception as e:
        print(f"Could not generate Excel file: {e}")
