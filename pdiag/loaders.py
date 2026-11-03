"""Input loaders. Each accepts a file path OR an uploaded file object (anything with .read and .name), and has a
*_from_df form. Problems are raised as FatalInputError; the library never calls sys.exit."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from pdiag.config import HOLD_ATTRS
from pdiag.errors import FatalInputError
from pdiag.utils import has


def read_table(path) -> pd.DataFrame:
    p = str(getattr(path, "name", path)).lower()
    if hasattr(path, "seek"): path.seek(0)
    if p.endswith((".xlsx", ".xls")): df = pd.read_excel(path)
    else:
        try: df = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)
        except Exception:
            if hasattr(path, "seek"): path.seek(0)
            df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [str(c).replace("\ufeff", "").strip().lower().replace(" ", "_") for c in df.columns]
    return df


def holdings_from_df(df):
    aliases = {
        "symbol": ["symbol", "stock", "ticker", "scrip", "nse_symbol"],
        "quantity": ["quantity", "qty", "shares", "units"],
        "buy_price": ["buy_price", "price", "avg_cost", "cost", "rate"],
        "buy_date": ["buy_date", "date", "purchase_date"]
    }
    for canonical, variants in aliases.items():
        if canonical in df.columns: continue
        for v in variants:
            if v in df.columns:
                df = df.rename(columns={v: canonical}); break
    if "symbol" not in df.columns: raise FatalInputError("Missing 'symbol' column.")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["symbol"])

    def _has_values(col): return col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
    if not _has_values("quantity") or not _has_values("buy_price"):
        print("\n" + "!"*78 + "\nWATCHLIST MODE: Missing quantities/prices. Filling with 1 unit / current price.\n" + "!"*78 + "\n")

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0) if "quantity" in df.columns else 1.0
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce") if "buy_price" in df.columns else np.nan
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce") if "buy_date" in df.columns else pd.NaT
    for c in HOLD_ATTRS:
        if c not in df.columns: df[c] = np.nan
    df["asset_type"] = df["asset_type"].fillna("equity").astype(str).str.lower()
    return df


def load_holdings(path):
    return holdings_from_df(read_table(path))


def evidence_from_df(df):
    df = df.copy(); df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["symbol"])
    return {r["symbol"]: {k: v for k, v in r.items() if has(v)} for r in df.to_dict("records")}


def peers_from_df(df):
    df = df.copy(); df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["symbol"])
    return df.groupby("symbol")["peer"].apply(lambda s: [str(x).strip().upper() for x in s]).to_dict()


def _missing(path):
    return path is None or (not hasattr(path, "read") and (not path or not Path(path).exists()))


def load_evidence(path):
    if _missing(path): return {}
    return evidence_from_df(read_table(path))


def load_peers(path):
    if _missing(path): return {}
    return peers_from_df(read_table(path))
