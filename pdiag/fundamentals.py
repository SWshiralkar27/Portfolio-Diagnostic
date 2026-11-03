"""Fundamentals from Yahoo statements, and the valuation metric choice."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pdiag.config import FIN_TYPES, NO_STD_TYPES
from pdiag.utils import fnum


def _row(df, labels):
    if df is None or not hasattr(df, "index") or len(df) == 0: return None
    for lab in labels:
        if lab in df.index:
            s = pd.to_numeric(df.loc[lab], errors="coerce").dropna()
            if len(s):
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return None

def _cagr_series(s):
    if s is None or len(s) < 3 or s.iloc[0] <= 0 or s.iloc[-1] <= 0: return np.nan
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan

def fundamentals(info: dict, st: dict, sector_type: str) -> dict:
    fin, bs, cf = st.get("fin"), st.get("bs"), st.get("cf")
    rev = _row(fin, ["Total Revenue", "Operating Revenue"])
    ebitda = _row(fin, ["EBITDA", "Normalized EBITDA"])
    ebit = _row(fin, ["EBIT", "Operating Income"])
    ni = _row(fin, ["Net Income", "Net Income Common Stockholders"])
    intexp = _row(fin, ["Interest Expense", "Interest Expense Non Operating"])
    debt = _row(bs, ["Total Debt"])
    cash = _row(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
    equity = _row(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    assets = _row(bs, ["Total Assets"])
    curl = _row(bs, ["Current Liabilities", "Total Current Liabilities"])
    cfo = _row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    fcf = _row(cf, ["Free Cash Flow"])
    last = lambda s: s.iloc[-1] if s is not None and len(s) else np.nan
    o = dict(fy_end=(rev.index[-1].date().isoformat() if rev is not None else "n/a"),
             basis="Yahoo annual; consolidated/standalone not stated - verify vs filings")
    o["ebitda_latest"], o["ni_latest"] = last(ebitda), last(ni)
    if debt is not None and not pd.isna(last(ebitda)) and last(ebitda) > 0:
        o["nd_ebitda"] = (last(debt) - (last(cash) if cash is not None else 0)) / last(ebitda)
    if ebit is not None and intexp is not None:
        ie = abs(last(intexp))
        o["int_cov"] = last(ebit) / ie if ie > 0 else 99.0
    if cfo is not None and not pd.isna(last(ebitda)) and last(ebitda) > 0:
        o["cfo_ebitda"] = last(cfo) / last(ebitda)
    o["ni_cagr"], o["rev_cagr"], o["ebitda_cagr"] = _cagr_series(ni), _cagr_series(rev), _cagr_series(ebitda)
    if rev is not None and ebitda is not None:
        m = (ebitda / rev).dropna()
        if len(m) >= 2: o["margin_delta"] = m.iloc[-1] - m.iloc[0]
    if ebit is not None and assets is not None and curl is not None:
        cap_emp = last(assets) - last(curl)
        o["roce"] = 100 * last(ebit) / cap_emp if cap_emp > 0 else np.nan  # percent
    o["pe"] = fnum(info.get("trailingPE"))
    o["pb"] = fnum(info.get("priceToBook"))
    o["roe"] = fnum(info.get("returnOnEquity"))
    mcap = fnum(info.get("marketCap"))
    o["mcap_cr"] = mcap / 1e7 if not np.isnan(mcap) else np.nan
    if fcf is not None and mcap > 0: o["fcf_yield"] = last(fcf) / mcap
    o["neg_networth"] = equity is not None and last(equity) < 0
    o["ni_peak_ratio"], o["pe_norm"] = np.nan, np.nan
    if ni is not None:
        nn = ni.dropna().sort_index().iloc[-4:]
        if len(nn) >= 3 and nn.iloc[-1] > 0:
            base = float(nn.iloc[:-1].mean())
            yoy = [b / a - 1 for a, b in zip(nn.values[:-1], nn.values[1:]) if a > 0]
            spiky = bool((nn <= 0).any() or (yoy and (min(yoy) < -0.15 or max(yoy) > 1.0)))
            if spiky:
                o["ni_peak_ratio"] = float(nn.iloc[-1] / base) if base > 0 else 99.0
                avg = float(nn.mean())
                if avg > 0 and not np.isnan(mcap) and mcap > 0: o["pe_norm"] = mcap / avg
    o["eq_flag"] = ""
    o["eq_note"] = ""
    if sector_type not in FIN_TYPES and cfo is not None and ni is not None:
        j = pd.concat([cfo.rename("cfo"), ni.rename("ni")], axis=1, join="inner").dropna().sort_index()
        latest_fy = ni.dropna().index.max() if len(ni.dropna()) else None
        # Judge cash conversion over the latest three reporting years together: one year's cash flow swings with working
        # capital, and a stale year must never raise a flag today.
        j = j.iloc[-3:]
        if len(j) >= 2 and latest_fy is not None and j.index[-1] == latest_fy and j.ni.sum() > 0:
            cum_cfo, cum_ni = float(j.cfo.sum()), float(j.ni.sum())
            if cum_cfo <= 0:
                o["eq_flag"] = f"cumulative CFO over the last {len(j)} years is negative ({cum_cfo / 1e7:,.0f} Cr) against cumulative PAT of {cum_ni / 1e7:,.0f} Cr"
            elif cum_cfo / cum_ni < 0.5:
                o["eq_note"] = f"cumulative CFO/PAT over the last {len(j)} years is {cum_cfo / cum_ni:.2f} (CFO {cum_cfo / 1e7:,.0f} Cr vs PAT {cum_ni / 1e7:,.0f} Cr)"
            try:
                o["eq_detail"] = "; ".join(f"FY ending {pd.Timestamp(d):%b-%Y}: CFO {c / 1e7:,.0f} Cr vs PAT {n_ / 1e7:,.0f} Cr" for d, c, n_ in zip(j.index, j.cfo, j.ni))
                o["eq_asof"] = str(pd.Timestamp(j.index[-1]).date())
            except Exception:
                pass
    return o

def val_metric(info: dict, sector_type: str):
    pe, pb = fnum(info.get("trailingPE")), fnum(info.get("priceToBook"))
    ev_e, ev_s = fnum(info.get("enterpriseToEbitda")), fnum(info.get("enterpriseToRevenue"))
    eps = fnum(info.get("trailingEps"))
    if sector_type in FIN_TYPES: return "priceToBook", pb, "P/B (financials)"
    if sector_type in NO_STD_TYPES: return None, np.nan, "REIT/holdco: NAV/SOTP not assessed"
    if sector_type == "loss_making" or (not np.isnan(eps) and eps <= 0): return "enterpriseToRevenue", ev_s, "EV/Sales (loss-making)"
    if sector_type == "cyclical": return ("enterpriseToEbitda", ev_e, "EV/EBITDA (cyclical)") if ev_e > 0 else ("priceToBook", pb, "P/B (cyclical)")
    if pe > 0: return "trailingPE", pe, "trailing P/E"
    return "enterpriseToEbitda", ev_e, "EV/EBITDA"
