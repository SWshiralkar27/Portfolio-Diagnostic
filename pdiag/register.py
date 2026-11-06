"""Evidence Register: one row per flag, with source, threshold, date and status."""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
from pdiag.config import CFG


def build_evidence_register(HD, SCD, fundd, auto_evid, evid, total_value, verified=None):
    """One row per flag the report raises: what was observed, the threshold used, where it came from, how fresh it is
    and what it did to the output. Everything here is automated and NOT independently verified until you confirm it.
    To record a verification, add a row to verified_flags.csv next to evidence.csv:
        symbol,flag_type,verified_on,note        e.g.  LTM,Cost basis,2026-10-28,"checked against broker statement"
    """
    import json
    cols = ["Symbol", "FlagType", "Category", "Observed", "Threshold", "Source", "SourceLink", "SourceDate", "Retrieved", "Status", "Effect"]
    rows = []
    today_ = dt.date.today().isoformat()

    def add(sym, ftype, category, observed, threshold, source, link, src_date, retrieved, effect):
        rows.append(dict(Symbol=sym, FlagType=ftype, Category=category, Observed=str(observed), Threshold=threshold, Source=source,
                         SourceLink=str(link or ""), SourceDate=str(src_date or ""), Retrieved=str(retrieved or ""),
                         Status="Automated - not independently verified", Effect=effect))

    f2 = SCD.set_index("Symbol")["F2"] if len(SCD) else pd.Series(dtype=float)

    # 1. flags raised by the scorecard (earnings quality and other rule-based flags)
    for _, r in SCD.iterrows():
        s = r["Symbol"]
        fd = fundd.get(s, {}) or {}
        for item in [x for x in str(r.get("Review_Risk", "")).split(" | ") if x]:
            hard = item.startswith("HARD OVERRIDE")
            text = item.replace("HARD OVERRIDE: ", "")
            if "Earnings-quality" in text:
                add(s, "Earnings quality", "Financial statements", fd.get("eq_detail") or text,
                    "Latest 3 reporting years together: cumulative CFO at or below zero while cumulative PAT is positive (hard override), or cumulative CFO/PAT below 0.5 (risk tag)",
                    "Yahoo Finance annual statements via yfinance (not the audited filing)", "", fd.get("eq_asof", ""), today_,
                    "Hard override: label becomes 'Candidate for review'" if hard else "Risk tag")
            else:
                add(s, "Rule-based flag", "Scorecard rule", text, "see the Method sheet", "Tool rule on Yahoo / NSE data", "", "", today_,
                    "Hard override: label becomes 'Candidate for review'" if hard else "Risk tag")

    # 2. disclosures (NSE announcements, last 90 days)
    for s, rec in auto_evid.items():
        raw = rec.get("flag_log")
        if not raw:
            continue
        try:
            items = json.loads(raw)
        except Exception:
            continue
        for it in items:
            effect = {"red": "Scored F4 = 5", "down": "Scored F4 = 4", "watch": "Risk tag only (not scored)"}.get(it.get("kind"), "")
            add(s, "Disclosure", "NSE announcement", f"{it.get('label')}: {it.get('text')} [matched: '{it.get('matched')}']",
                "Keyword rule on the announcement text, last 90 days", "NSE corporate announcements via dalal", it.get("url", ""),
                it.get("date", ""), str(rec.get("fetched_at", ""))[:10], effect)

    # 3. promoter holding falls
    for s, rec in evid.items():
        try:
            pn, p1 = float(rec.get("promoter_holding_now")), float(rec.get("promoter_holding_1y"))
        except (TypeError, ValueError):
            continue
        chg = pn - p1
        if chg <= -1.0 and not (pn < 1.0 and p1 < 1.0):
            soft = " (softened one notch: majority control intact)" if pn >= 51.0 else ""
            sc_ = f2.get(s, np.nan)
            add(s, "Promoter holding", "NSE shareholding pattern", f"{p1:.2f}% -> {pn:.2f}% ({chg:+.2f}pp)",
                "Fall of 1pp or more over about one year (a fall can also be dilution)", ("NSE shareholding pattern via dalal" if rec.get("promoter_holding_source") != "typed" else "Typed by you in the app (not fetched from NSE; not independently verified)"), "",
                rec.get("promoter_holding_asof", ""), str(rec.get("fetched_at", ""))[:10],
                (f"F2 scored {sc_:g}" if pd.notna(sc_) else "F2 scored") + soft)

    # 4. cost-basis anomalies in the holdings file
    for s, r in HD.iterrows():
        if r.get("CostFlag"):
            add(s, "Cost basis", "Your holdings file", f"average cost {r.AvgCost:,.2f} vs price {r.CMP:,.2f} ({r.PnLPct:+,.0f}%)",
                "average cost below Rs 5, gain above 1000% or loss beyond 90% (CFG ANOM_*)", "holdings.csv (supplied by you) and the Yahoo price", "", "",
                today_, "P&L treated as unreconciled; the summary shows P&L with and without these lines")

    # 5. concentration
    cap = CFG["CONC_SINGLE_PCT"]
    for s, r in HD[(HD.AssetType == "equity") & (HD.Weight > cap)].iterrows():
        excess = (r.Weight - cap) / 100.0 * total_value
        add(s, "Concentration", "Portfolio weights", f"{r.Weight:.1f}% of the portfolio", f"above {cap:.0f}% (CFG CONC_SINGLE_PCT, a model setting)",
            "holdings.csv x Yahoo prices", "", "", today_, f"Risk alert; reaching the limit would mean selling about Rs {excess / 1e5:.1f}L (illustrative)")

    REG = pd.DataFrame(rows, columns=cols)
    if verified is not None and len(verified) and len(REG):
        v = verified.copy()
        v.columns = [str(c).strip().lower() for c in v.columns]
        if {"symbol", "flag_type", "verified_on"} <= set(v.columns):
            for _, vr in v.iterrows():
                m = (REG.Symbol.str.upper() == str(vr["symbol"]).strip().upper()) & (REG.FlagType.str.lower() == str(vr["flag_type"]).strip().lower())
                note = str(vr.get("note", "")) if pd.notna(vr.get("note", np.nan)) else ""
                REG.loc[m, "Status"] = f"Verified {vr['verified_on']}" + (f": {note}" if note else "")
    return REG
