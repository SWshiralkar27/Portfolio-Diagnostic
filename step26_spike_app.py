"""step26_spike_app.py - Step 26: can THIS host (Streamlit's server) reach Yahoo Finance and NSE?

A throw-away test page. No uploads, no portfolio data, nothing stored. It fetches a few public symbols, shows what worked, how
long it took and the exact error shapes, and offers a plain-text report to paste into the chat. It tests the libraries directly
AND the app's own provider code (pdiag), so the answer is about the real design, not a toy.
Decision support only; not investment advice.
"""
from __future__ import annotations
import datetime as dt
import importlib.metadata as md
import platform
import re
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 26 hosting test", layout="centered")

IMPORT_ERR = ""
try:
    from pdiag.providers import YahooProvider, NSEProvider, CompositeProvider
    from pdiag.evidence import fetch_fresh_evidence
except Exception as e:                                   # the pdiag folder was not uploaded, or a library is missing
    IMPORT_ERR = f"{type(e).__name__}: {e}"
    NSEProvider = None

NSE_BASE = getattr(NSEProvider, "BASE", "https://www.nseindia.com")
NSE_HEADERS = getattr(NSEProvider, "HEADERS", {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36",
                                               "Referer": "https://www.nseindia.com/"})
SYMBOL_OK = re.compile(r"^[A-Z0-9&\-]{1,15}$")


def timed(fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        return True, out, round(time.perf_counter() - t0, 2), ""
    except Exception as e:
        return False, None, round(time.perf_counter() - t0, 2), f"{type(e).__name__}: {str(e)[:180]}"


def add(rows, test, sym, ok, secs, detail):
    rows.append({"Test": test, "Symbol": sym, "OK": bool(ok), "Seconds": secs, "Detail": detail})


def has_len(x):
    try:
        return x is not None and len(x) > 0
    except Exception:
        return False


def test_symbol(sym, rows):
    import yfinance as yf
    import dalal
    t = yf.Ticker(sym + ".NS")
    ok, h, s, err = timed(lambda: t.history(period="1mo"))
    add(rows, "Yahoo prices (yfinance)", sym, ok and has_len(h), s, f"{len(h)} daily rows" if ok and has_len(h) else (err or "empty result"))
    ok, d, s, err = timed(lambda: dict(t.info or {}))
    add(rows, "Yahoo company info (yfinance)", sym, ok and has_len(d), s,
        f"{len(d)} fields; marketCap {'yes' if d.get('marketCap') else 'no'}; trailingPE {'yes' if d.get('trailingPE') else 'no'}" if ok and has_len(d) else (err or "empty result"))
    ok, st3, s, err = timed(lambda: (t.financials, t.balance_sheet, t.cashflow))
    add(rows, "Yahoo statements (yfinance)", sym, ok and all(has_len(x) for x in st3), s,
        "financials %s, balance sheet %s, cash flow %s" % tuple(str(getattr(x, "shape", "?")) for x in st3) if ok else err)
    ok, r, s, err = timed(lambda: dalal.shareholding(sym))
    add(rows, "NSE shareholding (dalal)", sym, ok and has_len(r), s,
        f"{len(r)} quarters; latest {r[0].get('date')}; promoter field {'yes' if 'pr_and_prgrp' in r[0] else 'MISSING'}" if ok and has_len(r) else (err or "empty result"))
    ok, r, s, err = timed(lambda: dalal.announcements(sym))
    add(rows, "NSE announcements (dalal)", sym, ok and has_len(r), s,
        f"{len(r)} rows; newest {r[0].get('an_dt')}; fields {sorted(r[0])[:6]}" if ok and has_len(r) else (err or "empty result"))


def test_nse_direct(sym, rows):
    import requests

    def go():
        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        a = s.get(NSE_BASE, timeout=10)
        b = s.get(f"{NSE_BASE}/api/quote-equity", params={"symbol": sym}, timeout=10)
        return a.status_code, b.status_code, len(b.content)
    ok, r, s, err = timed(go)
    add(rows, "NSE site direct (requests)", sym, ok and r[1] == 200, s, f"home page HTTP {r[0]}; quote API HTTP {r[1]}; {r[2]} bytes" if ok else err)


def test_app_code(sym, rows):
    """The app's own provider and evidence code, exactly as the real app calls them."""
    if IMPORT_ERR:
        add(rows, "App code (pdiag) imports", sym, False, 0, IMPORT_ERR + "  (upload the pdiag folder to GitHub)")
        return
    tmp = Path(tempfile.mkdtemp(prefix="step26_"))
    tkr = sym + ".NS"
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    cp = CompositeProvider(tmp, use_cache=False)
    ok, h, s, err = timed(lambda: cp.history(tkr, start, end))
    add(rows, "App: 6-year price history", sym, ok and has_len(h), s, f"{len(h)} rows, last {h.index[-1].date()}" if ok and has_len(h) else (err or "None returned (errors are swallowed after 3 tries)"))
    ok, i, s, err = timed(lambda: cp.info(tkr))
    add(rows, "App: company info", sym, ok and has_len(i), s, f"{len(i)} fields" if ok and has_len(i) else (err or "empty"))
    ok, stt, s, err = timed(lambda: cp.statements(tkr))
    full = ok and has_len(stt) and all(has_len(v) for v in stt.values())
    add(rows, "App: annual statements", sym, full, s,
        "tables " + ", ".join(f"{k} {getattr(v, 'shape', '?')}" for k, v in sorted(stt.items())) if ok and has_len(stt) else (err or "empty"))
    ok, sv, s, err = timed(lambda: cp.surveillance(sym))
    add(rows, "App: NSE surveillance lists (ASM/GSM)", sym, ok and cp.nse.available, s, f"flag text '{sv}' (empty = not listed); NSE reachable: {cp.nse.available}" if ok else err)
    ok, ev, s, err = timed(lambda: fetch_fresh_evidence([sym]))
    rec = (ev or {}).get(sym, {})
    got = bool(rec) and str(rec.get("fetch_notes")) == "ok"
    add(rows, "App: evidence fetch (shareholding, announcements, results date)", sym, ok and got, s,
        f"promoter {rec.get('promoter_holding_now')}%; notes: {rec.get('fetch_notes')}" if ok else err)


def versions():
    out = {}
    for p in ("streamlit", "pandas", "yfinance", "dalal", "requests"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "not installed"
    return out


def build_report(rows, symbols, secs_total, run_no):
    df = pd.DataFrame(rows)
    v = versions()
    lines = ["STEP 26 HOSTING TEST REPORT", f"UTC time: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}   run number in this session: {run_no}",
             f"Python {sys.version.split()[0]} on {platform.platform()}   (dalal needs 3.11 or newer: {'OK' if sys.version_info >= (3, 11) else 'TOO OLD'})",
             "Libraries: " + ", ".join(f"{k} {x}" for k, x in v.items()), f"Symbols: {', '.join(symbols)}   total time {secs_total:.0f}s", ""]
    for grp, sel in df.groupby("Test", sort=False):
        lines.append(f"{grp}: {int(sel.OK.sum())}/{len(sel)} OK")
    lines.append("")
    for r in rows:
        lines.append(f"[{'OK ' if r['OK'] else 'FAIL'}] {r['Test']} | {r['Symbol']} | {r['Seconds']}s | {r['Detail']}")
    return "\n".join(lines)


def what_it_means(rows):
    def share(prefix):
        sel = [r for r in rows if r["Test"].startswith(prefix)]
        return (sum(r["OK"] for r in sel) / len(sel)) if sel else None
    y, n_sh, n_an = share("Yahoo prices"), share("NSE shareholding"), share("NSE announcements")
    app = share("App: 6-year price")
    out = []
    if y == 1 and app == 1:
        out.append("Yahoo prices work from this host, both directly and through the app's own code.")
    elif y is not None and y < 1:
        out.append("Yahoo prices failed for some or all symbols. Shared hosting addresses are sometimes throttled in bursts: wait a few minutes and run once more before drawing a conclusion. If it keeps failing, Yahoo is not usable from this host and another host would have to be considered (a decision for you).")
    if n_sh == 1 and n_an == 1:
        out.append("NSE data (shareholding and announcements) came back. The live evidence refresh can run inside the app.")
    elif n_sh is not None and (n_sh < 1 or (n_an is not None and n_an < 1)):
        out.append("NSE data did not come back reliably. The app can still run: the evidence cache (evidence_auto.csv) is already part of the carry-over pack, so it can be refreshed on your own PC and uploaded. Do not build a live NSE fetch into the app until this passes.")
    if not out:
        out.append("Not enough results to say. Run the test again.")
    return out


st.title("Step 26: hosting test")
st.caption("Checks whether this server can fetch Yahoo Finance and NSE data. No uploads, no private data, nothing is stored. Decision support only; not investment advice.")
if IMPORT_ERR:
    st.warning(f"The app's own code could not be loaded ({IMPORT_ERR}). The direct tests still run; upload the pdiag folder to GitHub for the full test.")

text = st.text_input("NSE symbols (up to 5, comma-separated)", "RELIANCE, TCS, HDFCBANK")
clicked = st.button("Run the test")
st.caption("It makes a few dozen requests and can take a minute or two. If you run it again straight away and it fails, that is itself useful to know: tell me.")

if clicked:
    syms = [x.strip().upper() for x in text.split(",") if x.strip()][:5]
    bad = [x for x in syms if not SYMBOL_OK.match(x)]
    if not syms or bad:
        st.error("Use NSE symbols such as RELIANCE or TCS (letters, digits, & and -).")
    else:
        st.session_state["runs"] = st.session_state.get("runs", 0) + 1
        rows, t0 = [], time.perf_counter()
        prog = st.progress(0.0, text="Starting ...")
        steps = len(syms) + 2
        for i, sym in enumerate(syms):
            prog.progress(i / steps, text=f"Fetching {sym} ...")
            try:
                test_symbol(sym, rows)
            except Exception as e:                       # a missing library must show up as a row, not a crash
                add(rows, "Library load", sym, False, 0, f"{type(e).__name__}: {str(e)[:180]}")
            time.sleep(0.5)
        prog.progress(len(syms) / steps, text="Testing the NSE site directly ...")
        test_nse_direct(syms[0], rows)
        prog.progress((len(syms) + 1) / steps, text=f"Testing the app's own code on {syms[0]} ...")
        test_app_code(syms[0], rows)
        prog.progress(1.0, text="Done")
        df = pd.DataFrame(rows)
        st.subheader("Result")
        for msg in what_it_means(rows):
            st.info(msg)
        st.dataframe(df, hide_index=True)
        report = build_report(rows, syms, time.perf_counter() - t0, st.session_state["runs"])
        st.subheader("Report to paste into the chat")
        st.code(report, language="text")
        st.download_button("Download the report", report, file_name="step26_report.txt")
