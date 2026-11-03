#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_diagnostic.py - Indian listed-equity portfolio diagnostic (V4 Integrated)
Auto-fetches evidence data on every run, calculates V3 factors, and outputs V2 rich HTML.
"""
from __future__ import annotations
import argparse, datetime as dt, math, sys, time, warnings, zlib, webbrowser, html as _html
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# --- V3 Custom Engines ---
try:
    import scoring
    import earnings_engine
    import ownership_engine
    import catalyst_engine
    import leadership_engine
    FACTORS = scoring.FACTORS
    WEIGHTS = scoring.WEIGHTS
    FACTOR_NAMES = scoring.FACTOR_NAMES
except ImportError as e:
    sys.exit(f"Missing engine file: {e}. Ensure scoring.py, earnings_engine.py, etc. are in the folder.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import dalal
    HAS_DALAL = True
except ImportError:
    HAS_DALAL = False
try:
    import input_assistant          # optional pop-up for buy dates / promoter holding / peers (see input_assistant.py)
except ImportError:
    input_assistant = None
if sys.version_info < (3, 11):
    sys.exit("ERROR: Python 3.11 or newer is required for the dalal scraper. Please upgrade Python.")

VERSION = "4.0"

# =============================================================================
# CONFIG
# =============================================================================
CFG = dict(
    RISK_FREE=0.06, HISTORY_YEARS=6, RISK_WINDOW_YEARS=3, CORR_WINDOW_YEARS=2,
    PARTICIPATION=(0.10, 0.20), EXIT_DAYS_FLAG=10, REBAL_BAND_PP=2.0,
    CONC_SINGLE_PCT=10.0, CONC_SECTOR_PCT=30.0, CONC_TOP5_PCT=50.0, LTCG_SOON_DAYS=90,
    TAX_STCG=0.20, TAX_LTCG=0.125, LTCG_EXEMPTION=125000.0, CESS=0.04,
    CAP_LARGE_CR=100000.0, CAP_MID_CR=30000.0,
    ANOM_MIN_AVGCOST=5.0, ANOM_MAX_GAIN_PCT=1000.0, ANOM_MIN_PNL_PCT=-90.0,
    OVERWEIGHT_MAX=2.2, UNDERWEIGHT_MIN=3.2, MIN_COVERAGE=0.50,
    BENCH_DIV_YIELD=dict(NIFTY500=0.012, LARGE=0.013, MID=0.008, SMALL=0.008, DEFAULT=0.012),
)
BENCH_CANDIDATES = dict(
    NIFTY500=["^CRSLDX", "^NSEI"], LARGE=["^CNX100", "^NSEI"],
    MID=["NIFTYMIDCAP150.NS", "^NSEMDCP50", "MID150BEES.NS"],
    SMALL=["NIFTYSMLCAP250.NS", "^CNXSC", "HDFCSML250.NS"],
)
CAP_TO_BENCH = dict(large="LARGE", mid="MID", small="SMALL")
FIN_TYPES = {"bank", "nbfc", "hfc", "insurer", "financial"}
NO_STD_TYPES = {"reit", "holdco"}
CONF_RANK = dict(Low=0, Medium=1, High=2)
HOLD_ATTRS = ["sector", "sector_type", "cap_tier", "target_weight_pct", "asset_type", "benchmark", "surveillance", "yahoo_ticker", "horizon", "xirr"]


# =============================================================================
# HELPERS & PROVIDERS
# =============================================================================
def fnum(x, d=np.nan):
    try: return float(x) if np.isfinite(float(x)) else d
    except: return d
def has(x) -> bool: return x is not None and str(x).strip().lower() not in ("", "nan", "none", "nat")
def yes(x) -> bool: return str(x).strip().lower() in ("y", "yes", "true", "1", "t")
def bucket(v, cuts, scores):
    for c, s in zip(cuts, scores):
        if v < c: return s
    return scores[-1]
def pct(x, d=1): return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"
def to_ticker(sym) -> str:
    s = str(sym).strip().upper()
    return s if (s.startswith("^") or s.endswith(".NS") or s.endswith(".BO")) else s + ".NS"
def bare(sym) -> str: return str(sym).strip().upper().replace(".NS", "").replace(".BO", "")
def hist_years(s: pd.Series) -> float: return (s.index[-1] - s.index[0]).days / 365.25 if len(s) > 1 else 0.0
def cagr(s: pd.Series, years: int) -> float:
    s = s.dropna()
    if len(s) < 2: return np.nan
    end = s.index[-1]; target = end - pd.DateOffset(years=years)
    if s.index[0] > target + pd.Timedelta(days=10): return np.nan
    p0 = s[:target].iloc[-1] if (s.index <= target).any() else s.iloc[0]
    return (s.iloc[-1] / p0) ** (1 / years) - 1 if p0 > 0 else np.nan
def total_return_since(s: pd.Series) -> float:
    s = s.dropna(); return s.iloc[-1] / s.iloc[0] - 1 if len(s) > 1 else np.nan
def approx_tr(price: pd.Series, y: float) -> pd.Series:
    t = (price.index - price.index[0]).days.values / 365.25
    return price * np.exp(y * t)
def max_drawdown(s: pd.Series) -> float:
    s = s.dropna(); return float((s / s.cummax() - 1).min()) if len(s) else np.nan
def monthly(s: pd.Series) -> pd.Series: return s.groupby(s.index.to_period("M")).last()
def xirr(flows):
    fl = [(pd.Timestamp(d), float(a)) for d, a in flows if pd.notna(d) and a and np.isfinite(a)]
    if len(fl) < 2 or not (any(a < 0 for _, a in fl) and any(a > 0 for _, a in fl)): return np.nan
    t0 = min(d for d, _ in fl); t = np.array([(d - t0).days / 365.0 for d, _ in fl]); a = np.array([x for _, x in fl])
    f = lambda r: float(np.sum(a / (1 + r) ** t))
    lo, hi = -0.95, 50.0; flo, fhi = f(lo), f(hi)
    if flo * fhi > 0: return np.nan
    mid = np.nan
    for _ in range(200):
        mid = (lo + hi) / 2; fm = f(mid)
        if abs(fm) < 1e-7: break
        if flo * fm <= 0: hi = mid
        else: lo, flo = mid, fm
    return mid

def capm(rs: pd.Series, rm: pd.Series, rf: float, min_obs=120) -> dict:
    df = pd.concat([rs, rm], axis=1, join="inner").dropna()
    if len(df) < min_obs: return {}
    rf_d = (1 + rf) ** (1 / 252) - 1; y, x = df.iloc[:, 0].values - rf_d, df.iloc[:, 1].values - rf_d
    xm, ym = x.mean(), y.mean(); sxx = ((x - xm) ** 2).sum(); beta = ((x - xm) * (y - ym)).sum() / sxx; a = ym - beta * xm
    resid = y - (a + beta * x); n = len(df); s2 = (resid ** 2).sum() / (n - 2); se_a = math.sqrt(s2 * (1 / n + xm ** 2 / sxx))
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return dict(beta=beta, alpha_ann=a * 252, alpha_t=a / se_a if se_a > 0 else np.nan, r2=r2, idio_vol=resid.std(ddof=1) * math.sqrt(252))

def capture(port_wealth: pd.Series, bench_wealth: pd.Series):
    df = pd.concat([monthly(port_wealth).pct_change(), monthly(bench_wealth).pct_change()], axis=1, join="inner").dropna()
    df.columns = ["p", "b"]; up, dn = df[df.b > 0], df[df.b < 0]
    return (up.p.mean() / up.b.mean() if len(up) >= 6 else np.nan, dn.p.mean() / dn.b.mean() if len(dn) >= 6 else np.nan)

def clean_hist(h):
    if h is None or len(h) == 0: return None
    h = h.copy(); idx = pd.to_datetime(h.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    h.index = idx.normalize(); h = h[~h.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame(index=h.index)
    out["close"] = h["Close"]; out["adj"] = h["Adj Close"] if "Adj Close" in h else h["Close"]
    out["high"] = h["High"] if "High" in h else h["Close"]; out["low"] = h["Low"] if "Low" in h else h["Close"]
    out["volume"] = h["Volume"] if "Volume" in h else 0.0
    out["div"] = h["Dividends"] if "Dividends" in h else 0.0
    out["split"] = h["Stock Splits"] if "Stock Splits" in h else 0.0
    out = out.dropna(subset=["close"])
    return out if len(out) else None

class YahooProvider:
    name = "Yahoo Finance"
    def __init__(self, cache_dir: Path, use_cache=True):
        try: import yfinance as yf
        except ImportError: sys.exit("yfinance not installed.")
        self.yf = yf; self.cache = cache_dir; self.cache.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache; self.day = dt.date.today().isoformat()
    def _cached(self, kind, tkr, fn):
        p = self.cache / f"{kind}_{tkr.replace('^','IDX_').replace('/','_')}_{self.day}.pkl"
        if self.use_cache and p.exists():
            try: return pd.read_pickle(p)
            except: pass
        obj = None
        for k in range(3):
            try: obj = fn(); break
            except: time.sleep(1.5 * (k + 1))
        if obj is not None and self.use_cache:
            try: pd.to_pickle(obj, p)
            except: pass
        return obj
    def history(self, tkr, start, end):
        h = self._cached("hist", tkr, lambda: self.yf.Ticker(tkr).history(start=start, end=end, auto_adjust=False, actions=True))
        return clean_hist(h)
    def info(self, tkr): return self._cached("info", tkr, lambda: dict(self.yf.Ticker(tkr).info)) or {}
    def statements(self, tkr):
        def fn():
            t = self.yf.Ticker(tkr); return dict(fin=t.financials, bs=t.balance_sheet, cf=t.cashflow)
        return self._cached("stmt", tkr, fn) or {}

class NSEProvider:
    BASE = "https://www.nseindia.com"
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36", "Referer": "https://www.nseindia.com/"}
    def __init__(self, timeout=8):
        self.available = HAS_REQUESTS; self.timeout = timeout
        if not self.available: return
        self.session = requests.Session(); self.session.headers.update(self.HEADERS)
        self._primed = False; self._asm = None; self._gsm = None
    def _prime(self):
        if self._primed or not self.available: return self.available
        try: self.session.get(self.BASE, timeout=self.timeout); self._primed = True
        except: self.available = False
        return self.available
    def _get(self, path, **params):
        if not self._prime(): return None
        try:
            r = self.session.get(f"{self.BASE}{path}", params=params, timeout=self.timeout)
            if r.status_code == 200: return r.json()
        except: pass
        return None
    def quote(self, symbol): return self._get("/api/quote-equity", symbol=bare(symbol)) or {}
    def surveillance(self, symbol):
        if not self.available: return ""
        if self._asm is None:
            d = self._get("/api/reportASM", index="equities") or {}
            self._asm = {str(row.get("symbol", "")).upper() for row in d.get("data", [])}
        if self._gsm is None:
            d = self._get("/api/reportGSM", index="equities") or {}
            self._gsm = {str(row.get("symbol", "")).upper() for row in d.get("data", [])}
        s = bare(symbol); flags = []
        if s in self._asm: flags.append("ASM")
        if s in self._gsm: flags.append("GSM")
        return " / ".join(flags)
    def corporate_actions(self, symbol):
        d = self._get("/api/corporates-corporateActions", index="equities", symbol=bare(symbol))
        return d if isinstance(d, list) else d.get("data", []) if isinstance(d, dict) else []

class CompositeProvider:
    name = "Yahoo + NSE"
    def __init__(self, cache_dir: Path, use_cache=True):
        self.yahoo = YahooProvider(cache_dir, use_cache); self.nse = NSEProvider()
    def history(self, tkr, start, end): return self.yahoo.history(tkr, start, end)
    def info(self, tkr):
        info = dict(self.yahoo.info(tkr))
        if self.nse.available:
            q = self.nse.quote(tkr)
            if q:
                price = q.get("priceInfo", {}) or {}
                if price.get("lastPrice") and not info.get("regularMarketPrice"): info["regularMarketPrice"] = price["lastPrice"]
                meta = q.get("metadata", {}) or {}
                if meta.get("industry") and not info.get("sector"): info["sector"] = meta["industry"]
        return info
    def statements(self, tkr): return self.yahoo.statements(tkr)
    def surveillance(self, symbol):
        try: return self.nse.surveillance(symbol) if self.nse.available else ""
        except: return ""
    def corporate_actions(self, symbol):
        try: return self.nse.corporate_actions(symbol) if self.nse.available else []
        except: return []

# =============================================================================
# AUTOMATED EVIDENCE FETCHER  (NSE announcements + shareholding via dalal, earnings date via Yahoo)
# Field names below were checked against real dalal 0.2.1 responses:
#   shareholding  -> [{"date": "30-SEP-2026", "pr_and_prgrp": "50.44", "public_val": "49.56", ...}, ...]   (no FII/DII split)
#   announcements -> [{"an_dt": "24-Oct-2026 21:48:04", "desc": "General Updates", "attchmntText": "...", ...}, ...]
# =============================================================================
EVIDENCE_MAX_AGE_DAYS = 7
EVIDENCE_SCHEMA = 5          # bump when the classification rules change: older rows are then re-fetched automatically
# Fields owned by the fetcher. They live in evidence_auto.csv and are never read back from your evidence.csv,
# except the two promoter fields, which you may type in by hand when NSE has nothing for a stock.
AUTO_FIELDS = ("promoter_holding_now", "promoter_holding_1y", "promoter_holding_asof", "next_earnings",
               "ex_date", "announcements", "ann_watch", "flag_log", "fetched_at", "fetch_notes", "schema")
MANUAL_FALLBACK = ("promoter_holding_now", "promoter_holding_1y")


def evidence_is_stale(rec, max_age_days=EVIDENCE_MAX_AGE_DAYS):
    """True when a symbol's fetched evidence is missing, old, or made by older classification rules.
    A failed shareholding fetch is retried after 12 hours."""
    if not rec or not has(rec.get("fetched_at")):
        return True
    try:
        if int(float(rec.get("schema", 0))) != EVIDENCE_SCHEMA:
            return True
        age = (pd.Timestamp.now() - pd.Timestamp(rec["fetched_at"])).total_seconds() / 86400.0
    except Exception:
        return True
    if age > max_age_days:
        return True
    return (not has(rec.get("promoter_holding_now"))) and age > 0.5


def merge_evidence(manual, auto):
    """Your evidence.csv + the fetched evidence_auto.csv. Fetched values win for AUTO_FIELDS; everything else
    (pledge, auditor flag, GNPA/CAR, typed-in FII or sales-growth columns ...) stays exactly as you entered it."""
    out = {}
    for s in set(manual) | set(auto):
        rec = {k: v for k, v in manual.get(s, {}).items() if k not in AUTO_FIELDS or k in MANUAL_FALLBACK}
        for k, v in auto.get(s, {}).items():
            if k != "symbol":
                rec[k] = v
        rec["symbol"] = s
        out[s] = rec
    return out


def classify_disclosure(text):
    """Classify one NSE announcement. Returns (kind, label, matched_phrase); kind is 'red', 'down', 'watch' or None.
    RED = may lower F4. WATCH = shown as a risk tag only. Precision matters more than recall here:
    routine boilerplate ('Regulation 30 of SEBI LODR ...') must never trigger a flag."""
    import re
    t = " ".join(str(text).split()).lower()
    red = [
        ("auditor resignation", r"(statutory|joint)( statutory)? auditors?\b.{0,80}\bresign|\bresign\w*.{0,80}(statutory|joint)( statutory)? auditors?\b|\bresign\w*.{0,60}\bas (the )?auditors? of the company"),
        ("audit qualification", r"qualified (opinion|conclusion)|adverse opinion|disclaimer of opinion"),
        ("debt default or insolvency", r"\bdefault(ed)? (in|on) (the )?(payment|repayment|servicing|interest|principal|obligation|loan|debt|ncd|debenture)|\bdefaulted\b|delay in (payment|servicing|repayment)|invocation of (pledge|guarantee)|corporate insolvency resolution|\bcirp\b|insolvency (petition|proceedings?|application)|winding[- ]up petition"),
        ("SEBI action or fraud", r"(order|notice|penalty|fine|direction)s? (passed |issued |imposed |levied )?(by|from) (the )?(securities and exchange board of india|\bsebi\b)|\bsebi\b (has |had )?(imposed|levied|debarred|barred|restrained)|forensic audit|alleged fraud|fraud (detected|reported|identified|case)|search and seizure|enforcement directorate"),
    ]
    kmp = r"(chief executive|\bceo\b|chief financial|\bcfo\b|managing director|whole[- ]time director|chairman|company secretary|compliance officer)"
    watch = [
        ("key manager resignation", r"\bresign\w*.{0,80}" + kmp + "|" + kmp + r".{0,80}\bresign"),
        ("other auditor role change", r"\b(internal|secretarial|cost|tax) auditors?\b.{0,60}\bresign|\bresign\w*.{0,60}\b(internal|secretarial|cost|tax) auditors?\b"),
        ("tax, adjudication or penalty order", r"adjudication order|demand (notice|order)|show cause notice|assessment order|\bpenalty\b.{0,60}(imposed|levied)|fine (of|imposed)"),
    ]
    for name, pat in red:
        m = re.search(pat, t)
        if m:
            if name in ("debt default or insolvency", "auditor resignation") and "subsidiar" in t:
                return "watch", "subsidiary insolvency, default or auditor change", m.group(0)
            return "red", name, m.group(0)
    m = re.search(r"(credit )?rating.{0,80}downgrad|downgrad.{0,80}(credit )?rating", t)
    if m:
        return "down", "credit rating downgrade", m.group(0)
    for name, pat in watch:
        m = re.search(pat, t)
        if m:
            return "watch", name, m.group(0)
    return None, None, None


def fetch_fresh_evidence(symbols):
    """Fetch promoter holding (now and ~1 year ago), recent disclosures and the next results date per symbol."""
    if not HAS_DALAL:
        print("WARNING: 'dalal' library not installed. Skipping auto-fetch. Run: pip install dalal")
        return {}

    def safe_float(val):
        if val is None or str(val).strip() in ("", "-", "nan"):
            return np.nan
        try:
            return float(str(val).replace("%", "").strip())
        except Exception:
            return np.nan

    def call(fn, *a):
        err = ""
        for attempt in range(3):
            try:
                return fn(*a), ""
            except Exception as e:
                err = type(e).__name__
                if "RateLimit" in err and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                break
        return None, err

    import json
    print(f"\nFetching NSE evidence for {len(symbols)} symbols (shareholding, disclosures, results date) ...")
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=90)
    results = {}
    for i, sym in enumerate(symbols):
        print(f"[{i + 1}/{len(symbols)}] {sym}            ", end="\r")
        notes = []

        # 1. Promoter holding: latest quarter and the quarter closest to one year earlier
        prom_now, prom_1y, asof = np.nan, np.nan, ""
        raw, err = call(dalal.shareholding, sym)
        if raw is None:
            notes.append(f"shareholding: {err}")
        else:
            recs = []
            for r in raw:
                d = pd.to_datetime(r.get("date"), format="%d-%b-%Y", errors="coerce")
                p = safe_float(r.get("pr_and_prgrp"))
                if pd.notna(d) and not np.isnan(p):
                    recs.append((d, p))
            recs.sort(key=lambda x: x[0], reverse=True)
            if recs:
                d0, prom_now = recs[0]
                asof = d0.strftime("%Y-%m-%d")
                target = d0 - pd.Timedelta(days=365)
                near = min(recs[1:], key=lambda x: abs((x[0] - target).days), default=None)
                if near is not None and abs((near[0] - target).days) <= 60:
                    prom_1y = near[1]
                else:
                    notes.append("shareholding: no quarter close to 1 year back")
            else:
                notes.append("shareholding: no usable rows")

        # 2. Disclosures in the last 90 days
        ann_red, ann_watch, flag_log = [], [], []
        raw, err = call(dalal.announcements, sym)
        if raw is None:
            notes.append(f"announcements: {err}")
        else:
            seen = set()
            for a in raw[:400]:
                d = pd.to_datetime(a.get("an_dt"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
                if pd.isna(d) or d < cutoff:
                    continue
                kind, name, matched = classify_disclosure(f"{a.get('desc', '')} {a.get('attchmntText', '')}")
                if not kind or (kind, name, d.date()) in seen:
                    continue
                seen.add((kind, name, d.date()))
                snippet = " ".join(str(a.get("attchmntText") or a.get("desc") or "").split()).replace("|", "/")[:150]
                item = f"{name} ({d:%d-%b}) [matched: '{matched[:40]}']: {snippet}"
                if len(flag_log) < 6: flag_log.append(dict(kind=kind, label=name, matched=matched[:60], date=d.strftime("%Y-%m-%d"), url=str(a.get("attchmntFile") or ""), text=snippet))
                if kind == "red" and len(ann_red) < 3:
                    ann_red.append("RED_FLAG: " + item)
                elif kind == "down" and len(ann_red) < 3:
                    ann_red.append("downgrade: " + item)
                elif kind == "watch" and len(ann_watch) < 3:
                    ann_watch.append(item)

        # 3. Next results date (Yahoo calendar)
        next_earnings = ""
        try:
            import yfinance as yf
            cal = yf.Ticker(sym + ".NS").calendar
            if cal and "Earnings Date" in cal and cal["Earnings Date"]:
                next_earnings = str(cal["Earnings Date"][0])
        except Exception as e:
            notes.append(f"results date: {type(e).__name__}")

        results[sym] = {
            "symbol": sym,
            "promoter_holding_now": prom_now, "promoter_holding_1y": prom_1y, "promoter_holding_asof": asof,
            "next_earnings": next_earnings,
            "announcements": " | ".join(ann_red), "ann_watch": " | ".join(ann_watch), "flag_log": json.dumps(flag_log),
            "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "fetch_notes": "; ".join(notes) if notes else "ok",
            "schema": EVIDENCE_SCHEMA,
        }

    n_sh = sum(1 for r in results.values() if not np.isnan(r["promoter_holding_now"]))
    n_red = sum(1 for r in results.values() if r["announcements"])
    print(f"\nFinished: promoter holding found for {n_sh}/{len(results)} symbols; adverse disclosures for {n_red}.")
    return results

# =============================================================================
# FUNDAMENTALS EXTRACTION
# =============================================================================
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

# =============================================================================
# INPUT LOADERS
# =============================================================================
def read_table(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".xlsx", ".xls")): df = pd.read_excel(path)
    else:
        try: df = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)
        except: df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [str(c).replace("\ufeff", "").strip().lower().replace(" ", "_") for c in df.columns]
    return df

def load_holdings(path):
    df = read_table(path)
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
    if "symbol" not in df.columns: sys.exit("Missing 'symbol' column.")
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

def load_evidence(path):
    if not path or not Path(path).exists(): return {}
    df = read_table(path); df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["symbol"])
    return {r["symbol"]: {k: v for k, v in r.items() if has(v)} for r in df.to_dict("records")}

def load_peers(path):
    if not path or not Path(path).exists(): return {}
    df = read_table(path); df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["symbol"])
    return df.groupby("symbol")["peer"].apply(lambda s: [str(x).strip().upper() for x in s]).to_dict()

# =============================================================================
# SCORING ENGINE
# =============================================================================
def score_stock(sector_type, ev, fund, perf, val_ratio, n_peers, val_label):
    S = {k: None for k in FACTORS}
    E, proxies, overrides, bull, bear = [], [], [], [], []
    data_rs = []

    try:
        s1, r1 = earnings_engine.score_earnings(ev, fund)
        S["F1"] = s1
        if s1 is not None:
            E.append(("F1", s1, "earnings_engine", r1))
        else:
            data_rs.append(f"F1 Unscored: {r1}")
    except Exception as e:
        data_rs.append(f"F1 Engine Error: {e}")

    try:
        s2, r2 = ownership_engine.score_ownership(ev)
        S["F2"] = s2
        if s2 is not None:
            E.append(("F2", s2, "ownership_engine", r2))
        else:
            data_rs.append(f"F2 Unscored: {r2}")
    except Exception as e:
        data_rs.append(f"F2 Engine Error: {e}")

    try:
        s3, r3 = catalyst_engine.score_catalyst(ev)
        S["F3"] = s3
        if s3 is not None:
            E.append(("F3", s3, "catalyst_engine", r3))
        else:
            data_rs.append(f"F3 Unscored: {r3}")
    except Exception as e:
        data_rs.append(f"F3 Engine Error: {e}")

    try:
        s4, r4 = leadership_engine.score_leadership(ev)
        S["F4"] = s4
        if s4 is not None:
            E.append(("F4", s4, "leadership_engine", r4))
        else:
            data_rs.append(f"F4 Unscored: {r4}")
    except Exception as e:
        data_rs.append(f"F4 Engine Error: {e}")
    # F3: Catalysts & Events (Earnings dates, M&A)
    f3_score = None
    f3_reason = ""
    next_earn = str(ev.get("next_earnings", ""))
    ann = str(ev.get("announcements", ""))
    
    if "POSITIVE" in ann:
        f3_score, f3_reason = 1, "M&A or restructuring news"
    elif next_earn:
        try:
            earn_dt = pd.to_datetime(next_earn)
            days_to = (earn_dt.normalize() - pd.Timestamp.today().normalize()).days
            if 0 <= days_to <= 14:
                bear.append(f"Results due in {days_to} days (event risk)")
        except Exception:
            pass
            
    # A red-flag disclosure is scored once, in F4 (not double-counted in F3).
        
    if f3_score is not None:
        S["F3"] = f3_score
        E.append(("F3", f3_score, "events_scraper", f3_reason))

    # F4: Leadership & Management (Auditor resignations, penalties)
    f4_score = None
    f4_reason = ""
    
    if "RED_FLAG" in ann:
        first = ann.split(" | ")[0].replace("RED_FLAG: ", "")
        f4_score, f4_reason = 5, "Adverse disclosure (last 90 days): " + first[:240]
    elif "downgrade" in ann.lower():
        f4_score, f4_reason = 4, "Rating downgrade"
        
    if f4_score is not None:
        S["F4"] = f4_score
        E.append(("F4", f4_score, "news_scraper", f4_reason))
    watch = str(ev.get("ann_watch", ""))
    if watch:
        bear.append("Recent disclosure to check: " + watch.split(" | ")[0][:90])

    if not pd.isna(val_ratio):
        S["F5"] = float(bucket(val_ratio, [0.7, 0.9, 1.1, 1.4], [1, 2, 3, 4, 5]))
        E.append(("F5", S["F5"], f"peer median ({n_peers})", f"{val_label} = {val_ratio:.2f}x"))
    elif pd.notna(fund.get("pe_norm", np.nan)) and pd.notna(fund.get("ni_peak_ratio", np.nan)) and fund["ni_peak_ratio"] >= 1.5:
        pk = fund["ni_peak_ratio"]
        pk_txt = "follows earlier losses" if pk >= 50 else f"is {pk:.1f}x the prior average"
        S["F5"] = float(bucket(fund["pe_norm"], [10, 20, 40, 60], [1, 2, 3, 4, 5]))
        pe_t = fund.get("pe", np.nan)
        E.append(("F5", S["F5"], "normalised P/E", f"P/E on average earnings = {fund['pe_norm']:.1f}x (trailing {pe_t:.1f}x; latest profit {pk_txt})"))
        proxies.append("valuation: normalised P/E for spiky earnings, no peer set")
        bear.append("Earnings far above their prior average (possible cyclical peak)")
    elif "pe" in fund and not pd.isna(fund["pe"]) and fund["pe"] > 0:
        S["F5"] = float(bucket(fund["pe"], [10, 20, 40, 60], [1, 2, 3, 4, 5]))
        E.append(("F5", S["F5"], "absolute fallback", f"P/E = {fund['pe']:.1f}x"))
        proxies.append("valuation: absolute P/E fallback, no peer set")
    elif "pb" in fund and not pd.isna(fund["pb"]) and fund["pb"] > 0:
        S["F5"] = float(bucket(fund["pb"], [1.5, 3, 5, 8], [1, 2, 3, 4, 5]))
        E.append(("F5", S["F5"], "absolute fallback", f"P/B = {fund['pb']:.1f}x"))
        proxies.append("valuation: absolute P/B fallback, no peer set")

    if sector_type in FIN_TYPES:
        subs, txt = [], []
        if not pd.isna(fnum(ev.get("gnpa_pct"))):
            subs.append(bucket(fnum(ev["gnpa_pct"]), [1.5, 3, 5, 8], [1, 2, 3, 4, 5])); txt.append(f"GNPA {ev['gnpa_pct']}%")
        if not pd.isna(fnum(ev.get("car_pct"))):
            subs.append(bucket(fnum(ev["car_pct"]), [11.5, 13, 15, 18], [5, 4, 3, 2, 1])); txt.append(f"CAR {ev['car_pct']}%")
        if subs:
            S["F6"] = float(np.mean(subs)); E.append(("F6", S["F6"], "user-supplied", ", ".join(txt)))
    elif sector_type not in NO_STD_TYPES:
        subs, txt = [], []
        nd, ic, ce = fund.get("nd_ebitda", np.nan), fund.get("int_cov", np.nan), fund.get("cfo_ebitda", np.nan)
        if not pd.isna(fund.get("ebitda_latest", np.nan)) and fund["ebitda_latest"] <= 0:
            subs.append(5); txt.append("EBITDA <= 0")
        elif not pd.isna(nd):
            subs.append(1 if nd < 0 else bucket(nd, [1, 2, 3, 4], [1, 2, 3, 4, 5])); txt.append(f"ND/EBITDA {nd:.1f}x")
        if not pd.isna(ic):
            subs.append(bucket(ic, [1.5, 2, 3, 5], [5, 4, 3, 2, 1])); txt.append(f"interest cover {min(ic, 99):.1f}x")
        if not pd.isna(ce):
            subs.append(bucket(ce, [0.4, 0.6, 0.7, 0.8], [5, 4, 3, 2, 1])); txt.append(f"CFO/EBITDA {ce * 100:.0f}%")
        if subs:
            S["F6"] = max(float(np.mean(subs)), 4.0) if max(subs) == 5 else float(np.mean(subs))
            E.append(("F6", S["F6"], "Yahoo annual", ", ".join(txt)))

    ex = [perf.get(f"excess_{h}y") for h in (3, 5)]
    ex = [x for x in ex if not pd.isna(x)]
    if ex:
        m = float(np.mean(ex))
        S["F7"] = float(bucket(m, [-0.10, -0.02, 0.02, 0.05], [5, 4, 3, 2, 1]))
        E.append(("F7", S["F7"], "computed", f"avg annualised excess {pct(m)}"))

    dd = perf.get("maxdd_3y")
    if not pd.isna(dd):
        bench_dd = perf.get("bench_maxdd_3y", -0.25)
        if dd < bench_dd * 1.5: S["F8"] = 5
        elif dd < bench_dd * 1.2: S["F8"] = 4
        elif dd < bench_dd: S["F8"] = 3
        elif dd < bench_dd * 0.8: S["F8"] = 2
        else: S["F8"] = 1
        E.append(("F8", S["F8"], "computed", f"3y MaxDD {pct(dd)} vs benchmark {pct(bench_dd)}"))

    dte = perf.get("days_to_exit", np.nan)
    # F9 is a flag, not a score: at this portfolio size every stock exits in under a day, so it cannot discriminate.
    if not pd.isna(dte) and dte > 2.0:
        bear.append(f"Illiquid: {dte:.1f} days to exit at 10% of volume")

    # F10 is currently disabled to remove overlap with F7.
    # In the future, this should use sector-index relative strength.
    S["F10"] = None

    pl, rising = fnum(ev.get("promoter_pledge_pct")), yes(ev.get("pledge_rising_2q"))
    aud, cov_ = yes(ev.get("auditor_issue")), yes(ev.get("covenant_or_rating_stress"))
    reg = yes(ev.get("regulatory_action_3y"))
    if (not np.isnan(pl)) and (pl > 25 or (pl > 10 and rising)): overrides.append(f"Promoter pledge {pl}% (rising={rising})")
    if aud: overrides.append("Auditor qualification")
    if cov_: overrides.append("Covenant breach")
    if reg: overrides.append("Adverse regulatory order")
    if fund.get("neg_networth"): overrides.append("Negative net worth")
    if fund.get("eq_flag"): overrides.append(f"Earnings-quality flag: {fund['eq_flag']}")

    a, t = perf.get("alpha_ann", np.nan), perf.get("alpha_t", np.nan)
    if not pd.isna(a):
        (bull if a > 0 and t > 1 else bear if a < 0 and t < -1 else []).append(f"alpha {pct(a)} (t={t:.1f})")
    if not pd.isna(fund.get("nd_ebitda", np.nan)):
        (bull if fund["nd_ebitda"] < 1 else bear if fund["nd_ebitda"] > 3 else []).append(f"ND/EBITDA {fund['nd_ebitda']:.1f}x")
    if perf.get("dma200_gap", 0) < 0: bear.append("below 200-DMA")
    elif perf.get("dma200_gap", 0) > 0: bull.append("above 200-DMA")
    
    return S, E, proxies, overrides, bull, bear, data_rs

# Factors the free data feeds can score today. F3 (catalysts), F4 (adverse events only), F9 (liquidity) and F10 are
# flags or switched off, so they neither count towards coverage nor appear as "missing".
SCORABLE = ("F1", "F2", "F5", "F6", "F7", "F8")

def weighted(S, W):
    av = {k: v for k, v in S.items() if v is not None}
    if not av: return np.nan, 0.0
    tw = sum(W[k] for k in av)
    base = sum(W[k] for k in SCORABLE)
    return sum(W[k] * v for k, v in av.items()) / tw, min(1.0, sum(W[k] for k in av if k in SCORABLE) / base)

def schemes():
    base = WEIGHTS
    return dict(base=base, equal={k: 1 / len(FACTORS) for k in FACTORS}, fwd_heavy={k: (0.40 if k == "F1" else v * 0.60 / (1 - base["F1"])) for k, v in base.items()}, gov_heavy={k: (0.25 if k == "F2" else v * 0.75 / (1 - base["F2"])) for k, v in base.items()})

def classify(score, cov):
    if pd.isna(score) or cov < CFG["MIN_COVERAGE"]: return "Not classified"
    if score <= CFG["OVERWEIGHT_MAX"]: return "Overweight"
    if score >= CFG["UNDERWEIGHT_MIN"]: return "Underweight"
    return "Neutral"

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def build_actions(SCD, HD, liq, total_value, TAX=None, enforce_cap=True):
    """Three separate layers per holding. Nothing here is an order.

      1. Research signal - what the factor model says about the company. Never carries an amount.
      2. Risk alert      - hard-override flags and concentration. A concentration alert shows an ILLUSTRATIVE amount
                           (what selling down to the limit would mean, before tax and costs).
      3. Allocation      - an Add/Trim amount appears ONLY when you supplied a target weight for the holding
                           (target_weight_pct in holdings.csv) and it sits outside the rebalancing band.
                           Without a target the tool says so and shows no amount.
    The single-stock limit (CONC_SINGLE_PCT) is a model setting you can change, not a regulatory rule.
    """
    cap_pct = CFG["CONC_SINGLE_PCT"]
    cap_val = cap_pct / 100.0 * total_value
    band = CFG["REBAL_BAND_PP"]

    tax_state = {}
    if TAX is not None and len(TAX) and "Status" in TAX.columns:
        for sym, g in TAX.groupby("Symbol"):
            st = g["Status"].astype(str)
            if st.str.startswith("STCG").any():
                tax_state[sym] = "Includes STCG lots (held < 1 year) - check lot selection"
            elif st.str.startswith("unknown").any():
                tax_state[sym] = "Holding period unknown (no buy_date) - tax not estimated"
            else:
                tax_state[sym] = "All lots long-term (LTCG)"
    no_tax = "Holding period unknown (no buy_date) - tax not estimated"

    rows = []
    for _, r in SCD.iterrows():
        s = r["Symbol"]
        w = float(r["Weight"])
        val = w / 100.0 * total_value
        label, conf = r["FinalLabel"], str(r.get("Confidence", ""))
        cov = r.get("Coverage", np.nan)
        tgt = HD.at[s, "Target"] if (s in HD.index and "Target" in HD.columns) else np.nan
        tgt_val = tgt / 100.0 * total_value if pd.notna(tgt) else np.nan

        # 1. research signal
        if label == "Candidate for review":
            signal = "Review flag"
        elif label == "Not classified" or conf == "Low" or (pd.notna(cov) and cov < CFG["MIN_COVERAGE"]):
            signal = "Insufficient evidence"
        else:
            signal = {"Overweight": "Favourable", "Underweight": "Unfavourable"}.get(label, "Neutral")
        act, amt, why = "Hold", 0.0, [f"Research signal: {signal}"]
        if "cyclical peak" in str(r.get("Bear_Signals", "")):
            why.append("Caution: earnings are far above their prior average (possible cyclical peak) - treat 'Favourable' with care")

        # 2. hard-override flags
        if label == "Candidate for review":
            act = "Review / Exit"
            flags = [x.replace("HARD OVERRIDE: ", "") for x in str(r.get("Review_Risk", "")).split(" | ") if x.startswith("HARD OVERRIDE")]
            why.append("Verify before any sale: " + "; ".join(flags) if flags else "Hard override raised")
        # 3. allocation: only against a target you supplied
        elif pd.notna(tgt_val):
            gap = tgt_val - val
            if abs(gap) > band / 100.0 * total_value:
                if gap < 0:
                    act, amt = "Trim", -gap
                    why.append(f"Above your {tgt:.1f}% target by more than the {band:.0f}pp band: trim back to target")
                elif signal in ("Favourable", "Neutral"):
                    act, amt = "Add", gap
                    why.append(f"Below your {tgt:.1f}% target by more than the {band:.0f}pp band: add up to target")
                else:
                    why.append(f"Below your {tgt:.1f}% target, but the research signal is {signal.lower()}: no add")
            else:
                why.append(f"Within the {band:.0f}pp band of your {tgt:.1f}% target")
        else:
            why.append("No allocation action: no target weight supplied (target_weight_pct in holdings.csv)")

        # 4. concentration risk alert (independent of the research signal)
        if enforce_cap and val > cap_val + 1:
            excess = val - cap_val
            note = f"Above the {cap_pct:.0f}% single-stock limit ({w:.1f}%) - a model setting, not a regulatory rule"
            if pd.notna(tgt) and tgt > cap_pct:
                why.append(f"Above the {cap_pct:.0f}% model limit ({w:.1f}%), but your own {tgt:.1f}% target takes precedence")
            elif act == "Hold":
                act, amt = "Risk alert", excess
                why.append(note + f". Reaching the limit would mean selling about ₹{excess / 1e5:.1f}L (illustrative; before tax and costs)")
            else:
                if act == "Trim" and excess > amt:
                    amt = excess
                why.append(note)

        tax_note = tax_state.get(s, no_tax) if act in ("Trim", "Review / Exit", "Risk alert") else ""
        liq_note = ""
        if s in liq and liq[s]["DaysToExit_10pct"] > 2.0:
            liq_note = f"Illiquid ({liq[s]['DaysToExit_10pct']:.1f} days to exit)"
        rows.append(dict(Symbol=s, ResearchSignal=signal, Action=act, Amount_INR=amt, Reason="; ".join(why), Tax_Note=tax_note, Liquidity_Note=liq_note))
    return pd.DataFrame(rows).set_index("Symbol")


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
                "Fall of 1pp or more over about one year (a fall can also be dilution)", "NSE shareholding pattern via dalal", "",
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


def run(args):
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    notes_global, gaps = [], []
    
    provider = CompositeProvider(out / ".cache", use_cache=not args.no_cache)
    lots = load_holdings(args.holdings)
    _dir = Path(args.holdings).resolve().parent
    for _a, _f in (("evidence", "evidence.csv"), ("peers", "peers.csv")):
        if not Path(getattr(args, _a)).exists() and (_dir / _f).exists():
            setattr(args, _a, str(_dir / _f))
    if input_assistant and not args.demo:
        input_assistant.check_holdings_layout(args.holdings)
        lots = input_assistant.apply_saved_buy_dates(lots, _dir)
    
    # --- Evidence: your evidence.csv (never rewritten) + evidence_auto.csv (fetched; refreshed per symbol) ---
    manual_evid = load_evidence(args.evidence)
    auto_path = out / "evidence_auto.csv"
    auto_evid = load_evidence(auto_path) if auto_path.exists() else {}
    fetch_syms = (lots.loc[~lots["asset_type"].isin(["cash", "bond"]), "symbol"]
                  .dropna().astype(str).str.strip().str.upper().unique().tolist())
    if args.fetch_evidence and not args.demo:
        force = bool(getattr(args, "refresh_evidence", False))
        stale = [s for s in fetch_syms if force or evidence_is_stale(auto_evid.get(s))]
        if stale:
            print(f"\nEvidence: refreshing {len(stale)} of {len(fetch_syms)} symbols from NSE "
                  f"(the rest are under {EVIDENCE_MAX_AGE_DAYS} days old; use --refresh-evidence to force all)")
            fresh = fetch_fresh_evidence(stale)
            if fresh:
                auto_evid.update({s: {k: v for k, v in rec.items() if has(v)} for s, rec in fresh.items()})
                pd.DataFrame(list(auto_evid.values())).to_csv(auto_path, index=False)
                print(f"Fetched evidence saved to {auto_path} (your {args.evidence} is never modified)")
        else:
            print("\nEvidence: all symbols are fresh - skipping the NSE fetch.")
    evid = merge_evidence(manual_evid, auto_evid)
        
    peers = load_peers(args.peers)
    if input_assistant and not args.demo:
        _mode = "off" if getattr(args, "no_prompt", False) else ("force" if getattr(args, "prompt", False) else "auto")
        lots, evid, peers = input_assistant.run_prompts(
            lots, evid, peers, folder=_dir, mode=_mode,
            evidence_path=args.evidence if Path(args.evidence).exists() else _dir / "evidence.csv",
            peers_path=args.peers if Path(args.peers).exists() else _dir / "peers.csv")
    review_date = pd.Timestamp(args.review_date) if args.review_date else pd.Timestamp.today().normalize()

    attrs = lots.groupby("symbol")[HOLD_ATTRS].first()
    lots["ticker"] = [to_ticker(yt if has(yt) else s) for s, yt in zip(lots.symbol, lots.yahoo_ticker)]
    lots.loc[lots.asset_type.isin(["cash", "bond"]), "ticker"] = ""
    tick = lots.groupby("symbol")["ticker"].first()

    oldest = lots.buy_date.min()
    start = min(pd.Timestamp.today() - pd.DateOffset(years=CFG["HISTORY_YEARS"]), (oldest - pd.Timedelta(days=10)) if pd.notna(oldest) else pd.Timestamp.max)
    start = max(start, pd.Timestamp.today() - pd.DateOffset(years=15)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    H = {}
    print(f"\nFetching price history for {sum(1 for s in tick.index if tick[s])} instruments ...")
    for s in tick.index:
        t = tick[s]
        if not t: continue
        h = provider.history(t, start, end)
        if h is not None and len(h) >= 30: H[s] = h
        else: gaps.append((s, "price history", "not returned"))

    B = {}
    def make_bench(key, cands):
        for c in cands:
            h = provider.history(c, start, end)
            if h is not None and len(h) > 250:
                y = CFG["BENCH_DIV_YIELD"].get(key, 0.012)
                return dict(tr=approx_tr(h["close"], y), px=h["close"], true_tr=False, src=f"{c} (approx. TR)")
        return None
    for k, c in BENCH_CANDIDATES.items(): B[k] = make_bench(k, c)
    broad = B.get("NIFTY500")
    if broad is None: sys.exit("No broad benchmark available.")

    rows = []
    for s, g in lots.groupby("symbol", sort=False):
        a = attrs.loc[s]; qty = g.quantity.sum(); cost = (g.quantity * g.buy_price).sum(); at = a.asset_type
        if at in ("cash", "bond") or s not in H: cmp_, asof = (cost / qty if at in ("cash", "bond") else np.nan), pd.NaT
        else: cmp_, asof = float(H[s]["close"].iloc[-1]), H[s].index[-1]
        auto_surv = ""
        if at == "equity":
            try: auto_surv = provider.surveillance(s)
            except: pass
        rows.append(dict(Symbol=s, Ticker=tick[s], AssetType=at, Qty=qty, AvgCost=cost / qty, Invested=cost, CMP=cmp_, PriceAsOf=asof,
                         Value=qty * cmp_ if not np.isnan(cmp_) else np.nan, Sector=a.sector if has(a.sector) else "", SectorType=str(a.sector_type).lower() if has(a.sector_type) else "",
                         CapTier=a.cap_tier if has(a.cap_tier) else "", Target=fnum(a.target_weight_pct), UserXIRR=fnum(a.xirr), BenchmarkUser=a.benchmark if has(a.benchmark) else "",
                         Surveillance=(str(a.surveillance) if has(a.surveillance) else auto_surv) or ""))


    HD = pd.DataFrame(rows).set_index("Symbol")
    for s in HD.index[HD.Value.isna()]: gaps.append((s, "market value", "no price: left out of value, cost and P&L totals (not shown as zero)"))
    HD["Priced"] = HD.Value.notna()
    
    HD["Value"] = HD["Value"].fillna(0.0)
    fill_mask = HD["AvgCost"].isna() & HD["CMP"].notna()
    if fill_mask.any():
        HD.loc[fill_mask, "AvgCost"] = HD.loc[fill_mask, "CMP"]
        HD.loc[fill_mask, "Invested"] = HD.loc[fill_mask, "Qty"] * HD.loc[fill_mask, "AvgCost"]

    HD["Weight"] = HD.Value / HD.Value.sum() * 100
    HD["PnL"] = (HD.Value - HD.Invested).where(HD.Priced)
    HD["PnLPct"] = HD.PnL / HD.Invested * 100
    _cost_ok = set()      # lines whose cost you confirmed in the cost review (cost_overrides.csv)
    if input_assistant and not args.demo:
        HD, _cost_ok, _cost_factors = input_assistant.review_costs(
            HD, CFG, folder=_dir, mode=("off" if getattr(args, "no_prompt", False) else ("force" if getattr(args, "prompt", False) else "auto")))
        for _s, _f in _cost_factors.items():      # keep lot prices consistent with a corrected average cost (XIRR, tax)
            lots.loc[lots.symbol == _s, "buy_price"] *= _f
    # Cost-basis sanity check on the average costs supplied in holdings.csv. Bonus shares, splits, demergers and merged
    # lots often break an average cost, and one wrong line can dominate the portfolio's headline return.
    cf = []
    for s_, r_ in HD.iterrows():
        why_ = []
        if r_.Priced and r_.AssetType == "equity" and s_ not in _cost_ok:
            if r_.AvgCost < CFG["ANOM_MIN_AVGCOST"]: why_.append(f"average cost below Rs {CFG['ANOM_MIN_AVGCOST']:.0f}")
            if r_.PnLPct > CFG["ANOM_MAX_GAIN_PCT"]: why_.append(f"gain above {CFG['ANOM_MAX_GAIN_PCT']:.0f}%")
            if r_.PnLPct < CFG["ANOM_MIN_PNL_PCT"]: why_.append(f"loss beyond {abs(CFG['ANOM_MIN_PNL_PCT']):.0f}%")
        cf.append("; ".join(why_))
    HD["CostFlag"] = cf
    for s_ in HD.index[HD.CostFlag != ""]:
        gaps.append((s_, "cost basis", f"{HD.loc[s_, 'CostFlag']} (avg cost {HD.loc[s_, 'AvgCost']:,.2f} vs price {HD.loc[s_, 'CMP']:,.2f}): check broker statement and corporate actions; P&L treated as unreconciled"))
    HD["TargetDev_pp"] = HD.Weight - HD.Target
    HD["OutOfBand"] = HD.TargetDev_pp.abs() > CFG["REBAL_BAND_PP"]
    total_value = HD.Value.sum()
    equity_syms = [s for s in HD.index if HD.loc[s, "AssetType"] not in ("cash", "bond")]

    perf, fundd, liq, ca_rows, info_cache = {}, {}, {}, [], {}
    print("Computing returns, risk, liquidity and fundamentals ...")
    bench_win_start = pd.Timestamp.today() - pd.DateOffset(years=CFG["RISK_WINDOW_YEARS"])
    bench_maxdd_3y = max_drawdown(broad["tr"][broad["tr"].index >= bench_win_start])
    for s in equity_syms:
        if s not in H: continue
        h, t = H[s], tick[s]
        info = provider.info(t) if HD.loc[s, "AssetType"] == "equity" else {}
        info_cache[s] = info
        if not HD.loc[s, "Sector"]: HD.loc[s, "Sector"] = info.get("sector", "") or ""
        stype = HD.loc[s, "SectorType"]
        if not stype:
            sec = str(info.get("sector", "")).lower()
            HD.loc[s, "SectorType"] = ("financial" if "financial" in sec else "cyclical" if sec in ("basic materials", "energy") else "standard")
        if not HD.loc[s, "CapTier"] and info.get("marketCap"):
            cr = info["marketCap"] / 1e7
            HD.loc[s, "CapTier"] = ("Large" if cr >= CFG["CAP_LARGE_CR"] else "Mid" if cr >= CFG["CAP_MID_CR"] else "Small") + " (approx)"
        
        capk = CAP_TO_BENCH.get((str(HD.loc[s, "CapTier"]).split() or [""])[0].lower())
        pb, pb_name = (B.get(capk), capk) if HD.loc[s, "BenchmarkUser"] == "" else (broad, "NIFTY500")
        if pb is None: pb, pb_name = broad, "NIFTY500"
        
        P = dict(primary_bench=pb_name, hist_years=hist_years(h["adj"]), bench_maxdd_3y=bench_maxdd_3y)
        for n in (1, 3, 5):
            P[f"tr_{n}y"], P[f"pr_{n}y"] = cagr(h["adj"], n), cagr(h["close"], n)
            P[f"bench_{n}y"], P[f"broad_{n}y"] = cagr(pb["tr"], n), cagr(broad["tr"], n)
            P[f"excess_{n}y"] = P[f"tr_{n}y"] - P[f"bench_{n}y"]
            P[f"excess_broad_{n}y"] = P[f"tr_{n}y"] - P[f"broad_{n}y"]
        
        win = pd.Timestamp(h.index[-1]) - pd.DateOffset(years=CFG["RISK_WINDOW_YEARS"])
        rs = h["adj"][h.index >= win].pct_change().dropna()
        rm = broad["tr"][broad["tr"].index >= win].pct_change().dropna()
        P.update(capm(rs, rm, CFG["RISK_FREE"]))
        
        c = h["close"]
        P["mom_12_1"] = c.iloc[-22] / c.iloc[-253] - 1 if len(c) > 253 else np.nan
        P["ret_6m"] = c.iloc[-1] / c.iloc[-127] - 1 if len(c) > 127 else np.nan
        bc = broad["px"]
        P["rel_6m"] = (P["ret_6m"] - (bc.iloc[-1] / bc.iloc[-127] - 1)) if len(bc) > 127 and not pd.isna(P["ret_6m"]) else np.nan
        P["dist_52w_high"] = c.iloc[-1] / c.iloc[-252:].max() - 1
        P["dma200_gap"] = c.iloc[-1] / c.iloc[-200:].mean() - 1 if len(c) >= 200 else np.nan
        P["maxdd_3y"] = max_drawdown(c[c.index >= win])
        
        d = h.tail(90); d = d[(d.volume > 0) & (d.high != d.low)]
        tv = d.close * d.volume
        if len(tv) >= 15:
            base = min(tv.tail(20).mean(), tv.tail(60).median())
            v = HD.loc[s, "Value"]
            liq[s] = dict(ADTV20=tv.tail(20).mean(), ADTV60_median=tv.tail(60).median(), ADTV_used=base,
                          DaysToExit_10pct=v / (CFG["PARTICIPATION"][0] * base), DaysToExit_20pct=v / (CFG["PARTICIPATION"][1] * base))
            P["days_to_exit"] = liq[s]["DaysToExit_10pct"]
        else:
            P["days_to_exit"] = np.nan
        
        h5 = h[h.index >= h.index[-1] - pd.DateOffset(years=5)]
        for d_, r_ in h5[h5.split > 0].split.items(): ca_rows.append((s, d_.date(), f"Split/bonus {r_:g} (Yahoo)", "Auto-adjusted"))
        ret = h5.close.pct_change()
        near_split = set()
        for d_ in h5[h5.split > 0].index: near_split |= {d_, d_ - pd.Timedelta(days=1), d_ + pd.Timedelta(days=1)}
        for d_, r_ in ret[(ret < -0.25) | (ret > 0.5)].items():
            if d_ not in near_split: ca_rows.append((s, d_.date(), f"Unexplained 1-day move {r_ * 100:.0f}%", "Possible demerger/rights/data error"))

        if HD.loc[s, "AssetType"] == "equity":
            try:
                for ca in (provider.corporate_actions(s) or [])[:3]:
                    ca_rows.append((s, ca.get("exDate", ""), f"NSE: {ca.get('subject', '')[:60]}", "Source: NSE"))
            except: pass

        perf[s] = P
        fundd[s] = (fundamentals(info, provider.statements(t), HD.loc[s, "SectorType"]) if HD.loc[s, "AssetType"] == "equity" else {})
        
    def flows_for(g, s):
        fl = []; divs = H[s]["div"][H[s]["div"] > 0]
        for _, r in g.iterrows():
            if pd.isna(r.buy_date): return None
            fl.append((r.buy_date, -r.quantity * r.buy_price))
            for d_, v_ in divs[divs.index >= r.buy_date].items(): fl.append((d_, r.quantity * v_))
        return fl
    def bench_terminal(fl, bser, end_ts):
        units = 0.0
        for d_, a_ in fl:
            b = bser.asof(d_) if d_ >= bser.index[0] else np.nan
            if pd.isna(b): return np.nan
            units += (-a_) / b
        return units * bser.iloc[-1]

    all_flows, xirr_rows = [], {}
    for s in equity_syms:
        if s not in H or HD.loc[s, "AssetType"] == "cash": continue
        fl = flows_for(lots[lots.symbol == s], s)
        if fl is None: continue
        end_ts, val = H[s].index[-1], HD.loc[s, "Value"]
        s_x = xirr(fl + [(end_ts, val)])
        bt = bench_terminal(fl, broad["tr"], end_ts)
        b_x = xirr(fl + [(end_ts, bt)]) if not pd.isna(bt) else np.nan
        xirr_rows[s] = dict(XIRR=s_x, XIRR_bench_same_flows=b_x, XIRR_excess=s_x - b_x if not pd.isna(b_x) else np.nan)
        if not pd.isna(bt): all_flows.append((fl, val, end_ts))
    port_x = {}
    if all_flows:
        flat = [f for fl, _, _ in all_flows for f in fl]; endt = max(e for _, _, e in all_flows); tv_ = sum(v for _, v, _ in all_flows)
        px_ = xirr(flat + [(endt, tv_)]); bt = bench_terminal(flat, broad["tr"], endt); bx_ = xirr(flat + [(endt, bt)]) if not pd.isna(bt) else np.nan
        port_x = dict(XIRR=px_, Bench=bx_, Excess=px_ - bx_ if not pd.isna(bx_) else np.nan)
    for s, d in xirr_rows.items(): perf[s].update(d)

    peer_med, no_peer = {}, []
    for s in equity_syms:
        if s not in H or HD.loc[s, "AssetType"] != "equity": continue
        key, own, label = val_metric(info_cache.get(s, {}), HD.loc[s, "SectorType"])
        vals = []
        if key and s in peers:
            for p in peers[s]:
                _, pv, _ = val_metric(provider.info(to_ticker(p)), HD.loc[s, "SectorType"])
                if not pd.isna(pv) and pv > 0: vals.append(pv)
        ratio = own / np.median(vals) if (key and vals and not pd.isna(own) and own > 0) else np.nan
        peer_med[s] = dict(ratio=ratio, n=len(vals), label=label, own=own, med=np.median(vals) if vals else np.nan)

    SC = schemes(); score_rows, evid_rows, trig_rows = [], [], []
    for s in equity_syms:
        if s not in H or HD.loc[s, "AssetType"] != "equity": continue
        ev, f, P = evid.get(s, {}), fundd.get(s, {}), perf[s]
        pm = peer_med.get(s, dict(ratio=np.nan, n=0, label="", own=np.nan, med=np.nan))
        S, E, proxies, overrides, bull, bear, engine_rs = score_stock(HD.loc[s, "SectorType"], ev, f, P, pm["ratio"], pm["n"], pm["label"])
        hzn = str(HD.loc[s, "Horizon"]).lower() if "Horizon" in HD.columns and pd.notna(HD.loc[s, "Horizon"]) else "base"
        if hzn not in SC:
            hzn = "base"
            
        ws, cov = weighted(S, SC[hzn])        
        cls = classify(ws, cov)
        alts = {k: classify(weighted(S, w)[0], weighted(S, w)[1]) for k, w in SC.items() if k != hzn}
        sensitive = len({cls, *alts.values()}) > 1
        risk_rs = []
        data_rs = [x for x in engine_rs if "Engine Error" in x]
        risk_rs += [f"HARD OVERRIDE: {o}" for o in overrides]
        if cov < CFG["MIN_COVERAGE"]: data_rs.append(f"Insufficient evidence: only {cov * 100:.0f}% scorable")
        miss = [f"{k} ({FACTOR_NAMES[k]})" for k in SCORABLE if S[k] is None]
        if miss: data_rs.append("Unscored: " + "; ".join(miss))
        if any(x in HD.loc[s, "Surveillance"].upper() for x in ("T2T", "ASM", "GSM")): risk_rs.append(f"Surveillance: {HD.loc[s, 'Surveillance']}")
        if f.get("eq_note"): risk_rs.append(f"Earnings-quality note: {f['eq_note']}")
        conf = "High" if cov >= 0.85 and not proxies else "Medium" if cov >= 0.60 else "Low"
        conf_reason = f"coverage {cov * 100:.0f}%" + (f"; proxy: {','.join(proxies)}" if proxies else "") + (f"; only {pm['n']} peers" if S["F5"] is not None and 0 < pm["n"] < 5 else "")        
        if proxies and conf == "High": conf = "Medium"
        
        final = ("Candidate for review" if any(r.startswith("HARD") for r in risk_rs) else cls)
        score_rows.append(dict(
            Symbol=s, Weight=HD.loc[s, "Weight"], Horizon=hzn, Sector=HD.loc[s, "Sector"], CapTier=HD.loc[s, "CapTier"], SectorType=HD.loc[s, "SectorType"],
            PrimaryBenchmark=P["primary_bench"], Excess_1y=P["excess_1y"], Excess_3y=P["excess_3y"], Excess_5y=P["excess_5y"],
            **{k: S[k] for k in FACTORS}, Weighted=ws, Coverage=cov, Equal=weighted(S, SC["equal"])[0], FwdHeavy=weighted(S, SC["fwd_heavy"])[0], GovHeavy=weighted(S, SC["gov_heavy"])[0],
            Classification=cls, ClassificationSensitive=sensitive, FinalLabel=final, Review_Risk=" | ".join(risk_rs), Review_DataGap=" | ".join(data_rs),
            Confidence=conf, ConfidenceReason=conf_reason, Bull_Signals="; ".join(bull[:4]), Bear_Signals="; ".join(bear[:4])))
        for fct, sc, basis, txt in E: evid_rows.append(dict(Symbol=s, Factor=f"{fct} {FACTOR_NAMES[fct]}", Score=sc, Basis=basis, Evidence=txt))
        
        T = []
        if not pd.isna(P.get("dma200_gap", np.nan)): T.append(f"Weekly close {'below' if P['dma200_gap'] > 0 else 'back above'} 200-DMA")
        if not pd.isna(f.get("nd_ebitda", np.nan)): T.append(f"Net debt/EBITDA crosses 3x (now {f['nd_ebitda']:.1f}x)")
        if not pd.isna(f.get("cfo_ebitda", np.nan)): T.append(f"CFO/EBITDA below 60% (now {f['cfo_ebitda'] * 100:.0f}%)")
        T.append("Promoter pledge rises or crosses 10%/25%")
        if not pd.isna(pm["ratio"]): T.append(f"{pm['label']} beyond 1.4x peer median (now {pm['ratio']:.2f}x)")
        trig_rows.append(dict(Symbol=s, Triggers=" || ".join(T)))
        
    SCD = pd.DataFrame(score_rows)
    if len(SCD):
        SCD["_c"] = SCD.Confidence.map(CONF_RANK)
        SCD = SCD.sort_values(["Weighted", "Weight", "_c"], ascending=[False, False, True], na_position="last").drop(columns="_c").reset_index(drop=True)

    ret_cols = [s for s in equity_syms if s in H]
    R = pd.concat({s: H[s]["adj"].pct_change() for s in ret_cols}, axis=1)
    w_all = HD.loc[ret_cols, "Value"] / HD.loc[ret_cols, "Value"].sum()
    win_end = R.index.max(); Rw = R[R.index >= win_end - pd.DateOffset(years=CFG["RISK_WINDOW_YEARS"])]
    mask = Rw.notna().astype(float); denom = (mask * w_all.values).sum(axis=1).replace(0, np.nan)
    pr = ((Rw.fillna(0) * w_all.values).sum(axis=1) / denom).dropna()
    port = {}
    bm_r = broad["tr"].pct_change().reindex(pr.index); pw = (1 + pr).cumprod(); bw = (1 + bm_r.fillna(0)).cumprod()
    if len(pr) > 252:
        yrs = len(pr) / 252
        port["Back-cast portfolio TR CAGR (current weights held constant)"] = pw.iloc[-1] ** (1 / yrs) - 1
        port["Nifty 500 TR CAGR (same window)"] = bw.iloc[-1] ** (1 / yrs) - 1
        port["Annualised volatility"] = pr.std() * math.sqrt(252)
        port["Sharpe ratio (rf=%.1f%%)" % (CFG["RISK_FREE"] * 100)] = (port["Back-cast portfolio TR CAGR (current weights held constant)"] - CFG["RISK_FREE"]) / port["Annualised volatility"]
        port["Max drawdown (window)"] = max_drawdown(pw)
        cp = capm(pr, bm_r.dropna(), CFG["RISK_FREE"])
        if cp: port.update({"Beta vs Nifty 500": cp["beta"], "Jensen alpha (annualised)": cp["alpha_ann"], "Alpha t-stat": cp["alpha_t"], "R-squared": cp["r2"]})
    
    ra = HD[~HD.AssetType.isin(["cash", "bond"])]
    ra_w = (ra.Value / ra.Value.sum() * 100).sort_values(ascending=False)
    port["Cash/bond weight % of total"] = float(HD[HD.AssetType.isin(["cash", "bond"])].Weight.sum())
    port["Top-1 weight % (of risk assets)"] = float(ra_w.iloc[0]) if len(ra_w) else 0.0
    port["Top-5 weight % (of risk assets)"] = ra_w.head(5).sum()
    port["Top-10 weight % (of risk assets)"] = ra_w.head(10).sum()
    port["HHI (risk-asset weights)"] = float(((ra_w / 100) ** 2).sum())
    port["Effective number of positions (1/HHI)"] = 1 / port["HHI (risk-asset weights)"]
    
    sec = (HD[HD.AssetType == "equity"].groupby(HD.Sector.replace("", "Unclassified")).Weight.sum().sort_values(ascending=False))
    cap = (HD.groupby(HD.CapTier.replace("", "Unclassified").where(HD.AssetType == "equity", HD.AssetType)).Weight.sum().sort_values(ascending=False))
    CONC = pd.concat([pd.DataFrame(dict(Dimension="Sector", Bucket=sec.index, Weight=sec.values)), pd.DataFrame(dict(Dimension="Cap tier / asset type", Bucket=cap.index, Weight=cap.values))])
    CONC["Flag"] = ""
    CONC.loc[(CONC.Dimension == "Sector") & (CONC.Weight > CFG["CONC_SECTOR_PCT"]), "Flag"] = f"> {CFG['CONC_SECTOR_PCT']}% (model setting)"
    single_ = HD[(HD.AssetType == "equity") & (HD.Weight > CFG["CONC_SINGLE_PCT"])].sort_values("Weight", ascending=False)
    extra_ = [pd.DataFrame(dict(Dimension="Single stock", Bucket=single_.index, Weight=single_.Weight.values, Flag=f"> {CFG['CONC_SINGLE_PCT']}% (model setting)"))] if len(single_) else []
    if port["Top-5 weight % (of risk assets)"] > CFG["CONC_TOP5_PCT"]:
        extra_.append(pd.DataFrame(dict(Dimension="Top-5 combined", Bucket=["Top 5 holdings"], Weight=[port["Top-5 weight % (of risk assets)"]], Flag=f"> {CFG['CONC_TOP5_PCT']}% (model setting)")))
    if extra_: CONC = pd.concat([CONC] + extra_, ignore_index=True)

    n_hard = int(SCD.Review_Risk.str.contains("HARD").sum()) if len(SCD) else 0
    n_risk = int((SCD.Review_Risk != "").sum()) if len(SCD) else 0
    n_data = int((SCD.Review_DataGap != "").sum()) if len(SCD) else 0
    top_concerns = (SCD[SCD.FinalLabel.isin(["Candidate for review", "Underweight"]) & (SCD.Coverage >= CFG["MIN_COVERAGE"])].head(5) if len(SCD) else SCD)


    
    # --- ENHANCED PORTFOLIO SUMMARY (Recommendation 3) ---
    # 1. Calculate value-weighted score
    scd_valid = SCD.dropna(subset=["Weighted"])
    if len(scd_valid) > 0:
        w_score = (scd_valid["Weighted"] * scd_valid["Weight"]).sum() / scd_valid["Weight"].sum()
    else:
        w_score = float("nan")
        
    # 2. Calculate % of capital in each label
    label_weights = SCD.groupby("FinalLabel")["Weight"].sum()
    under_wt = label_weights.get("Underweight", 0)
    review_wt = label_weights.get("Candidate for review", 0)
    neutral_wt = label_weights.get("Neutral", 0)
    over_wt = label_weights.get("Overweight", 0)
    
    # 3. Calculate capital in hard overrides
    hard_wt = SCD[SCD["Review_Risk"].str.contains("HARD", na=False)]["Weight"].sum()
    
    # 4. Get Top 5 concerns by rupee value
    concerns = SCD[SCD["FinalLabel"].isin(["Underweight", "Candidate for review"])].sort_values("Weight", ascending=False).head(5)
    concern_strs = []
    for _, r in concerns.iterrows():
        val_inr = (r["Weight"] / 100) * total_value
        concern_strs.append(f"{r['Symbol']} (₹{val_inr/1e5:.1f}L)")
        
    # 5. Build the final summary list
    summary = [
        f"Tool v{VERSION} | analysis {dt.date.today()} | price as-of {HD.PriceAsOf.max().date() if HD.PriceAsOf.notna().any() else 'n/a'} | INR | data: {provider.name}",
        f"Portfolio INR {total_value:,.0f} across {len(HD)} lines ({len(ret_cols)} priced); cost INR {HD.Invested.sum():,.0f}; unrealised P&L {HD.PnL.sum() / HD.Invested.sum() * 100:+.1f}%.",
        f"Portfolio Score: {w_score:.2f} (Value-weighted average of all scored holdings).",
        f"Capital Allocation: Underweight: {under_wt:.1f}% | Review: {review_wt:.1f}% | Neutral: {neutral_wt:.1f}% | Overweight: {over_wt:.1f}%.",
    ]
    
    if hard_wt > 0:
        summary.append(f"WARNING: {hard_wt:.1f}% of your capital is in stocks with HARD overrides (governance/earnings red flags).")
        
    if concern_strs:
        summary.append(f"Top 5 Concerns by Capital: " + ", ".join(concern_strs))
        
    if port_x and not pd.isna(port_x.get("XIRR", np.nan)):
        summary.append(f"Investor-level XIRR {pct(port_x['XIRR'])} vs {pct(port_x['Bench'])} had the same cash flows gone into Nifty 500 TR -> excess {pct(port_x['Excess'])}.")
        
    summary.append(f"Flags: {n_hard} hard-override, {n_risk} risk-evidenced, {n_data} data-gap.")
    # --- Tax Calculation ---
    tax_rows = []
    for _, r in lots.iterrows():
        if r.asset_type in ("cash", "bond") or r.symbol not in H:
            continue
        cmp_ = HD.loc[r.symbol, "CMP"]
        row = dict(Symbol=r.symbol, Qty=r.quantity, BuyPrice=r.buy_price, BuyDate=r.buy_date,
                   CMP=cmp_, UnrealisedGain=(cmp_ - r.buy_price) * r.quantity)
        if pd.isna(r.buy_date):
            row.update(Status="unknown (no buy_date)", LTCG_Date=pd.NaT, DaysToLTCG=np.nan)
        else:
            ltcg_date = r.buy_date + pd.DateOffset(months=12) + pd.Timedelta(days=1)
            long_ = review_date >= ltcg_date
            row.update(Status="LTCG (indicative)" if long_ else "STCG (indicative)",
                       LTCG_Date=ltcg_date,
                       DaysToLTCG=0 if long_ else (ltcg_date - review_date).days,
                       CrossesWithin90d=(not long_) and (ltcg_date - review_date).days <= CFG["LTCG_SOON_DAYS"])
        tax_rows.append(row)
    TAX = pd.DataFrame(tax_rows)
    # --- RECOMMENDATION 6: CHANGE TRACKING ---
    snapshot_path = out / ".cache" / "last_run_snapshot.csv"
    if snapshot_path.exists():
        try:
            prev_scd = pd.read_csv(snapshot_path)
            # Keep only relevant columns
            prev_scd = prev_scd[["Symbol", "Weighted", "FinalLabel", "Review_Risk"]].rename(columns={
                "Weighted": "Prev_Score",
                "FinalLabel": "Prev_Label",
                "Review_Risk": "Prev_Risk"
            })
            SCD = SCD.merge(prev_scd, on="Symbol", how="left")
            SCD["Score_Delta"] = SCD["Weighted"] - SCD["Prev_Score"]
            SCD["Label_Changed"] = SCD["FinalLabel"] != SCD["Prev_Label"]
        except Exception as e:
            print(f"Could not load previous snapshot: {e}")
            SCD["Score_Delta"], SCD["Prev_Label"], SCD["Label_Changed"] = np.nan, np.nan, False
    else:
        SCD["Score_Delta"], SCD["Prev_Label"], SCD["Label_Changed"] = np.nan, np.nan, False

    # Save current snapshot for next run
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    SCD[["Symbol", "Weighted", "FinalLabel", "Review_Risk"]].to_csv(snapshot_path, index=False)
    # --- ACTION LAYER (see build_actions) ---
    ACT = build_actions(SCD, HD, liq, total_value, TAX)
    SCD = SCD.merge(ACT, left_on="Symbol", right_index=True, how="left")

    # --- RECOMMENDATION 3: SUMMARY ROLL-UP ---
    scd_valid = SCD.dropna(subset=["Weighted"])
    if len(scd_valid) > 0:
        w_score = (scd_valid["Weighted"] * scd_valid["Weight"]).sum() / scd_valid["Weight"].sum()
    else:
        w_score = float("nan")
        
    label_weights = SCD.groupby("FinalLabel")["Weight"].sum()
    under_wt = label_weights.get("Underweight", 0)
    review_wt = label_weights.get("Candidate for review", 0)
    neutral_wt = label_weights.get("Neutral", 0)
    over_wt = label_weights.get("Overweight", 0)
    hard_wt = SCD[SCD["Review_Risk"].str.contains("HARD", na=False)]["Weight"].sum()
    
    concerns = SCD[SCD["FinalLabel"].isin(["Underweight", "Candidate for review"])].sort_values("Weight", ascending=False).head(5)
    concern_strs = []
    for _, r in concerns.iterrows():
        val_inr = (r["Weight"] / 100) * total_value
        concern_strs.append(f"{r['Symbol']} (₹{val_inr/1e5:.1f}L)")
        
    # --- evidence register ---
    try:
        _vp = Path(args.evidence).parent / "verified_flags.csv"
        _ver = pd.read_csv(_vp) if _vp.exists() else None
    except Exception:
        _ver = None
    REG = build_evidence_register(HD, SCD, fundd, auto_evid, evid, total_value, _ver)
    n_unver = int(REG.Status.str.startswith("Automated").sum()) if len(REG) else 0

    # --- data-integrity status line ---
    priced_inv = HD.loc[HD.Priced, "Invested"].sum()
    n_unpriced = int((~HD.Priced).sum())
    bad = HD[HD.CostFlag != ""]
    pnl_all = HD.PnL.sum() / priced_inv * 100 if priced_inv else np.nan
    pnl_line = (f"Portfolio INR {total_value:,.0f} across {len(HD)} lines ({int(HD.Priced.sum())} priced); cost INR {priced_inv:,.0f} (priced lines only); "
                f"unrealised P&L {pnl_all:+.1f}% as reported.")
    ok_ = HD[HD.Priced & (HD.CostFlag == "")]
    if len(bad) and ok_.Invested.sum() > 0:
        pnl_line += (f" {len(bad)} lines ({bad.Value.sum() / total_value * 100:.1f}% of value but {bad.Invested.sum() / priced_inv * 100:.1f}% of cost) "
                     f"have unreconciled cost bases; without them P&L is {ok_.PnL.sum() / ok_.Invested.sum() * 100:+.1f}%.")
    reasons = []
    if len(bad): reasons.append(f"{len(bad)} lines with an unreconciled cost basis (see Data gaps)")
    if n_unpriced: reasons.append(f"{n_unpriced} line(s) with no price, left out of value, cost and P&L")
    if not (port_x and not pd.isna(port_x.get("XIRR", np.nan))): reasons.append("no dated transactions, so your XIRR and actual performance cannot be calculated (return figures are a back-cast at today's weights)")
    f5_ = [e for e in evid_rows if str(e.get("Factor", "")).startswith("F5")]
    if f5_ and sum(1 for e in f5_ if e.get("Basis") == "absolute fallback") / len(f5_) > 0.5: reasons.append("valuation uses absolute P/E or P/B bands (no peer set)")
    status_line = ("STATUS: PROVISIONAL - " + "; ".join(reasons) + ".") if reasons else "STATUS: data checks passed."

    summary = [
        f"Tool v{VERSION} | analysis {dt.date.today()} | price as-of {HD.PriceAsOf.max().date() if HD.PriceAsOf.notna().any() else 'n/a'} | INR | data: {provider.name}",
        status_line,
        pnl_line,
        f"Portfolio Score: {w_score:.2f} (Value-weighted average of all scored holdings).",
        f"Capital Allocation: Underweight: {under_wt:.1f}% | Review: {review_wt:.1f}% | Neutral: {neutral_wt:.1f}% | Overweight: {over_wt:.1f}%.",
    ]
    
    if hard_wt > 0:
        summary.append(f"WARNING: {hard_wt:.1f}% of your capital is in stocks with HARD overrides (governance/earnings red flags).")
    if concern_strs:
        summary.append(f"Top 5 Concerns by Capital: " + ", ".join(concern_strs))
    if port_x and not pd.isna(port_x.get("XIRR", np.nan)):
        summary.append(f"Investor-level XIRR {pct(port_x['XIRR'])} vs {pct(port_x['Bench'])} had the same cash flows gone into Nifty 500 TR -> excess {pct(port_x['Excess'])}.")
        
    n_hard = int(SCD.Review_Risk.str.contains("HARD").sum()) if len(SCD) else 0
    n_risk = int((SCD.Review_Risk != "").sum()) if len(SCD) else 0
    n_data = int((SCD.Review_DataGap != "").sum()) if len(SCD) else 0
    summary.append(f"Flags: {n_hard} hard-override, {n_risk} risk-evidenced, {n_data} data-gap.")
    summary.append(f"Evidence register: {len(REG)} flags, {n_unver} not independently verified (Evidence_Register sheet; add verified_flags.csv to record checks).")
    
    HOLD = HD.reset_index()[["Symbol", "Ticker", "AssetType", "Sector", "SectorType", "CapTier", "Qty", "AvgCost", "Invested", "CMP", "PriceAsOf", "Value", "PnL", "PnLPct", "Priced", "CostFlag", "Weight", "Target", "TargetDev_pp", "OutOfBand", "Surveillance", "UserXIRR"]]
    LQ = pd.DataFrame(liq).T.reset_index().rename(columns={"index": "Symbol"}) if liq else pd.DataFrame()
    if len(LQ):
        LQ["Exit_flag"] = LQ.DaysToExit_10pct > CFG["EXIT_DAYS_FLAG"]
        LQ = LQ.merge(HD.reset_index()[["Symbol", "Value", "Surveillance"]], on="Symbol")
        LQ["Tier_note"] = np.where(LQ.Surveillance.str.upper().str.contains("T2T"), "T2T => treat as Tier-B liquidity", "")
    PERF = pd.DataFrame(perf).T.reset_index().rename(columns={"index": "Symbol"}) if perf else pd.DataFrame()
    TRIG = pd.DataFrame(trig_rows); EVD = pd.DataFrame(evid_rows)
    CA = pd.DataFrame(ca_rows, columns=["Symbol", "Date", "Event", "Note"])
    GAPS = pd.DataFrame(gaps + [("portfolio", "note", n) for n in notes_global], columns=["Item", "Missing/Issue", "Impact"])
    PORT = pd.DataFrame([(k, v) for k, v in port.items()], columns=["Metric", "Value"])
    METHOD = pd.DataFrame([(k, str(v)) for k, v in CFG.items()], columns=["Parameter", "Value"])
    SUMM = pd.DataFrame(dict(Executive_summary=summary))

    sheets = {"Summary": SUMM, "Portfolio": PORT, "Holdings": HOLD, "Scorecard_A": SCD, "Evidence": EVD, "Performance_Risk": PERF, "Liquidity": LQ, "Concentration": CONC, "Tax_Lots": pd.DataFrame(), "Corporate_Actions": CA, "Monitoring_Triggers": TRIG, "Data_Gaps": GAPS, "Evidence_Register": REG, "Method": METHOD}
    
    meta = dict(demo=args.demo, source=provider.name, bench_src=broad["src"])
    html_path = write_html(out, sheets, meta, pr=pr, bm_r=bm_r, port=port, port_x=port_x, summary=summary, conc_pairs=[], fps=SCD.head(0), RC=pd.DataFrame(), liq=liq, tax_est=None)
    # --- EXCEL EXPORT (Recommendation 10) ---
    print("Generating Excel workbook with all data sheets...")
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    xlsx_path = out / f"portfolio_diagnostic_{stamp}.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            for name, df in sheets.items():
                if len(df) > 0:
                    df.to_excel(xw, sheet_name=name[:31], index=False)
        print(f"Excel report: {xlsx_path}")
    except Exception as e:
        print(f"Could not generate Excel file: {e}")

    print(f"\nHTML report: {html_path}")
    if not args.no_open:
        try: webbrowser.open(html_path.resolve().as_uri())
        except Exception as e: print(f"Could not open browser: {e}")
    return sheets

# =============================================================================
# HTML REPORT (Rich V2 Template)
# =============================================================================
BADGE_CLS = {"Overweight": "ok", "Neutral": "neu", "Underweight": "bad", "Not classified": "warn", "Candidate for review": "alert"}
_CSS = """
:root{--bg:#f4f6fb;--card:#fff;--card2:#f8f9fd;--ink:#192037;--muted:#66708d;--line:#e3e7f1;--accent:#3556d8;--accent-soft:rgba(53,86,216,.09);--pos:#168a5c;--neg:#cf3d3d;--warn:#c98216;--s1:#1f9d6b;--s2:#78b957;--s3:#e0b52f;--s4:#e8843a;--s5:#d64545;--shadow:0 1px 2px rgba(25,32,55,.05),0 8px 24px rgba(25,32,55,.06)}
@media (prefers-color-scheme:dark){:root{--bg:#0e1220;--card:#161b2e;--card2:#1b2138;--ink:#e9ecf7;--muted:#98a1c0;--line:#272f4a;--accent:#7f99ff;--accent-soft:rgba(127,153,255,.14);--pos:#3cc48f;--neg:#ff6b6b;--warn:#e9a23b;--s1:#2fae7c;--s2:#82c761;--s3:#d9b640;--s4:#e58a42;--s5:#e05252;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.28)}}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;font-variant-numeric:tabular-nums}
.wrap{max-width:1280px;margin:0 auto;padding:0 22px 60px}
header.hero{padding:34px 0 18px}.hero h1{margin:0;font-size:30px;letter-spacing:-.02em}
.hero .sub{color:var(--muted);margin-top:6px;font-size:13.5px}
.demo{background:var(--neg);color:#fff;padding:8px 14px;border-radius:10px;font-weight:600;margin-top:14px;display:inline-block}
nav.top{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 86%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
nav.top .in{max-width:1280px;margin:0 auto;padding:10px 22px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
nav.top a{color:var(--muted);text-decoration:none;padding:5px 11px;border-radius:999px;font-size:13px}
nav.top a:hover{background:var(--accent-soft);color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:8px 0 26px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;box-shadow:var(--shadow)}
.kpi .l{font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.kpi .v{font-size:25px;font-weight:700;margin:4px 0 2px;letter-spacing:-.01em}
.kpi .s{font-size:12.5px;color:var(--muted)}
.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--muted)}
section.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px 24px;margin:0 0 22px;box-shadow:var(--shadow)}
section.card h2{margin:0 0 4px;font-size:19px;letter-spacing:-.01em}
section.card .lead{color:var(--muted);margin:0 0 16px;font-size:13.5px}
h3{font-size:14px;margin:18px 0 8px;color:var(--ink)}
ul.clean{margin:0;padding-left:18px}ul.clean li{margin:5px 0}.summary li{margin:8px 0}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13.2px}
th{background:var(--card2);color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;text-align:left;padding:10px 12px;white-space:nowrap;border-bottom:1px solid var(--line)}
td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:none}tbody tr:hover td{background:var(--accent-soft)}
td.r,th.r{text-align:right}td.c,th.c{text-align:center}.sym{font-weight:700}
.chip{display:inline-block;min-width:30px;text-align:center;padding:2px 7px;border-radius:8px;font-weight:700;font-size:12px;color:#fff}
.chip.s1{background:var(--s1)}.chip.s2{background:var(--s2)}.chip.s3{background:var(--s3);color:#2b2200}.chip.s4{background:var(--s4)}.chip.s5{background:var(--s5)}
.chip.na{background:transparent;color:var(--muted);border:1px dashed var(--line);font-weight:500}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap;border:1px solid transparent}
.badge.ok{background:rgba(31,157,107,.14);color:var(--pos)}.badge.neu{background:var(--accent-soft);color:var(--accent)}.badge.bad{background:rgba(214,69,69,.14);color:var(--neg)}.badge.warn{background:rgba(224,170,47,.18);color:var(--warn)}.badge.alert{background:var(--neg);color:#fff}
.bar{height:8px;background:var(--line);border-radius:6px;overflow:hidden;min-width:70px}
.bar span{display:block;height:100%;background:var(--accent);border-radius:6px}
.bar.s1 span{background:var(--s1)}.bar.s2 span{background:var(--s2)}.bar.s3 span{background:var(--s3)}.bar.s4 span{background:var(--s4)}.bar.s5 span{background:var(--s5)}.bar.alt span{background:var(--warn)}
.ws{display:flex;align-items:center;gap:8px}.ws b{min-width:34px}
.dbar{position:relative;height:8px;min-width:110px;background:var(--line);border-radius:6px}.dbar:after{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted);opacity:.6}
.dbar span{position:absolute;top:0;bottom:0;border-radius:6px}.dbar span.pos{background:var(--pos)}.dbar span.neg{background:var(--neg)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:24px}
.row{display:grid;grid-template-columns:120px 1fr 60px;gap:10px;align-items:center;margin:7px 0;font-size:13px}
.row .lab{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.row .num{text-align:right;color:var(--muted)}
.pair{display:grid;grid-template-columns:120px 1fr 56px;gap:10px;align-items:center;margin:5px 0;font-size:13px}.pair .bars{display:grid;gap:3px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin:2px 0 10px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-2px}
svg.chart{width:100%;height:auto;display:block}svg .grid{stroke:var(--line);stroke-width:1}svg .axis{fill:var(--muted);font-size:11px}
svg .line{fill:none;stroke-width:2.2;stroke-linejoin:round}svg .line.a{stroke:var(--accent)}svg .line.b{stroke:var(--muted);stroke-dasharray:5 4}
details.hold{border:1px solid var(--line);border-radius:14px;margin:10px 0;background:var(--card2)}
details.hold>summary{list-style:none;cursor:pointer;padding:13px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
details.hold>summary::-webkit-details-marker{display:none}
details.hold>summary:before{content:"\\25B8";color:var(--muted);transition:transform .15s}
details.hold[open]>summary:before{transform:rotate(90deg)}
details.hold .body{padding:4px 18px 18px;border-top:1px solid var(--line)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:22px}
dl.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;margin:0;font-size:13px}dl.kv dt{color:var(--muted)}dl.kv dd{margin:0;text-align:right}
.tag{display:inline-block;font-size:12px;padding:2px 9px;border-radius:8px;margin:2px 4px 2px 0}
.tag.risk{background:rgba(214,69,69,.13);color:var(--neg)}.tag.data{background:rgba(224,170,47,.17);color:var(--warn)}.tag.bull{background:rgba(31,157,107,.13);color:var(--pos)}.tag.bear{background:rgba(214,69,69,.10);color:var(--neg)}
.note{background:var(--accent-soft);border-radius:12px;padding:12px 16px;font-size:13px;margin:12px 0}
.heat td,.heat th{padding:5px 6px;text-align:center;font-size:11.5px}.heat th{white-space:nowrap}
footer{color:var(--muted);font-size:12.5px;margin-top:30px;padding-top:16px;border-top:1px solid var(--line)}
@media print{nav.top{display:none}body{background:#fff}details.hold .body{display:block}section.card{box-shadow:none;break-inside:avoid-page}}
"""
def esc(x): return "" if x is None or (isinstance(x, float) and np.isnan(x)) else _html.escape(str(x))
def inr(x):
    if x is None or pd.isna(x): return "&ndash;"
    n = int(round(float(x))); sign, s = ("-" if n < 0 else ""), str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]; parts = [head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1]; s = ",".join(parts) + "," + tail
    return f"{sign}&#8377;{s}"
def inr_short(x):
    if x is None or pd.isna(x): return "&ndash;"
    a, sign = abs(float(x)), ("-" if x < 0 else "")
    if a >= 1e7: return f"{sign}&#8377;{a / 1e7:.2f} Cr"
    if a >= 1e5: return f"{sign}&#8377;{a / 1e5:.2f} L"
    return f"{sign}&#8377;{a:,.0f}"
def p1(x, d=1, sign=False): return "&ndash;" if x is None or pd.isna(x) else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")
def num(x, d=2, suffix=""): return "&ndash;" if x is None or pd.isna(x) else f"{x:.{d}f}{suffix}"
def tone(x): return "" if x is None or pd.isna(x) or abs(x) < 1e-9 else ("pos" if x > 0 else "neg")
def chip(v):
    if v is None or pd.isna(v): return '<span class="chip na" title="not scored">&ndash;</span>'
    r = int(min(5, max(1, round(float(v))))); return f'<span class="chip s{r}" title="score {float(v):.2f}">{float(v):.1f}</span>'
def badge(t): return f'<span class="badge {BADGE_CLS.get(t, "neu")}">{esc(t)}</span>'
def td(content, cls=""): return f'<td class="{cls}">{content}</td>' if cls else f"<td>{content}</td>"
def th(text, cls=""): return f'<th class="{cls}">{text}</th>' if cls else f"<th>{text}</th>"
def table(headers, rows, extra=""): return f'<div class="scroll"><table {extra}><thead><tr>{"".join(th(h, c) for h, c in headers)}</tr></thead><tbody>{"".join("<tr>" + "".join(r) + "</tr>" for r in rows)}</tbody></table></div>'
def bar(value, vmax, cls=""): return f'<div class="bar {cls}"><span style="width:{0 if (pd.isna(value) or vmax <= 0) else min(100, max(0, value / vmax * 100)):.1f}%"></span></div>'
def divbar(v, scale=0.30):
    if v is None or pd.isna(v): return '<span class="muted">&ndash;</span>'
    w = min(50.0, abs(v) / scale * 50)
    return f'<div class="dbar"><span class="pos" style="left:50%;width:{w:.1f}%"></span></div>' if v >= 0 else f'<div class="dbar"><span class="neg" style="right:50%;width:{w:.1f}%"></span></div>'

def svg_growth(series, w=960, h=320):
    ml, mr, mt, mb = 50, 14, 12, 28
    frame = pd.concat([s.rename(i) for i, (_, s, _) in enumerate(series)], axis=1).dropna(how="all")
    if frame.empty: return ""
    d0, d1 = frame.index.min(), frame.index.max(); lo, hi = float(frame.min().min()), float(frame.max().max())
    pad = (hi - lo) * 0.06 or 1.0; lo, hi = lo - pad, hi + pad; span = max(1, (d1 - d0).days)
    X = lambda d: ml + (d - d0).days / span * (w - ml - mr); Y = lambda v: mt + (1 - (v - lo) / (hi - lo)) * (h - mt - mb)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Growth of 100">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        o.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text class="axis" x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    for yr in range(d0.year + 1, d1.year + 1):
        d = pd.Timestamp(year=yr, month=1, day=1)
        if d0 < d < d1: o.append(f'<line class="grid" x1="{X(d):.1f}" x2="{X(d):.1f}" y1="{mt}" y2="{h - mb}"/><text class="axis" x="{X(d):.1f}" y="{h - 8}" text-anchor="middle">{yr}</text>')
    for _, s, cls in series:
        s = s.dropna(); s2 = s.iloc[::max(1, len(s) // 320)]
        if s.index[-1] != s2.index[-1]: s2 = pd.concat([s2, s.iloc[[-1]]])
        o.append(f'<polyline class="line {cls}" points="' + " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in s2.items()) + '"/>')
    o.append("</svg>")
    return "".join(o)

def write_html(out, sheets, meta, pr, bm_r, port, port_x, summary, conc_pairs, fps, RC, liq, tax_est):
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M"); path = out / f"portfolio_report_{stamp}.html"
    G = lambda k: sheets.get(k, pd.DataFrame())
    SCD = G("Scorecard_A"); HD = G("Holdings").set_index("Symbol") if len(G("Holdings")) else pd.DataFrame()
    perf = G("Performance_Risk").set_index("Symbol").to_dict("index") if len(G("Performance_Risk")) else {}
    TV = HD.Value.sum() if len(HD) else 0; TC = HD.loc[HD.Priced, "Invested"].sum() if len(HD) else 0; pnl = TV - TC
    n_hard = int(SCD.Review_Risk.str.contains("HARD").sum()) if len(SCD) else 0
    n_risk = int((SCD.Review_Risk != "").sum()) if len(SCD) else 0
    n_data = int((SCD.Review_DataGap != "").sum()) if len(SCD) else 0
    asof = (HD.PriceAsOf.max().date() if (len(HD) and HD.PriceAsOf.notna().any()) else "n/a")
    px = port_x or {}
    kp = [("Portfolio value", inr_short(TV), f"cost {inr_short(TC)}", ""), ("Unrealised P&amp;L", p1(pnl / TC, 1, True) if TC else "&ndash;", inr_short(pnl), tone(pnl)), ("Your XIRR", p1(px.get("XIRR")), ("money-weighted, lot level" if not pd.isna(px.get("XIRR", np.nan)) else "not calculable: no dated transactions"), tone(px.get("XIRR"))), ("XIRR vs Nifty 500", p1(px.get("Excess"), 1, True), f"index on same cash flows: {p1(px.get('Bench'))}", tone(px.get("Excess"))), ("Beta / volatility", f"{num(port.get('Beta vs Nifty 500'))} / {p1(port.get('Annualised volatility'), 0)}", "current weights, back-cast", ""), ("Max drawdown", p1(port.get("Max drawdown (window)")), f"{CFG['RISK_WINDOW_YEARS']}y window, back-cast", "neg" if not pd.isna(port.get("Max drawdown (window)", np.nan)) else ""), ("Effective positions", num(port.get("Effective number of positions (1/HHI)"), 1), f"top-5 = {num(port.get('Top-5 weight % (of risk assets)'), 0)}% of risk assets", ""), ("Review flags", f"{n_hard} hard &middot; {n_risk} risk", f"{n_data} with data gaps (separate)", "neg" if n_hard else "")]
    kpis = "".join(f'<div class="kpi"><div class="l">{l}</div><div class="v {c}">{v}</div><div class="s">{s}</div></div>' for l, v, s, c in kp)
    summ = "<ul class='clean summary'>" + "".join(f"<li>{esc(s)}</li>" for s in summary) + "</ul>"
    if conc_pairs: summ += f"<h3>Most correlated pairs</h3><p class='muted'>{esc('; '.join(conc_pairs))}</p>"
    sc_rows = []
    for i, r in enumerate(SCD.itertuples(), 1):
        sens = ' <span title="classification changes" class="muted">&#9888;</span>' if r.ClassificationSensitive else ""
        w_sc = f'<div class="ws"><b>{r.Weighted:.2f}</b>{bar((r.Weighted - 1) / 4 * 100, 100, "s" + str(int(min(5, max(1, round(r.Weighted))))))}</div>' if not pd.isna(r.Weighted) else "&ndash;"
        change_str = "&ndash;"
        if not pd.isna(r.Score_Delta):
            if r.Score_Delta > 0.05:
                change_str = f"<span class='neg'>↑ {r.Score_Delta:.2f}</span>"
            elif r.Score_Delta < -0.05:
                change_str = f"<span class='pos'>↓ {abs(r.Score_Delta):.2f}</span>"
            else:
                change_str = "<span class='muted'>0.00</span>"
                
        sc_rows.append([td(str(i), "c"), td(f'<a class="sym" href="#h-{esc(r.Symbol)}" style="color:inherit;text-decoration:none">{esc(r.Symbol)}</a>'), td(f"{r.Weight:.1f}%", "r"), td(esc(r.Sector) or "&ndash;")] + [td(chip(getattr(r, f)), "c") for f in FACTORS] + [td(w_sc), td(f"{r.Coverage * 100:.0f}%", "r"), td(badge(r.Classification) + sens), td(badge(r.FinalLabel)), td(esc(r.Confidence)), td(change_str, "c")])        

    sc_tbl = table([("#", "c"), ("Stock", ""), ("Weight", "r"), ("Sector", "")]
                   + [(f, "c") for f in FACTORS]
                   + [("Weighted", ""), ("Coverage", "r"), ("Classification", ""),
                      ("Final label", ""), ("Confidence", ""), ("Δ Score", "c")], sc_rows) \
             if sc_rows else "<p>No scorable holdings.</p>"
    fnames = " &middot; ".join(f"<b>{f}</b> {esc(FACTOR_NAMES[f])} ({int(WEIGHTS[f] * 100)}%)" for f in FACTORS)
    legend = '<div class="legend"><span><i style="background:var(--s1)"></i>1 favourable</span><span><i style="background:var(--s3)"></i>3 mixed</span><span><i style="background:var(--s5)"></i>5 unfavourable</span><span>&ndash; not scored</span></div>'
    EV, TR = G("Evidence"), G("Monitoring_Triggers"); cards = []
    for r in SCD.itertuples():
        s, p = r.Symbol, perf.get(r.Symbol, {}); risk = [x.strip() for x in str(r.Review_Risk).split(" | ") if x.strip()]; data = [x.strip() for x in str(r.Review_DataGap).split(" | ") if x.strip()]
        tags = "".join(f'<span class="tag risk">{esc(x)}</span>' for x in risk) + "".join(f'<span class="tag data">{esc(x)}</span>' for x in data)
        ev_rows = []
        if len(EV):
            for e in EV[EV.Symbol == s].itertuples(): ev_rows.append([td(esc(e.Factor)), td(chip(e.Score), "c"), td(esc(e.Basis)), td(esc(e.Evidence))])
        ev_tbl = table([("Factor", ""), ("Score", "c"), ("Basis", ""), ("Evidence", "")], ev_rows) if ev_rows else "<p class='muted'>No factor scored.</p>"
        trg = ""
        if len(TR):
            tt = TR[TR.Symbol == s]
            if len(tt): trg = "<ul class='clean'>" + "".join(f"<li>{esc(x)}</li>" for x in str(tt.iloc[0].Triggers).split(" || ")) + "</ul>"
        L = liq.get(s)
        act_text = "HOLD"
        if r.Action == "Add": act_text = f"<b class='pos'>ADD ₹{r.Amount_INR/1e5:.1f}L</b> <span class='muted'>to your target</span>"
        elif r.Action == "Trim": act_text = f"<b class='neg'>TRIM ₹{r.Amount_INR/1e5:.1f}L</b> <span class='muted'>to your target</span>"
        elif r.Action == "Review / Exit": act_text = "<b class='neg'>REVIEW</b> <span class='muted'>verify the flag before any sale</span>"
        elif r.Action == "Risk alert": act_text = f"<b class='neg'>RISK ALERT</b> <span class='muted'>illustrative: ₹{r.Amount_INR/1e5:.1f}L to reach the limit</span>"
        
        tax_line = f"{r.Tax_Note}" if r.Tax_Note else "&ndash;"
        liq_line = f"{r.Liquidity_Note}" if r.Liquidity_Note else "&ndash;"
        kv = [
            ("Horizon Profile", str(r.Horizon).title()),
            ("Weight / target", f"{r.Weight:.1f}% / {num(HD.loc[s, 'Target'], 1, '%') if (s in HD.index and not pd.isna(HD.loc[s, 'Target'])) else '&ndash;'}"),
            ("Action", act_text),
            ("Research signal", r.ResearchSignal),
            ("Action Rationale", r.Reason),
            ("Tax Note", tax_line),
            ("Liquidity Note", liq_line),
            ("Total return 1y / 3y / 5y", f"{p1(p.get('tr_1y'))} / {p1(p.get('tr_3y'))} / {p1(p.get('tr_5y'))}"),
            (f"Primary benchmark ({esc(p.get('primary_bench', ''))}) 3y", p1(p.get("bench_3y"))),
            ("Excess vs primary 3y / 5y (p.a.)", f"{p1(p.get('excess_3y'), 1, True)} / {p1(p.get('excess_5y'), 1, True)}"),
            ("Excess vs Nifty 500 3y (p.a.)", p1(p.get("excess_broad_3y"), 1, True)),
            ("Beta / alpha p.a. (t-stat)", f"{num(p.get('beta'))} / {p1(p.get('alpha_ann'), 1, True)} ({num(p.get('alpha_t'), 1)})"),
            ("Your Broker XIRR", f"<b>{num(HD.loc[s, 'UserXIRR'], 1, '%')}</b>" if 'UserXIRR' in HD.columns and s in HD.index and not pd.isna(HD.loc[s, 'UserXIRR']) else "&ndash;"),
            ("Script XIRR vs index", f"{p1(p.get('XIRR'))} vs {p1(p.get('XIRR_bench_same_flows'))}"),
            ("12-1m momentum / 6m rel. to index", f"{p1(p.get('mom_12_1'), 0, True)} / {p1(p.get('rel_6m'), 0, True)}"),
            ("vs 200-DMA / from 52w high", f"{p1(p.get('dma200_gap'), 1, True)} / {p1(p.get('dist_52w_high'), 1, True)}"),
            ("Max drawdown (3y)", p1(p.get("maxdd_3y"))),
            ("Days to exit @10% / @20% ADTV", f"{num(L['DaysToExit_10pct'], 1)} / {num(L['DaysToExit_20pct'], 1)}" if L else "&ndash;")
        ]


        kvh = "<dl class='kv'>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in kv) + "</dl>"
        sig = "".join(f'<span class="tag bull">{esc(x)}</span>' for x in str(r.Bull_Signals).split("; ") if x.strip()) + "".join(f'<span class="tag bear">{esc(x)}</span>' for x in str(r.Bear_Signals).split("; ") if x.strip())
        change_tags = ""
        if not pd.isna(r.Score_Delta):
            if r.Score_Delta > 0.05:
                change_tags += f"<span class='badge bad'>Score ↑ {r.Score_Delta:.2f}</span> "
            elif r.Score_Delta < -0.05:
                change_tags += f"<span class='badge ok'>Score ↓ {abs(r.Score_Delta):.2f}</span> "
        if r.Label_Changed:
            change_tags += f"<span class='badge warn'>Label: {esc(r.Prev_Label)} &rarr; {esc(r.FinalLabel)}</span> "
        is_open = " open" if any(x.startswith("HARD") for x in risk) else ""
        change_tags = ""
        if not pd.isna(r.Score_Delta):
            if r.Score_Delta > 0.05:
                change_tags += f"<span class='badge bad'>Score ↑ {r.Score_Delta:.2f}</span> "
            elif r.Score_Delta < -0.05:
                change_tags += f"<span class='badge ok'>Score ↓ {abs(r.Score_Delta):.2f}</span> "
        if r.Label_Changed:
            change_tags += f"<span class='badge warn'>Label: {esc(r.Prev_Label)} &rarr; {esc(r.FinalLabel)}</span> "

        is_open = " open" if any(x.startswith("HARD") for x in risk) else ""
        cards.append(
            f'<details class="hold" id="h-{esc(s)}"{is_open}><summary>'
            f'<span class="sym" style="font-size:16px">{esc(s)}</span>'
            f'{badge(r.Classification)}'
            f'{badge(r.FinalLabel) if r.FinalLabel != r.Classification else ""}'
            f'{change_tags}'
            f'<span class="muted">weight {r.Weight:.1f}% &middot; score {num(r.Weighted)} '
            f'&middot; coverage {r.Coverage * 100:.0f}% &middot; confidence {esc(r.Confidence)}</span>'
            f'</summary><div class="body">'
            f'<div style="margin-top:12px">{tags or "<span class=muted>No review reasons raised.</span>"}</div>'
            f'<div class="cols"><div><h3>Key metrics</h3>{kvh}'
            f'<h3>Computed signals</h3>{sig or "<span class=muted>none</span>"}</div>'
            f'<div><h3>Monitoring triggers</h3>{trg}</div></div>'
            f'<h3>Evidence behind each score</h3>{ev_tbl}'
            f'<p class="muted" style="margin:8px 0 0">Confidence: {esc(r.ConfidenceReason)}</p>'
            f'</div></details>')
    CO = G("Concentration")
    def weight_bars(dim, cls=""):
        d = CO[CO.Dimension == dim] if len(CO) else pd.DataFrame()
        if not len(d): return "<p class='muted'>n/a</p>"
        mx = max(d.Weight.max(), 1)
        return "".join(f'<div class="row"><div class="lab" title="{esc(b.Bucket)}">{esc(b.Bucket)}</div>{bar(b.Weight, mx, cls)}<div class="num">{b.Weight:.1f}%</div></div>' for b in d.itertuples())
    # Check concentration limits
    top1_wt = port.get('Top-1 weight % (of risk assets)', 0)
    top5_wt = port.get('Top-5 weight % (of risk assets)', 0)
    top1_flag = " <span class='badge bad'>ABOVE MODEL LIMIT</span>" if top1_wt > CFG["CONC_SINGLE_PCT"] else ""
    top5_flag = " <span class='badge bad'>ABOVE MODEL LIMIT</span>" if top5_wt > CFG["CONC_TOP5_PCT"] else ""

    conc_kv = [
        ("Top-1 weight", f"{num(top1_wt, 1)}%{top1_flag}"),
        ("Top-5 weight", f"{num(top5_wt, 1)}%{top5_flag}"),
        ("Top-10 weight", f"{num(port.get('Top-10 weight % (of risk assets)'), 1)}%"),
        ("HHI (Capital)", num(port.get("HHI (risk-asset weights)"), 3)),
        ("Effective positions (Capital)", num(port.get("Effective number of positions (1/HHI)"), 1)),
        ("Effective risk contributors", num(port.get("Effective number of risk contributors"), 1)),
        ("PC1 share of variance", p1(port.get("PC1 share of return variance (single-factor dominance)"), 0)),
        ("Avg pairwise correlation", num(port.get("Average pairwise correlation"))),
        ("Cash / bonds", f"{num(port.get('Cash/bond weight % of total'), 1)}%"),
    ]
    h_rows = []
    for r in HD.reset_index().itertuples():
        dev = "" if pd.isna(r.Target) else f'<span class="{"neg" if r.OutOfBand else "muted"}">{r.TargetDev_pp:+.1f}pp</span>'
        h_rows.append([td(f'<span class="sym">{esc(r.Symbol)}</span>'), td(esc(r.AssetType)), td(esc(r.Sector) or "&ndash;"), td(f"{r.Qty:,.0f}", "r"), td(num(r.AvgCost), "r"), td(num(r.CMP), "r"), td(inr(r.Value) if r.Priced else "<span class='muted'>no price</span>", "r"), td((f'<span class="{tone(r.PnLPct)}">{r.PnLPct:+.1f}%</span>' + (f" <span class='badge bad' title='{esc(r.CostFlag)}'>unreconciled</span>" if r.CostFlag else "")) if not pd.isna(r.PnLPct) else "<span class='muted'>n/a</span>", "r"), td(f"<b>{num(r.UserXIRR, 1, '%')}</b>" if "UserXIRR" in r and not pd.isna(r.UserXIRR) else "&ndash;", "r"), td(f'<div class="ws"><b style="min-width:42px">{r.Weight:.1f}%</b>{bar(r.Weight, HD.Weight.max())}</div>'), td(dev, "r"), td(esc(r.Surveillance) or "&ndash;")])
    h_tbl = table([("Holding", ""), ("Type", ""), ("Sector", ""), ("Qty", "r"), ("Avg cost", "r"), ("CMP", "r"), ("Value", "r"), ("P&amp;L", "r"), ("Your XIRR", "r"), ("Weight", ""), ("vs target", "r"), ("Surveillance", "")], h_rows)
    growth = ""
    if len(pr) > 252:
        pw, bw = 100 * (1 + pr).cumprod(), 100 * (1 + bm_r.fillna(0)).cumprod()
        growth = '<div class="legend"><span><i style="background:var(--accent)"></i>Portfolio at today\'s weights (back-cast)</span><span><i style="background:var(--muted)"></i>Nifty 500 total return</span></div>' + svg_growth([("Portfolio", pw, "a"), ("Nifty 500", bw, "b")])
    pf_rows = []
    for s in HD.index:
        p = perf.get(s)
        if not p: continue
        pf_rows.append([td(f'<span class="sym">{esc(s)}</span>'), td(num(p.get("hist_years"), 1), "r"), td(p1(p.get("tr_1y")), "r"), td(p1(p.get("tr_3y")), "r"), td(p1(p.get("tr_5y")), "r"), td(f'<span class="{tone(p.get("excess_3y"))}">{p1(p.get("excess_3y"), 1, True)}</span>', "r"), td(divbar(p.get("excess_3y"))), td(f'<span class="{tone(p.get("excess_5y"))}">{p1(p.get("excess_5y"), 1, True)}</span>', "r"), td(f'{p1(p.get("alpha_ann"), 1, True)} <span class="muted">(t {num(p.get("alpha_t"), 1)})</span>', "r"), td(num(p.get("beta")), "r")])
    pf_tbl = table([("Holding", ""), ("History (y)", "r"), ("TR 1y", "r"), ("TR 3y", "r"), ("TR 5y", "r"), ("Excess 3y", "r"), ("", ""), ("Excess 5y", "r"), ("Alpha p.a.", "r"), ("Beta", "r")], pf_rows)
    lq_rows = []
    LQ = G("Liquidity")
    for r in (LQ.itertuples() if len(LQ) else []):
        flag = '<span class="badge bad">&gt; %d days</span>' % CFG["EXIT_DAYS_FLAG"] if r.Exit_flag else '<span class="badge ok">ok</span>'
        lq_rows.append([td(f'<span class="sym">{esc(r.Symbol)}</span>'), td(inr_short(r.ADTV_used), "r"), td(num(r.DaysToExit_10pct, 1), "r"), td(num(r.DaysToExit_20pct, 1), "r"), td(flag), td(esc(r.Surveillance) or "&ndash;")])
    lq_tbl = table([("Holding", ""), ("Traded value/day", "r"), ("Days to exit @10%", "r"), ("@20%", "r"), ("Flag", ""), ("Surveillance", "")], lq_rows) if lq_rows else "<p class='muted'>n/a</p>"
    GP, MT = G("Data_Gaps"), G("Method")
    RG = G("Evidence_Register")
    register_html = (table([("Holding", ""), ("Flag", ""), ("Observed", ""), ("Threshold", ""), ("Source", ""), ("Date", ""), ("Status", ""), ("Effect", "")],
                           [[td(f'<span class="sym">{esc(r.Symbol)}</span>'), td(esc(r.FlagType)), td(esc(r.Observed)), td(esc(r.Threshold)),
                             td(f'<a href="{esc(r.SourceLink)}" target="_blank" rel="noopener">{esc(r.Source)}</a>' if r.SourceLink else esc(r.Source)),
                             td(esc(r.SourceDate)), td(esc(r.Status)), td(esc(r.Effect))] for r in RG.itertuples()])
                     if len(RG) else "<p>No flags raised.</p>")
    gaps_html = "<ul class='clean'>" + "".join(f"<li><b>{esc(x[0])}</b> &mdash; {esc(x[1])}: {esc(x[2])}</li>" for x in GP.itertuples(index=False)) + "</ul>" if len(GP) else "<p>None recorded.</p>"
    meth_html = table([("Parameter", ""), ("Value", "")], [[td(esc(x[0])), td(esc(x[1]))] for x in MT.itertuples(index=False)]) if len(MT) else ""
    # --- RISK HTML GENERATION ---
    risk_html = "<p class='muted'>Not enough overlapping history.</p>"
    if len(RC):
        d = RC.sort_values("RiskContribution", ascending=False).head(15)
        mx = max(d.RiskContribution.max(), d.Weight.max(), 1)
        risk_html = (
            "<p class='muted' style='margin-bottom:10px;'>This compares how much <b>capital</b> you have in a stock (blue) vs how much <b>portfolio volatility</b> it generates (orange).</p>"
            '<div class="legend"><span><i style="background:var(--accent)"></i>Capital Weight</span>'
            '<span><i style="background:var(--warn)"></i>Risk Contribution</span></div>'
            + "".join(
                f'<div class="pair"><div class="lab" title="{esc(i)}">{esc(i)}</div>'
                f'<div class="bars">{bar(x.Weight, mx)}{bar(x.RiskContribution, mx, "alt")}</div>'
                f'<div class="num muted" style="text-align:right">{x.RiskContribution:.0f}%</div></div>'
                for i, x in d.iterrows()))

    CR = G("Correlation")
    heat = ""
    if len(CR) and "Symbol" in CR:
        names = [s for s in HD.sort_values("Value", ascending=False).index if s in CR.columns][:14]
        if len(names) >= 3:
            m = CR.set_index("Symbol")
            hr = []
            for a in names:
                cells = [f"<th>{esc(a)}</th>"]
                for b in names:
                    v = m.loc[a, b]
                    hue, al = (210, abs(v) * .75) if v >= 0 else (0, abs(v) * .75)
                    cells.append(f'<td style="background:hsla({hue},75%,52%,{al:.2f})" title="{esc(a)} / {esc(b)}: {v:.2f}">{v:.2f}</td>')
                hr.append("".join(cells))
            heat = ('<h3>Correlation (daily TR, last 2y)</h3><div class="scroll">'
                    '<table class="heat"><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table></div>'
                    % ("".join(f"<th>{esc(n)}</th>" for n in names), "".join(f"<tr>{r_}</tr>" for r_ in hr)))
            
    if conc_pairs:
        heat += f"<h3>Highest Correlated Pairs</h3><p class='muted' style='font-size:13px;'>{esc('; '.join(conc_pairs))}</p>"
    demo = '<div class="demo">SYNTHETIC DEMO DATA</div>' if meta.get("demo") else ""
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in [("summary", "Summary"), ("score", "Scorecard"), ("holdings", "Holdings"), ("allocation", "Allocation"), ("perf", "Performance"), ("liq", "Liquidity"), ("gaps", "Data gaps"), ("register", "Evidence register"), ("method", "Method")])
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Portfolio V4 &middot; {stamp}</title><style>{_CSS}</style></head><body>
<nav class="top"><div class="in">{nav}</div></nav><div class="wrap"><header class="hero"><h1>Portfolio Diagnostic V4</h1><div class="sub">Prices as of {asof} &middot; generated {stamp} &middot; data: {esc(meta['source'])} &middot; benchmark: {esc(meta['bench_src'])}</div>{demo}</header>
<div class="kpis">{kpis}</div><section class="card" id="summary"><h2>Executive summary</h2><p class="lead">Research and decision support only. Scores are relative ranking tools.</p>{summ}</section>
<section class="card" id="score"><h2>Scorecard</h2><p class="lead">Worst-scoring first. {fnames}</p>{legend}{sc_tbl}<h3>Holding detail</h3>{''.join(cards)}</section>
<section class="card" id="holdings"><h2>Holdings</h2><p class="lead">Weights are current market value.</p>{h_tbl}</section>
<section class="card" id="allocation"><h2>Allocation &amp; concentration</h2><div class="grid2"><div><h3>Sector (equities)</h3>{weight_bars("Sector")}</div><div><h3>Cap tier / asset type</h3>{weight_bars("Cap tier / asset type", "alt")}</div></div><h3>Concentration measures</h3><dl class="kv" style="max-width:520px">{''.join(f'<dt>{k}</dt><dd>{v}</dd>' for k, v in conc_kv)}</dl></section>
<section class="card" id="perf"><h2>Performance</h2><p class="lead">Total return vs benchmark.</p>{growth}<h3>By holding</h3>{pf_tbl}</section>
<section class="card" id="liq"><h2>Liquidity</h2><p class="lead">Days to exit assume 10&ndash;20% of ADTV.</p>{lq_tbl}</section>
<section class="card" id="gaps"><h2>Data gaps &amp; assumptions</h2><p class="lead">Absence of data is never treated as evidence of quality.</p>{gaps_html}</section>
<section class="card" id="register"><h2>Evidence register</h2><p class="lead">Every flag the report raises, with its source and status. Automated flags are not independently verified.</p>{register_html}</section>
<section class="card" id="method"><h2>Method &amp; configuration</h2><details><summary style="cursor:pointer">Show all parameters</summary><div style="margin-top:12px">{meth_html}</div></details></section>
<footer>Not investment advice. Verify anything material against exchange filings. Generated locally by portfolio_diagnostic.py v{VERSION}.</footer></div></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path

# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Indian equity portfolio diagnostic (local)")
    ap.add_argument("--holdings", default="holdings.csv", help="CSV/XLSX one row per lot")
    ap.add_argument("--evidence", default="evidence.csv", help="optional CSV with judgment inputs")
    ap.add_argument("--peers", default="peers.csv", help="optional CSV (symbol, peer)")
    ap.add_argument("--outdir", default="./diagnostic_output", help="output directory")
    ap.add_argument("--review-date", help="YYYY-MM-DD for holding-period/tax")
    ap.add_argument("--rf", type=float, help="annual risk-free rate as decimal")
    ap.add_argument("--tax-estimate", action="store_true", help="add INDICATIVE tax estimate")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not auto-open the HTML")
    ap.add_argument("--no-fetch", action="store_true", help="skip auto-fetching fresh evidence (uses existing evidence.csv)")
    ap.add_argument("--refresh-evidence", action="store_true", help="re-fetch NSE evidence for every symbol now (default: only symbols older than 7 days)")
    ap.add_argument("--no-prompt", action="store_true", help="never open the pop-up for missing buy dates / promoter holding / peers")
    ap.add_argument("--prompt", action="store_true", help="ask again for items you previously chose not to be asked about")
    ap.add_argument("--init", action="store_true", help="write template input CSVs and exit")
    ap.add_argument("--demo", action="store_true", help="offline run on SYNTHETIC data")
    a = ap.parse_args()
    
    # By default, fetch evidence unless --no-fetch is specified
    a.fetch_evidence = not a.no_fetch
    
    if a.init:
        write_templates(Path(a.outdir))
        return
    run(a)

if __name__ == "__main__":
    import traceback
    if len(sys.argv) == 1:
        here = Path(__file__).resolve().parent
        found = next((p for p in [here / "holdings.csv", here / "diagnostic_output" / "holdings.csv", Path.cwd() / "holdings.csv"] if p.exists()), None)
        if found:
            print(f"(No CLI args - auto-using {found})")
            sys.argv.extend(["--holdings", str(found)])
            for k, fn in [("evidence", "evidence.csv"), ("peers", "peers.csv")]:
                p = found.parent / fn
                if p.exists(): sys.argv.extend([f"--{k}", str(p)])
        else:
            print("(No CLI args, no holdings.csv found - running --demo so you can see a report.)")
            sys.argv.append("--demo")
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n" + "=" * 78 + "\nTHE SCRIPT CRASHED - full traceback below:\n" + "=" * 78)
        traceback.print_exc()
        sys.exit(1)
