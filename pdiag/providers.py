"""Yahoo and NSE data providers (the cache folder is passed in)."""
from __future__ import annotations
import datetime as dt
import time
from pathlib import Path
import pandas as pd
from pdiag.errors import FatalInputError
from pdiag.utils import bare, clean_hist
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class YahooProvider:
    name = "Yahoo Finance"
    def __init__(self, cache_dir: Path, use_cache=True):
        try: import yfinance as yf
        except ImportError: raise FatalInputError("yfinance not installed.")
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
