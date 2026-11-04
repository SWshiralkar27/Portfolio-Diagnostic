"""App-mode wrapper around a data provider (Step 28b).

Yahoo can refuse company info (sector, market cap, P/E, P/B ...) when many requests come from one shared server. Before this
wrapper the refusal was silent: sector, cap tier and valuation simply went blank for every stock. The wrapper:
  * paces the company-info calls (INFO_PACE seconds apart) and stops asking after BREAKER failures in a row, so a refusing
    Yahoo cannot make a run hang;
  * keeps the last good company info per stock (it travels in the carry-over pack) and uses it when today's call fails,
    labelled with its age; a copy older than STALE_DAYS only gives the sector (valuation and market cap would be out of date);
  * reports what happened (info_notes -> Data_Gaps, info_reasons -> the STATUS line), including the error Yahoo gave.
Everything else (prices, statements, NSE lists) passes straight through to the wrapped provider."""
from __future__ import annotations
from collections import Counter
import pandas as pd

INFO_KEYS = ("sector", "marketCap", "trailingPE", "priceToBook", "returnOnEquity", "enterpriseToEbitda", "enterpriseToRevenue", "trailingEps")
KEY_FIELDS = ("marketCap", "trailingPE", "priceToBook", "sector")      # none of these present = "not returned"
INFO_PACE = 0.25
STALE_DAYS = 30
BREAKER = 8


def _ok(v):
    return v is not None and not (isinstance(v, float) and v != v) and str(v).strip() != ""


class ResilientProvider:
    def __init__(self, inner, stored=None, pace=INFO_PACE, breaker=BREAKER, today=None):
        self.inner = inner
        self.stored = {k: dict(v) for k, v in (stored or {}).items()}
        self.pace, self.breaker = pace, breaker
        self.today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
        self.status, self.errors, self.ages = {}, Counter(), []
        self._run, self._streak, self._tripped, self._diagnosed = {}, 0, False, 0

    def __getattr__(self, name):                 # history, statements, surveillance, corporate_actions, nse, name ...
        if name == "inner": raise AttributeError(name)
        return getattr(self.inner, name)

    # -- company info -------------------------------------------------------------------------
    def _diagnose(self, tkr, got):
        """Record why Yahoo gave nothing (only for the first few failures, to keep the run short)."""
        self._diagnosed += 1
        if self._diagnosed > 3: return
        try:
            yf = self.inner.yahoo.yf
        except Exception:
            self.errors[f"empty answer ({len(got)} fields)"] += 1; return
        try:
            d = dict(yf.Ticker(tkr).info or {})
            self.errors[f"answered with {len(d)} fields but none of {', '.join(KEY_FIELDS)}"] += 1
        except Exception as e:
            self.errors[f"{type(e).__name__}: {str(e)[:90]}"] += 1

    def info(self, tkr):
        if tkr in self._run: return dict(self._run[tkr])
        got, fresh = {}, False
        if not self._tripped:
            if self.pace:
                import time; time.sleep(self.pace)
            try: got = dict(self.inner.info(tkr) or {})
            except Exception as e: got = {}; self.errors[f"{type(e).__name__}: {str(e)[:90]}"] += 1
            fresh = any(_ok(got.get(k)) for k in KEY_FIELDS)
            if fresh: self._streak = 0
            else:
                self._streak += 1; self._diagnose(tkr, got)
                if self._streak >= self.breaker: self._tripped = True
        if fresh:
            self.stored[tkr] = dict(fetched=self.today.strftime("%Y-%m-%d"), info={k: got[k] for k in INFO_KEYS if _ok(got.get(k))})
            self.status[tkr], result = "fresh", got
        else:
            rec = self.stored.get(tkr)
            if rec and rec.get("info"):
                age = (self.today - pd.Timestamp(rec["fetched"])).days
                self.ages.append(age)
                if age <= STALE_DAYS: self.status[tkr], result = "saved", dict(rec["info"])
                else: self.status[tkr], result = "saved-sector-only", {k: v for k, v in rec["info"].items() if k == "sector"}
            else:
                self.status[tkr], result = "missing", {}
        self._run[tkr] = result
        return dict(result)

    # -- what the run reports ---------------------------------------------------------------------
    def company_info_store(self):
        return self.stored

    def info_notes(self):
        n, total, notes = Counter(self.status.values()), len(self.status), []
        why = "; ".join(f"{k} (x{v})" for k, v in self.errors.most_common(3))
        if n["missing"]:
            notes.append(f"Company info (sector, cap tier, valuation inputs) was NOT returned by Yahoo for {n['missing']} of {total} lookups and no saved copy exists: "
                         f"those stocks have a blank sector and cap tier and an unscored valuation (F5)." + (f" Yahoo's answer: {why}." if why else ""))
        saved = n["saved"] + n["saved-sector-only"]
        if saved:
            notes.append(f"Company info for {saved} lookups was not returned today and comes from your carry-over pack instead (age {min(self.ages)}-{max(self.ages)} days)"
                         + (f"; {n['saved-sector-only']} of them are older than {STALE_DAYS} days, so only the sector was reused" if n["saved-sector-only"] else "")
                         + ". Valuation and market cap are as of that earlier date." + (f" Yahoo's answer today: {why}." if why else ""))
        if self._tripped:
            notes.append(f"Yahoo kept refusing company info, so after {self.breaker} failures in a row the app stopped asking for the rest of this run.")
        return notes

    def info_reasons(self):
        n, total = Counter(self.status.values()), len(self.status)
        out = []
        if n["missing"]: out.append(f"company info missing for {n['missing']} of {total} lookups (sector, cap tier and valuation unavailable for them)")
        if n["saved"] + n["saved-sector-only"]: out.append(f"company info for {n['saved'] + n['saved-sector-only']} lookups taken from your carry-over pack, not fetched today")
        return out
