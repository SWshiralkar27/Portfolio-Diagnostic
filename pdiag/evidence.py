"""NSE evidence fetch (promoter holding, disclosures, results date) and its rules."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from pdiag.config import EVIDENCE_MAX_AGE_DAYS, EVIDENCE_SCHEMA, AUTO_FIELDS, MANUAL_FALLBACK
from pdiag.utils import has
try:
    import dalal
    HAS_DALAL = True
except ImportError:
    HAS_DALAL = False


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

def fetch_fresh_evidence(symbols, progress=None):
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
        if progress: progress(i + 1, len(symbols), sym)
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
