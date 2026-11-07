"""Factor scoring, weights and the label rule."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pdiag.config import CFG, FIN_TYPES, NO_STD_TYPES, SCORABLE
from pdiag.engine_link import scoring, earnings_engine, ownership_engine, catalyst_engine, leadership_engine, FACTORS, WEIGHTS, FACTOR_NAMES
from pdiag.utils import fnum, has, yes, bucket, pct


def score_stock(sector_type, ev, fund, perf, val_ratio, n_peers, val_label, f8_new=False):
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
    if f8_new:
        # F8 redesign (approved 6 Nov 2026): average of three sub-scores (1 best, 5 worst); at least two of the three are needed, otherwise unscored
        subs, txt = [], []
        if not pd.isna(dd):
            a_ = bucket(-dd, [0.25, 0.35, 0.45, 0.55], [1, 2, 3, 4, 5]); subs.append(a_); txt.append(f"3y max drawdown {-dd * 100:.0f}% -> {a_}")
        bt_ = perf.get("beta", np.nan)
        if not pd.isna(bt_):
            b_ = bucket(bt_, [0.8, 1.0, 1.2, 1.5], [1, 2, 3, 4, 5]); subs.append(b_); txt.append(f"beta {bt_:.2f} -> {b_}")
        iv_ = perf.get("idio_vol", np.nan)
        if not pd.isna(iv_):
            c_ = bucket(iv_, [0.20, 0.30, 0.40, 0.55], [1, 2, 3, 4, 5]); subs.append(c_); txt.append(f"idiosyncratic volatility {iv_ * 100:.0f}% -> {c_}")
        if len(subs) >= 2:
            S["F8"] = float(np.mean(subs)); E.append(("F8", S["F8"], "computed (F8 redesign)", "; ".join(txt)))
    elif not pd.isna(dd):
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
