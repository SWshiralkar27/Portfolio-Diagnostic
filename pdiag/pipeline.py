"""The run, cut into stages. A command-line run calls them one after another; a Streamlit page calls them one at a time and
shows its forms between them (storing the RunContext between page reruns).

  stage_evidence -> hooks.prompts -> stage_prices -> apply_cost_review -> stage_analysis -> RunResult

No stage reads a file by name, opens a pop-up or a browser, or ends the program: files go through `state`, questions through
`hooks`, progress through `ctx.progress(stage, i, n, label)`.
"""
from __future__ import annotations
import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from pdiag.config import (VERSION, CFG, BENCH_CANDIDATES, CAP_TO_BENCH, FIN_TYPES, NO_STD_TYPES, CONF_RANK, HOLD_ATTRS,
                          EVIDENCE_MAX_AGE_DAYS, SCORABLE)
from pdiag.engine_link import FACTORS, WEIGHTS, FACTOR_NAMES
from pdiag.errors import FatalInputError
from pdiag.utils import (fnum, has, yes, bucket, pct, to_ticker, bare, hist_years, cagr, total_return_since, approx_tr,
                         max_drawdown, monthly, xirr, capm, capture)
from pdiag.evidence import evidence_is_stale, merge_evidence, fetch_fresh_evidence
from pdiag.fundamentals import fundamentals, val_metric
from pdiag.factors import score_stock, weighted, schemes, classify
from pdiag.actions import build_actions
from pdiag.register import build_evidence_register


def _no_progress(stage, i, n, label):
    pass


@dataclass
class RunOptions:
    fetch_evidence: bool = True
    refresh_evidence: bool = False
    demo: bool = False
    review_date: Optional[str] = None
    strict_nse: bool = False      # app mode: say so in Data_Gaps when the NSE website could not be used
    price_date_check: bool = False      # app mode: note in Data_Gaps when the latest price is older than the latest NSE trading day
    exact_dates_only: bool = False      # app mode: own XIRR only from lots whose buy date you marked exact; adds the broker-XIRR coverage line
    unknown_sector_guard: bool = False      # app mode: no sector from you or Yahoo -> SectorType 'unknown' and a cash-flow flag is a risk tag, not a hard override


class RunContext:
    """Everything one run carries from stage to stage."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


@dataclass
class RunResult:
    sheets: dict
    meta: dict
    pr: object
    bm_r: object
    port: dict
    port_x: dict
    summary: list
    liq: dict


def new_context(lots, provider, state, manual_evid=None, peers=None, options=None, verified=None, progress=None):
    return RunContext(lots=lots, provider=provider, state=state, manual_evid=manual_evid or {}, peers=peers or {},
                      options=options or RunOptions(), verified=verified, progress=progress or _no_progress,
                      gaps=[], notes_global=[])


def stage_evidence(ctx):
    """Your evidence.csv (never rewritten) + the fetched evidence (refreshed per symbol)."""
    lots, state, options = ctx.lots, ctx.state, ctx.options
    auto_evid = state.load_auto()
    fetch_syms = (lots.loc[~lots["asset_type"].isin(["cash", "bond"]), "symbol"]
                  .dropna().astype(str).str.strip().str.upper().unique().tolist())
    if options.fetch_evidence and not options.demo:
        force = bool(options.refresh_evidence)
        stale = [s for s in fetch_syms if force or evidence_is_stale(auto_evid.get(s))]
        if stale:
            print(f"\nEvidence: refreshing {len(stale)} of {len(fetch_syms)} symbols from NSE "
                  f"(the rest are under {EVIDENCE_MAX_AGE_DAYS} days old; use --refresh-evidence to force all)")
            fresh = fetch_fresh_evidence(stale, progress=lambda i, n, s: ctx.progress("evidence", i, n, s))
            if fresh:
                auto_evid.update({s: {k: v for k, v in rec.items() if has(v)} for s, rec in fresh.items()})
                print(state.save_auto(auto_evid))
        else:
            print("\nEvidence: all symbols are fresh - skipping the NSE fetch.")
    ctx.auto_evid = auto_evid
    ctx.evid = merge_evidence(ctx.manual_evid, auto_evid)
    return ctx


def stage_prices(ctx):
    """Prices, benchmarks and the holdings table (value, cost, P&L, weight)."""
    provider, lots, gaps, options = ctx.provider, ctx.lots, ctx.gaps, ctx.options
    review_date = pd.Timestamp(options.review_date) if options.review_date else pd.Timestamp.today().normalize()

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
    _n_inst, _i_inst = sum(1 for s in tick.index if tick[s]), 0
    for s in tick.index:
        t = tick[s]
        if not t: continue
        _i_inst += 1; ctx.progress("prices", _i_inst, _n_inst, s)
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
    if broad is None: raise FatalInputError("No broad benchmark available.")

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
    if options.strict_nse and not getattr(getattr(provider, "nse", None), "available", True):
        ctx.notes_global.append("NSE ASM/GSM surveillance lists and NSE corporate actions were NOT checked: the NSE website refused this server's connection"
                                f" (HTTP {getattr(provider.nse, 'refused', None) or 'error'}). A blank Surveillance cell means 'not checked', not 'clear'.")
    if options.price_date_check and HD.PriceAsOf.notna().any():
        from pdiag.calendar_check import stale_price_note
        _sp = stale_price_note(HD.PriceAsOf.max(), pd.Timestamp.now(tz="Asia/Kolkata"))
        if _sp: ctx.notes_global.append(_sp)
    ctx.review_date, ctx.tick, ctx.H, ctx.B, ctx.broad, ctx.HD = review_date, tick, H, B, broad, HD
    return ctx


def apply_cost_review(ctx, hooks):
    """The owner's cost review (hook), then the cost-basis sanity check, targets and totals."""
    HD, lots, gaps = ctx.HD, ctx.lots, ctx.gaps
    HD, _cost_ok, _cost_factors = hooks.cost_review(HD, CFG)
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
    ctx.HD, ctx.total_value, ctx.equity_syms = HD, total_value, equity_syms
    return ctx


def stage_analysis(ctx):
    """Returns, risk, liquidity, fundamentals, scoring, portfolio metrics, actions, register and the 14 sheets."""
    provider, state, options = ctx.provider, ctx.state, ctx.options
    lots, HD, H, B, broad, tick = ctx.lots, ctx.HD, ctx.H, ctx.B, ctx.broad, ctx.tick
    equity_syms, total_value = ctx.equity_syms, ctx.total_value
    peers, evid, auto_evid, gaps, notes_global = ctx.peers, ctx.evid, ctx.auto_evid, ctx.gaps, ctx.notes_global
    review_date = ctx.review_date
    perf, fundd, liq, ca_rows, info_cache = {}, {}, {}, [], {}
    print("Computing returns, risk, liquidity and fundamentals ...")
    bench_win_start = pd.Timestamp.today() - pd.DateOffset(years=CFG["RISK_WINDOW_YEARS"])
    bench_maxdd_3y = max_drawdown(broad["tr"][broad["tr"].index >= bench_win_start])
    for _i_eq, s in enumerate(equity_syms, 1):
        ctx.progress("fundamentals", _i_eq, len(equity_syms), s)
        if s not in H: continue
        h, t = H[s], tick[s]
        info = provider.info(t) if HD.loc[s, "AssetType"] == "equity" else {}
        info_cache[s] = info
        if not HD.loc[s, "Sector"]: HD.loc[s, "Sector"] = info.get("sector", "") or ""
        stype = HD.loc[s, "SectorType"]
        if not stype:
            sec = str(info.get("sector", "")).lower()
            HD.loc[s, "SectorType"] = ("financial" if "financial" in sec else "cyclical" if sec in ("basic materials", "energy") else "unknown" if (options.unknown_sector_guard and not sec) else "standard")
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
        
    _n_unk = int((HD.SectorType == "unknown").sum())
    if _n_unk:
        notes_global.append(f"Sector unknown for {_n_unk} stocks (not in your file and Yahoo company info missing): a cash-flow flag on them is a risk tag, not a hard override, and financial-health (F6) uses industrial rules, which do not suit lenders. Give sector_type (bank, nbfc ...) in your holdings file or re-run when Yahoo answers.")
    def flows_for(g, s):
        fl = []; divs = H[s]["div"][H[s]["div"] > 0]
        for _, r in g.iterrows():
            if pd.isna(r.buy_date) or (options.exact_dates_only and pd.notna(r.get("date_exact", np.nan)) and not bool(r.get("date_exact"))): return None
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
    if state.has_snapshot():
        try:
            prev_scd = state.load_snapshot()
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
    state.save_snapshot(SCD[["Symbol", "Weighted", "FinalLabel", "Review_Risk"]])
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
    _ver = ctx.verified
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
    for r_ in getattr(provider, "info_reasons", lambda: [])(): reasons.append(r_)
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
    if options.exact_dates_only and HD.UserXIRR.notna().any():
        _bx = HD.UserXIRR.notna() & HD.Priced
        summary.append(f"Broker-reported XIRR is available for {int(_bx.sum())} of {int(HD.Priced.sum())} priced holdings ({HD.Value[_bx].sum() / HD.Value.sum() * 100:.1f}% of value). XIRRs are never averaged, so no portfolio XIRR is built from them; your own XIRR is computed only from lots whose buy date you marked as exact.")
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
    notes_global.extend(getattr(provider, "info_notes", lambda: [])())
    GAPS = pd.DataFrame(gaps + [("portfolio", "note", n) for n in notes_global], columns=["Item", "Missing/Issue", "Impact"])
    PORT = pd.DataFrame([(k, v) for k, v in port.items()], columns=["Metric", "Value"])
    METHOD = pd.DataFrame([(k, str(v)) for k, v in CFG.items()], columns=["Parameter", "Value"])
    SUMM = pd.DataFrame(dict(Executive_summary=summary))

    sheets = {"Summary": SUMM, "Portfolio": PORT, "Holdings": HOLD, "Scorecard_A": SCD, "Evidence": EVD, "Performance_Risk": PERF, "Liquidity": LQ, "Concentration": CONC, "Tax_Lots": pd.DataFrame(), "Corporate_Actions": CA, "Monitoring_Triggers": TRIG, "Data_Gaps": GAPS, "Evidence_Register": REG, "Method": METHOD}

    meta = dict(demo=options.demo, source=provider.name, bench_src=broad["src"])
    return RunResult(sheets=sheets, meta=meta, pr=pr, bm_r=bm_r, port=port, port_x=port_x, summary=summary, liq=liq)


def run_staged(ctx, hooks):
    """All stages in one call (no pauses between them other than the hooks)."""
    ctx.lots = hooks.after_load(ctx.lots)
    stage_evidence(ctx)
    ctx.lots, ctx.evid, ctx.peers = hooks.prompts(ctx.lots, ctx.evid, ctx.peers)
    stage_prices(ctx)
    apply_cost_review(ctx, hooks)
    return stage_analysis(ctx)
