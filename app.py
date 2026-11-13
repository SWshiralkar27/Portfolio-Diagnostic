"""Portfolio Diagnostic - Streamlit page (Step 28).
Your files are read into this session's memory, used for the run and never written to disk or to GitHub. At the end you download
the report, the Excel workbook and a carry-over pack; upload the pack at your next run to skip work that is already done.
Decision support only; not investment advice."""
from __future__ import annotations
import datetime as dt
import io
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Portfolio Diagnostic", layout="wide")
try:
    from pdiag import appcore as ac
    from pdiag.config import CFG, VERSION
    from pdiag.errors import FatalInputError
    from pdiag.loaders import read_table, holdings_from_df
    from pdiag.pipeline import RunOptions, new_context, stage_evidence, stage_prices, apply_cost_review, stage_analysis
    from pdiag.providers import CompositeProvider
    from pdiag.resilient import ResilientProvider
    from pdiag.state import MemoryState
except Exception as e:                                   # a library or a file is missing: say so plainly
    st.error(f"The app could not load its code ({type(e).__name__}: {e}). Check that the pdiag folder and the engine files are in the repository.")
    st.stop()

S = st.session_state
S.setdefault("stage", "upload")


def named(name, data):
    f = io.BytesIO(data); f.name = name
    return f


def pick(key, uploaded, fname):
    """The uploaded file, or (tests only) bytes placed in session state."""
    if uploaded is not None: return uploaded
    raw = S.get("_test_files", {}).get(key)
    return named(fname, raw) if raw is not None else None


def make_provider(stored):
    """Yahoo + NSE, wrapped so a refusal is reported and the last good company info (from your pack) fills the gap."""
    if S.get("_provider_factory"):
        inner = S["_provider_factory"]()
    else:
        tmp = Path(tempfile.mkdtemp(prefix="pdiag_"))
        S["_tmp"] = str(tmp)
        inner = CompositeProvider(tmp / ".cache", use_cache=False)      # market data only; nothing of yours is cached
        inner.nse.STRICT = True                                         # a refused NSE site is reported, not read as "clear"
    return ResilientProvider(inner, stored=stored, pace=S.get("_pace", 0.25))


def wipe():
    t = S.pop("_tmp", None)
    if t: shutil.rmtree(t, ignore_errors=True)
    for k in ("ctx", "pack", "n_saved", "n_est", "out", "shown_cost", "snap_b"): S.pop(k, None)


def read_as(label, fn, *a):
    """Run one file read; any problem is reported with the name of the file it came from."""
    try: return fn(*a)
    except FatalInputError as e: raise FatalInputError(f"{label}: {e}")
    except Exception as e: raise FatalInputError(f"{label} could not be read ({type(e).__name__}: {e}).")


def fail(where, e):
    st.error(str(e) if isinstance(e, FatalInputError) else f"{where} failed: {type(e).__name__}: {e}")
    st.stop()


st.title("Portfolio Diagnostic")
st.caption(ac.LABEL)
with st.expander("What happens to your files"):
    st.markdown("- Your files are read into **this browser session's memory** on the host's server, used for the run, and **not saved**; nothing goes to GitHub.\n"
                "- At the end you download the report, the Excel workbook and a **carry-over pack**. Keep the pack on your own device; it holds your buy dates, cost decisions and list of stocks.\n"
                "- Market data comes from Yahoo Finance and NSE. Many people share one server, so a run can be slow at busy times.\n"
                f"- Engine version {VERSION}. Scores are relative ranking tools; they are not predictions and not advice.")

if S.stage == "upload":
    st.subheader("Step 1 of 3 - your files")
    c1, c2 = st.columns(2)
    with c1:
        f_hold = st.file_uploader("Holdings file (required): .csv or .xlsx, one row per buy lot", type=["csv", "xlsx", "xls"], key="up_hold")
        st.download_button("Download a blank holdings template", ac.HOLDINGS_TEMPLATE, "holdings_template.csv", "text/csv", key="dl_template", on_click="ignore")
        st.caption("Columns: symbol, quantity and buy_price are needed; buy_date, xirr (your broker's figure, in %) and target_weight_pct are optional. "
                   "Broker statement (.xlsx / PDF) upload is the next step; until then use this file.")
        f_pack = st.file_uploader("Carry-over pack from your last run (optional, .zip)", type=["zip"], key="up_pack")
    with c2:
        f_ev = st.file_uploader("evidence.csv (optional): pledge, auditor flags, GNPA / CAR ...", type=["csv", "xlsx"], key="up_ev")
        f_peers = st.file_uploader("peers.csv (optional): columns symbol, peer", type=["csv", "xlsx"], key="up_peers")
        f_ver = st.file_uploader("verified_flags.csv (optional): flags you have checked", type=["csv", "xlsx"], key="up_ver")
    st.checkbox("Fetch fresh NSE evidence (promoter holding, disclosures, results dates)", value=True, key="opt_fetch")
    st.checkbox("Re-fetch NSE evidence for every stock, even if the pack has it (slower)", value=False, key="opt_force")
    if st.button("Start", type="primary", key="start"):
        hold = pick("holdings", f_hold, "holdings.csv")
        if hold is None:
            st.error("Please upload your holdings file first."); st.stop()
        try:
            lots = read_as("Holdings file", lambda: holdings_from_df(read_table(hold)))
            lots, n_est = ac.mark_file_dates(lots)
            pk = pick("pack", f_pack, "pack.zip")
            pack = read_as("Carry-over pack", ac.parse_pack, pk) if pk is not None else ac.empty_pack()
            lots, n_saved = ac.apply_saved_buy_dates(lots, pack["buy_dates"])
            ev, pf, vf = pick("evidence", f_ev, "evidence.csv"), pick("peers", f_peers, "peers.csv"), pick("verified", f_ver, "verified_flags.csv")
            peers, prep = read_as("Peers file", ac.read_peers, pf, set(lots.symbol)) if pf is not None else ({}, {})
            ctx = new_context(lots, make_provider(pack["company_info"]), MemoryState(auto_csv=pack["auto_csv"], snapshot_csv=pack["snapshot_csv"]),
                              manual_evid=read_as("Evidence file", ac.read_evidence, ev) if ev is not None else {},
                              peers=peers,
                              options=RunOptions(fetch_evidence=S.opt_fetch, refresh_evidence=S.opt_force, strict_nse=True, unknown_sector_guard=True, exact_dates_only=True, price_date_check=True, f8_redesign=True, confidence_ceiling=True, risk_contribution=True, f1_revenue_shadow=True),
                              verified=read_as("Verified-flags file", read_table, vf) if vf is not None else None)
            if prep and ac.peers_note(prep): ctx.notes_global.append(ac.peers_note(prep))
            bar = st.progress(0.0, text="Starting"); ctx.progress = ac.progress_to(bar)
            ac.prefetch_info(ctx)                               # company info first, before any other Yahoo request
            stage_evidence(ctx)
        except Exception as e:
            wipe(); fail("Reading your files", e)
        S.ctx, S.pack, S.n_saved, S.n_est, S.stage = ctx, pack, n_saved, n_est, "form_a"
        st.rerun()

elif S.stage == "form_a":
    ctx = S.ctx
    st.subheader("Step 2 of 3 - fill in what could not be fetched (all optional)")
    n_eq = int((~ctx.lots.asset_type.isin(["cash", "bond"])).groupby(ctx.lots.symbol).any().sum())
    if S.n_saved: st.info(f"{S.n_saved} buy date(s) were filled in from your carry-over pack.")
    if S.get("n_est"): st.info(f"{S.n_est} buy date(s) in your holdings file are treated as estimates: used for the holding period only, never for your own XIRR. If some are exact tradebook dates, add a column date_exact with yes for those rows.")
    und = ac.undated_lots(ctx.lots)
    ed_dates = und
    if len(und):
        st.markdown("**Buy dates.** Type a date for the holding period (long-term or not). Tick **Date is exact** only if it is the real purchase date from your tradebook: "
                    "your own XIRR is computed only from exact dates. An estimated date is used for the holding period only, and your broker's XIRR is shown instead. Nothing is guessed.")
        ed_dates = st.data_editor(und, key="ed_dates", hide_index=True, disabled=["lot", "symbol", "quantity", "buy_price"],
                                  column_order=["symbol", "quantity", "buy_price", "buy_date", "exact"],
                                  column_config={"buy_date": st.column_config.DateColumn("Buy date", format="YYYY-MM-DD"),
                                                 "exact": st.column_config.CheckboxColumn("Date is exact")})
    prom = ac.missing_promoter(ctx.evid, ctx.lots)
    ed_prom = prom
    if len(prom):
        st.markdown(f"**Promoter holding (%).** NSE gave no usable data for {len(prom)} of {n_eq} stocks. Type now and a year ago if you know them; blank stays unscored (never neutral).")
        ed_prom = st.data_editor(prom, key="ed_prom", hide_index=True, disabled=["symbol"],
                                 column_config={"promoter_holding_now": st.column_config.NumberColumn("Promoter % now", min_value=0.0, max_value=100.0),
                                                "promoter_holding_1y": st.column_config.NumberColumn("Promoter % a year ago", min_value=0.0, max_value=100.0)})
    if not len(und) and not len(prom): st.success("Nothing to fill in. Continue.")
    if st.button("Continue", type="primary", key="continue_a"):
        try:
            lots, _ = ac.apply_buy_date_edits(ctx.lots, ed_dates)
            S.pack["buy_dates"] = ac.merge_pack_dates(S.pack["buy_dates"], ctx.lots, ed_dates)
            ctx.lots = lots
            ctx.evid = ac.apply_promoter_edits(ctx.evid, ed_prom)
            bar = st.progress(0.0, text="Fetching prices"); ctx.progress = ac.progress_to(bar)
            stage_prices(ctx)
        except Exception as e:
            wipe(); fail("Fetching prices", e)
        S.stage = "form_b"
        st.rerun()

elif S.stage == "form_b":
    ctx = S.ctx
    if "snap_b" not in S: S.snap_b = ac.snapshot_ctx(ctx)
    st.subheader("Step 3 of 3 - check the cost of unusual lines (optional)")
    cand = ac.cost_candidates(ctx.HD, CFG, S.pack["cost_decisions"])
    ed_cost = cand
    if len(cand):
        st.markdown("Bonus shares, splits and demergers often break an average cost, and one wrong line can dominate your overall return. "
                    "Tick **Cost is correct** if you have checked it, or give a **correction factor** (new cost = old cost x factor, for example 0.5 after a 1:1 bonus). "
                    "Lines you leave alone stay flagged as unreconciled.")
        ed_cost = st.data_editor(cand, key="ed_cost", hide_index=True, disabled=["symbol", "avg_cost", "price", "pnl_pct", "flag"],
                                 column_config={"avg_cost": st.column_config.NumberColumn("Avg cost", format="%.2f"), "price": st.column_config.NumberColumn("Price", format="%.2f"),
                                                "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.0f"),
                                                "confirm": st.column_config.CheckboxColumn("Cost is correct"), "factor": st.column_config.NumberColumn("Correction factor", min_value=0.0, max_value=20.0)})
        prev = ac.cost_preview(ctx.HD, ed_cost, CFG)
        if len(prev):
            st.markdown("**What your correction would do** (check it before you run):")
            st.dataframe(prev, hide_index=True, column_config={"avg_cost_now": st.column_config.NumberColumn("Avg cost now", format="%.2f"), "avg_cost_after": st.column_config.NumberColumn("Avg cost after", format="%.2f"),
                                                                  "price": st.column_config.NumberColumn("Price", format="%.2f"), "pnl_pct_after": st.column_config.NumberColumn("P&L % after", format="%.0f")})
            if (prev.note != "ok").any(): st.warning("At least one correction still leaves an odd cost. New cost = old cost x factor, so a total entered as a price needs a factor below 1 (for example 0.001), not above 1.")
        st.caption("Press Enter or click outside the table cell before you press Run, and press Run only once.")
    else:
        st.success("No lines with an unusual cost.")
    if st.button("Run the analysis", type="primary", key="run_b"):
        try:
            confirmed, factors, decisions = ac.decisions_from_edits(ed_cost)
            bar = st.progress(0.0, text="Analysing"); ctx.progress = ac.progress_to(bar)
            ac.restore_ctx(ctx, S.snap_b)                  # safe if Run is pressed twice
            apply_cost_review(ctx, ac.FormHooks(confirmed, factors))
            res = stage_analysis(ctx)
            out = ac.package_outputs(res, ctx.state, S.pack, decisions, shown_cost_symbols=list(cand.symbol),
                                     company_info=getattr(ctx.provider, "company_info_store", lambda: {})())
        except Exception as e:
            wipe(); fail("The analysis", e)
        wipe()                                           # raw holdings and fetched data leave memory; only the outputs stay
        S.out, S.stage = out, "done"
        st.rerun()

else:
    out = S.out
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    st.success(f"Analysis complete ({out['n_sheets']} sheets). Download your files before leaving this page: nothing is stored.")
    c = st.columns(3)
    c[0].download_button("Report (HTML)", out["html"], f"portfolio_report_{stamp}.html", "text/html", key="dl_html", on_click="ignore")
    c[1].download_button("Workbook (Excel)", out["xlsx"], f"portfolio_diagnostic_{stamp}.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx", on_click="ignore")
    c[2].download_button("Carry-over pack (.zip) - keep on your device", out["pack"], f"carry_over_pack_{stamp}.zip", "application/zip", key="dl_pack", on_click="ignore")
    if "PROVISIONAL" in out.get("status", "") or out.get("notes"):
        with st.expander("Run notes: what could not be checked this time", expanded=True):
            if out.get("status"): st.warning(out["status"])
            for n in out.get("notes", []): st.write("- " + n)
    st.info("Upload the carry-over pack at your next run: NSE evidence already fetched, your buy dates, cost decisions and last good company info are then reused, so the run is faster and survives a Yahoo refusal.")
    if st.button("Start a new run", key="again"):
        wipe(); S.pop("out", None); S.stage = "upload"; st.rerun()
    st.iframe(out["html"], height=1100)           # the report is built by this app; text from NSE and your files is escaped inside it
