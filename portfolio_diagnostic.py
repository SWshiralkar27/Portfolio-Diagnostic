#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_diagnostic.py - Indian listed-equity portfolio diagnostic (V4 Integrated) - command-line / IDLE entry point.

Step 27: the code now lives in the pdiag package. This file reads your files from disk, runs the pdiag.pipeline stages
with the Tkinter pop-ups (input_assistant.py), and writes the HTML and Excel next to them: the same behaviour as the
single-file version, which is kept as legacy/portfolio_diagnostic_v4_monolith.py. Every name that used to be defined
here is re-exported below, so scripts that import portfolio_diagnostic keep working.
"""
from __future__ import annotations
import argparse, datetime as dt, sys, webbrowser
from pathlib import Path
import pandas as pd

if sys.version_info < (3, 11):
    sys.exit("ERROR: Python 3.11 or newer is required for the dalal scraper. Please upgrade Python.")

try:
    from pdiag.errors import FatalInputError
    from pdiag.engine_link import scoring, earnings_engine, ownership_engine, catalyst_engine, leadership_engine, FACTORS, WEIGHTS, FACTOR_NAMES
    from pdiag.config import (
        VERSION, CFG, BENCH_CANDIDATES, CAP_TO_BENCH, FIN_TYPES, NO_STD_TYPES, CONF_RANK, HOLD_ATTRS, EVIDENCE_MAX_AGE_DAYS, EVIDENCE_SCHEMA, AUTO_FIELDS, MANUAL_FALLBACK, SCORABLE)
    from pdiag.utils import (
        fnum, has, yes, bucket, pct, to_ticker, bare, hist_years, cagr, total_return_since, approx_tr, max_drawdown, monthly, xirr, capm, capture, clean_hist)
    from pdiag.providers import (
        YahooProvider, NSEProvider, CompositeProvider)
    from pdiag.evidence import (
        evidence_is_stale, merge_evidence, classify_disclosure, fetch_fresh_evidence)
    from pdiag.fundamentals import (
        _row, _cagr_series, fundamentals, val_metric)
    from pdiag.loaders import (
        read_table, load_holdings, holdings_from_df, load_evidence, evidence_from_df, load_peers, peers_from_df)
    from pdiag.factors import (
        score_stock, weighted, schemes, classify)
    from pdiag.actions import (
        build_actions)
    from pdiag.register import (
        build_evidence_register)
    from pdiag.report_html import (
        BADGE_CLS, _CSS, esc, inr, inr_short, p1, num, tone, chip, badge, td, th, table, bar, divbar, svg_growth, render_html, write_html, render_result_html)
    from pdiag.report_excel import (
        build_xlsx_bytes, write_xlsx)
    from pdiag.hooks import (
        NullHooks, TkHooks)
    from pdiag.state import (
        DiskState, MemoryState)
    from pdiag.pipeline import (
        _no_progress, RunOptions, RunContext, RunResult, new_context, stage_evidence, stage_prices, apply_cost_review, stage_analysis, run_staged)
    import pdiag.evidence as _ev
    import pdiag.providers as _pv
except FatalInputError as _e:
    sys.exit(str(_e))
HAS_DALAL, dalal = _ev.HAS_DALAL, getattr(_ev, "dalal", None)
HAS_REQUESTS = _pv.HAS_REQUESTS
try:
    import input_assistant          # optional pop-up for buy dates / promoter holding / peers (see input_assistant.py)
except ImportError:
    input_assistant = None


def _read_verified(args):
    try:
        _vp = Path(args.evidence).parent / "verified_flags.csv"
        return pd.read_csv(_vp) if _vp.exists() else None
    except Exception:
        return None


def run(args, provider_factory=None):
    """Same steps, same order and same files as the single-file run(); the work itself is done by the pdiag stages."""
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    provider = (provider_factory or CompositeProvider)(out / ".cache", use_cache=not args.no_cache)
    lots = load_holdings(args.holdings)
    _dir = Path(args.holdings).resolve().parent
    for _a, _f in (("evidence", "evidence.csv"), ("peers", "peers.csv")):
        if not Path(getattr(args, _a)).exists() and (_dir / _f).exists():
            setattr(args, _a, str(_dir / _f))
    hooks = TkHooks(input_assistant, args, _dir) if (input_assistant and not args.demo) else NullHooks()
    state = DiskState(out, evidence_label=args.evidence)
    options = RunOptions(fetch_evidence=args.fetch_evidence, refresh_evidence=bool(getattr(args, "refresh_evidence", False)),
                         demo=args.demo, review_date=args.review_date)

    lots = hooks.after_load(lots)
    ctx = new_context(lots, provider, state, manual_evid=load_evidence(args.evidence), options=options)
    stage_evidence(ctx)
    ctx.peers = load_peers(args.peers)
    ctx.lots, ctx.evid, ctx.peers = hooks.prompts(ctx.lots, ctx.evid, ctx.peers)
    stage_prices(ctx)
    apply_cost_review(ctx, hooks)
    ctx.verified = _read_verified(args)
    res = stage_analysis(ctx)

    html_path = write_html(out, res.sheets, res.meta, pr=res.pr, bm_r=res.bm_r, port=res.port, port_x=res.port_x, summary=res.summary,
                           conc_pairs=[], fps=res.sheets["Scorecard_A"].head(0), RC=pd.DataFrame(), liq=res.liq, tax_est=None)
    print("Generating Excel workbook with all data sheets...")
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    write_xlsx(out / f"portfolio_diagnostic_{stamp}.xlsx", res.sheets)

    print(f"\nHTML report: {html_path}")
    if not args.no_open:
        try: webbrowser.open(html_path.resolve().as_uri())
        except Exception as e: print(f"Could not open browser: {e}")
    return res.sheets


# =============================================================================
# CLI
# =============================================================================
def write_templates(outdir):
    if input_assistant is None:
        sys.exit("input_assistant.py not found next to portfolio_diagnostic.py")
    input_assistant.write_templates(outdir)


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
    try:
        run(a)
    except FatalInputError as e:
        sys.exit(str(e))

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
