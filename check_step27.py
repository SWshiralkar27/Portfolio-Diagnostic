"""check_step27.py - proves the modular version behaves exactly like the single-file version.

Run from IDLE (F5) in the project folder. Needs legacy/portfolio_diagnostic_v4_monolith.py (apply_step27.py puts it there).
Uses your real engine files if they are importable here; otherwise the stand-ins in tests/stub_engines (it says which).
Everything runs on SYNTHETIC data (tests/mock_world.py): no network, no real portfolio, no pop-ups, nothing outside a temp folder.

  0  the single-file version is exactly the file the modules were cut from, and moved functions are byte-for-byte the same
  1  same inputs -> same 14 sheets, HTML, Excel, evidence_auto.csv, snapshot and console text (two runs: second uses the snapshot)
  2  same again with pop-up hooks and the NSE fetch path switched on
  3  an in-memory run from uploaded-file objects writes nothing to disk and gives the same sheets
  4  the stages can be driven one at a time with pauses (what a Streamlit page does) and report progress
  5  loaders accept uploaded files; library errors are exceptions (not sys.exit)
"""
from __future__ import annotations
import argparse, contextlib, importlib.util, io, os, re, shutil, sys, tempfile, traceback, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"\n        {detail}" if (detail and not ok) else ""))
    return bool(ok)


def ensure_engines():
    try:
        import scoring, earnings_engine, ownership_engine, catalyst_engine, leadership_engine  # noqa: F401
        return f"your real engine files ({Path(scoring.__file__).resolve().parent})"
    except ImportError:
        sys.path.insert(0, str(HERE / "tests" / "stub_engines"))
        import scoring, earnings_engine, ownership_engine, catalyst_engine, leadership_engine  # noqa: F401
        return "TEST STAND-INS for the engine files (real ones not importable here)"


def load_legacy():
    p = HERE / "legacy" / "portfolio_diagnostic_v4_monolith.py"
    spec = importlib.util.spec_from_file_location("legacy_monolith", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


STAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{4}")


def norm_text(t, work):
    return STAMP.sub("STAMP", str(t).replace(str(work), "<W>").replace(str(work).replace("\\", "/"), "<W>"))


def make_args(work, fetch):
    return argparse.Namespace(holdings=str(work / "holdings.csv"), evidence=str(work / "evidence.csv"), peers=str(work / "peers.csv"),
                              outdir=str(work / "out"), review_date=None, rf=None, tax_estimate=False, no_cache=True, no_open=True,
                              no_fetch=not fetch, refresh_evidence=False, no_prompt=False, prompt=False, init=False, demo=False,
                              fetch_evidence=fetch)


@contextlib.contextmanager
def fake_network():
    import mock_world as mw
    saved = sys.modules.get("yfinance")
    sys.modules["yfinance"] = types.SimpleNamespace(Ticker=mw.FakeTicker)
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = saved


def execute(which, work, legacy, new, fetch, with_ia):
    """Run one variant ('old' = single file, 'new' = modules) in `work`; return what it produced."""
    import mock_world as mw
    import pdiag.evidence as pev
    ia = mw.FakeIA() if with_ia else None
    args = make_args(work, fetch)
    buf = io.StringIO()
    with fake_network(), contextlib.redirect_stdout(buf):
        if which == "old":
            legacy.CompositeProvider = mw.MockProvider
            legacy.input_assistant = ia
            legacy.dalal, legacy.HAS_DALAL = mw.FakeDalal, True
            sheets = legacy.run(args)
        else:
            new.input_assistant = ia
            pev.dalal, pev.HAS_DALAL = mw.FakeDalal, True
            sheets = new.run(args, provider_factory=mw.MockProvider)
    return dict(sheets=sheets, stdout=norm_text(buf.getvalue(), work), calls=(ia.calls if ia else []), work=work)


def read_files(work):
    import pandas as pd
    out = Path(work) / "out"
    res = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.suffix != ".pkl":
            res[norm_text(p.relative_to(out), work)] = p
    return res


def compare(tag, a, b):
    import pandas as pd
    ok = True
    if list(a["sheets"]) != list(b["sheets"]):
        return check(f"{tag}: same sheet names and order", False, f"{list(a['sheets'])} vs {list(b['sheets'])}")
    check(f"{tag}: same sheet names and order ({len(a['sheets'])} sheets)", True)
    bad = []
    for k in a["sheets"]:
        try:
            pd.testing.assert_frame_equal(a["sheets"][k], b["sheets"][k], check_exact=False, rtol=1e-9, atol=1e-9)
        except AssertionError as e:
            bad.append(f"{k}: {str(e).splitlines()[0][:150]}")
    ok &= check(f"{tag}: every sheet identical", not bad, "; ".join(bad))
    ok &= check(f"{tag}: console text identical", a["stdout"] == b["stdout"],
                "first difference: " + next((f"{x!r} != {y!r}" for x, y in zip(a["stdout"].splitlines(), b["stdout"].splitlines()) if x != y), "length differs"))
    ok &= check(f"{tag}: hook calls identical", a["calls"] == b["calls"], f"{a['calls']} vs {b['calls']}")
    fa, fb = read_files(a["work"]), read_files(b["work"])
    ok &= check(f"{tag}: same output files ({', '.join(fa) or 'none'})", list(fa) == list(fb), f"{list(fa)} vs {list(fb)}")
    for name in fa:
        if name not in fb:
            continue
        pa, pb = fa[name], fb[name]
        if pa.suffix == ".html":
            same = norm_text(pa.read_text(encoding="utf-8"), a["work"]) == norm_text(pb.read_text(encoding="utf-8"), b["work"])
        elif pa.suffix == ".xlsx":
            xa, xb = pd.read_excel(pa, sheet_name=None), pd.read_excel(pb, sheet_name=None)
            same = list(xa) == list(xb)
            for k in xa:
                if same:
                    try:
                        pd.testing.assert_frame_equal(xa[k], xb[k], check_exact=False, rtol=1e-9, atol=1e-9)
                    except AssertionError:
                        same = False
        elif name == "evidence_auto.csv":
            da, db = pd.read_csv(pa).drop(columns=["fetched_at"], errors="ignore"), pd.read_csv(pb).drop(columns=["fetched_at"], errors="ignore")
            same = da.equals(db)
        else:
            same = pa.read_text(encoding="utf-8") == pb.read_text(encoding="utf-8")
        ok &= check(f"{tag}: {name} identical", same)
    return ok


def main():
    import pandas as pd
    print("=" * 78 + "\nSTEP 27 CHECK - modular version vs single-file version\n" + "=" * 78)
    print("engines:", ensure_engines())
    legacy_path = HERE / "legacy" / "portfolio_diagnostic_v4_monolith.py"
    if not legacy_path.exists():
        print("legacy/portfolio_diagnostic_v4_monolith.py is missing - run apply_step27.py first.")
        return 1
    import mock_world as mw
    from pdiag.verify import legacy_mismatches, verbatim_mismatches
    map_path = HERE / "pdiag" / "moved_map.json"

    print("\n[0] the files line up")
    missing, changed = legacy_mismatches(legacy_path, map_path)
    check("single-file version is exactly the file the modules were cut from", not missing and not changed,
          f"missing {missing}; changed since the Project copy: {changed}. Send me the current file and I will re-cut the modules.")
    bad = verbatim_mismatches(HERE / "pdiag", legacy_path, map_path)
    check("every function marked 'moved unchanged' is byte-for-byte identical in its module", not bad, f"different: {bad}")

    import portfolio_diagnostic as new
    legacy = load_legacy()
    tmp = Path(tempfile.mkdtemp(prefix="step27_"))
    try:
        print("\n[1] same inputs, no pop-ups, no fetch (two runs: the second reads the first run's snapshot)")
        runs = {}
        for which in ("old", "new"):
            w = tmp / f"s1_{which}"
            mw.write_inputs(w)
            r1 = execute(which, w, legacy, new, fetch=False, with_ia=False)
            r2 = execute(which, w, legacy, new, fetch=False, with_ia=False)
            runs[which] = (r1, r2)
        compare("run 1", runs["old"][0], runs["new"][0])
        compare("run 2", runs["old"][1], runs["new"][1])
        sc = runs["new"][1]["sheets"]["Scorecard_A"]
        check("run 2 has real change-tracking values (Score_Delta filled)", sc["Score_Delta"].notna().any())
        sheets_new = runs["new"][0]["sheets"]
        n_hard = int(sheets_new["Scorecard_A"].Review_Risk.str.contains("HARD").sum())
        check(f"the mock portfolio exercises the interesting paths ({len(sheets_new['Scorecard_A'])} scored, {n_hard} hard overrides, "
              f"{len(sheets_new['Data_Gaps'])} data gaps, {len(sheets_new['Evidence_Register'])} register rows)", n_hard >= 2 and len(sheets_new["Data_Gaps"]) >= 2)

        print("\n[2] pop-up hooks and the NSE fetch path")
        res2 = {}
        for which in ("old", "new"):
            w = tmp / f"s2_{which}"
            mw.write_inputs(w)
            (w / "out" / "evidence_auto.csv").unlink()
            res2[which] = execute(which, w, legacy, new, fetch=True, with_ia=True)
        compare("fetch + hooks", res2["old"], res2["new"])
        check("hooks were called in the original order", [c[0] for c in res2["new"]["calls"]] ==
              ["check_holdings_layout", "apply_saved_buy_dates", "run_prompts", "review_costs"], str(res2["new"]["calls"]))

        print("\n[3] in-memory run from uploaded-file objects (what the app does)")
        from pdiag.hooks import NullHooks
        from pdiag.loaders import load_holdings, evidence_from_df, peers_from_df, read_table
        from pdiag.pipeline import new_context, run_staged, RunOptions
        from pdiag.report_excel import build_xlsx_bytes
        from pdiag.report_html import render_result_html
        from pdiag.state import MemoryState
        w = tmp / "s3"
        mw.write_inputs(w)
        # the snapshot the first run of scenario 1 left behind is what the user would upload next time
        snap_bytes = (tmp / "s1_new" / "out" / ".cache" / "last_run_snapshot.csv").read_bytes()
        (w / "out" / ".cache").mkdir(parents=True, exist_ok=True)

        def upload(name, data):
            f = io.BytesIO(data)
            f.name = name
            return f
        empty = tmp / "empty_cwd"
        empty.mkdir()
        old_cwd = os.getcwd()
        os.chdir(empty)
        try:
            lots = load_holdings(upload("holdings.csv", (w / "holdings.csv").read_bytes()))
            manual = evidence_from_df(read_table(upload("evidence.csv", (w / "evidence.csv").read_bytes())))
            peers = peers_from_df(read_table(upload("peers.csv", (w / "peers.csv").read_bytes())))
            state = MemoryState(auto_csv=(w / "out" / "evidence_auto.csv").read_bytes(), snapshot_csv=snap_bytes)
            ctx = new_context(lots, mw.MockProvider(), state, manual_evid=manual, peers=peers,
                              options=RunOptions(fetch_evidence=False), verified=None)
            res = run_staged(ctx, NullHooks())
            html = render_result_html(res, stamp="STAMP")
            xbytes = build_xlsx_bytes(res.sheets)
            left = [str(p) for p in empty.rglob("*")]
        finally:
            os.chdir(old_cwd)
        a = dict(sheets=runs["new"][1]["sheets"])
        bad = []
        for k in a["sheets"]:
            try:
                pd.testing.assert_frame_equal(a["sheets"][k], res.sheets[k], check_exact=False, rtol=1e-9, atol=1e-9)
            except AssertionError as e:
                bad.append(f"{k}: {str(e).splitlines()[0][:120]}")
        check("in-memory run gives the same 14 sheets as the disk run", not bad, "; ".join(bad))
        check("in-memory run created no files", not left, str(left))
        written_html = next(p for n, p in read_files(tmp / "s1_new").items() if n.endswith(".html"))
        check("render_result_html equals the HTML file the command line writes", norm_text(written_html.read_text(encoding="utf-8"), tmp / "s1_new") == norm_text(html, tmp / "s1_new"))
        xs = pd.read_excel(io.BytesIO(xbytes), sheet_name=None)
        check(f"Excel bytes open and hold the non-empty sheets ({len(xs)})", list(xs) == [k for k, v in res.sheets.items() if len(v) > 0])
        co = state.carry_over_csv()
        check("carry-over files for the next run are produced in memory (evidence cache + this run's scores)",
              set(co) == {"evidence_auto.csv", "last_run_snapshot.csv"} and co["last_run_snapshot.csv"].startswith("Symbol,Weighted"), str(list(co)))

        print("\n[4] staged run with pauses and progress (a Streamlit page reruns between these calls)")
        from pdiag.pipeline import stage_evidence, stage_prices, apply_cost_review, stage_analysis
        log, prog = [], []

        class PauseHooks(NullHooks):
            def after_load(self, lots):
                log.append("after_load")
                return lots

            def prompts(self, lots, evid, peers):
                log.append("prompts")
                lots = lots.copy()
                m = lots.buy_date.isna() & (lots.symbol == "BBB")
                lots.loc[m, "buy_date"] = mw.TODAY - pd.Timedelta(days=500)
                return lots, evid, peers

            def cost_review(self, HD, cfg):
                log.append("cost_review")
                return HD, {"EEE"}, {}
        hooks = PauseHooks()
        lots = load_holdings(w / "holdings.csv")
        state4 = MemoryState(auto_csv=(w / "out" / "evidence_auto.csv").read_bytes(), snapshot_csv=None)
        ctx = new_context(lots, mw.MockProvider(), state4, manual_evid=manual, peers=peers, options=RunOptions(fetch_evidence=False),
                          progress=lambda stage, i, n, label: prog.append((stage, i, n)))
        ctx.lots = hooks.after_load(ctx.lots)
        stage_evidence(ctx)
        check("after the evidence stage the merged evidence exists and the form has not run yet", hasattr(ctx, "evid") and log == ["after_load"])
        ctx.lots, ctx.evid, ctx.peers = hooks.prompts(ctx.lots, ctx.evid, ctx.peers)
        stage_prices(ctx)
        check("after the price stage the holdings table exists", len(ctx.HD) == len({r[0] for r in mw.HOLDINGS}) and "Value" in ctx.HD.columns)
        apply_cost_review(ctx, hooks)
        r4 = stage_analysis(ctx)
        check("hooks ran in order with the stages between them", log == ["after_load", "prompts", "cost_review"], str(log))
        check("the owner-confirmed line leaves the cost-basis flags", "EEE" not in set(r4.sheets["Holdings"].loc[r4.sheets["Holdings"].CostFlag != "", "Symbol"]))
        stages = {s for s, _, _ in prog}
        check("progress was reported for prices and fundamentals", {"prices", "fundamentals"} <= stages and prog[0][2] > 0, str(sorted(stages)))
        import pdiag.evidence as pev
        pev.dalal, pev.HAS_DALAL = mw.FakeDalal, True
        got = []
        with fake_network(), contextlib.redirect_stdout(io.StringIO()):
            pev.fetch_fresh_evidence(["AAA", "RED"], progress=lambda i, n, s: got.append((i, n, s)))
        check("the NSE fetch reports progress per symbol", got == [(1, 2, "AAA"), (2, 2, "RED")], str(got))

        print("\n[5] loaders and errors")
        csv_lots = load_holdings(w / "holdings.csv")
        xb = io.BytesIO()
        pd.read_csv(w / "holdings.csv").to_excel(xb, index=False)
        xb.seek(0)
        xb.name = "holdings.xlsx"
        xl_lots = load_holdings(xb)
        try:
            pd.testing.assert_frame_equal(csv_lots.reset_index(drop=True), xl_lots.reset_index(drop=True), check_dtype=False)
            same, why = True, ""
        except AssertionError as e:
            same, why = False, str(e).splitlines()[0][:200]
        check("an uploaded .xlsx holdings file loads like the .csv", same, why)
        from pdiag.errors import FatalInputError
        try:
            load_holdings(upload("x.csv", b"ticker_name,qty\nAAA,5\n"))
            raised = None
        except FatalInputError as e:
            raised = str(e)
        except SystemExit as e:
            raised = "SystemExit " + str(e)
        check("a bad holdings file raises FatalInputError (message unchanged)", raised == "Missing 'symbol' column.", str(raised))
        try:
            legacy.load_holdings(str(w / "nofile.csv"))
        except Exception:
            pass
        import ast
        srcs = []
        for p in (HERE / "pdiag").glob("*.py"):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.exit":
                    srcs.append(p.name)
        check("no pdiag module calls sys.exit (the library never ends the program)", not srcs, str(sorted(set(srcs))))
    except Exception:
        traceback.print_exc()
        check("test run finished without an unexpected error", False, "see the traceback above")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78 + f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed\n" + "=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    code = main()
    if code:
        sys.exit(code)
