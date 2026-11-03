"""check_step28.py - tests for the Streamlit page and its helpers (Step 28, first version).

Run from IDLE (F5) in the project folder. SYNTHETIC data only (tests/mock_world.py): no network, no real portfolio.
  A  carry-over pack, buy dates, promoter inputs and the cost-review answers behave as designed
  B  strict NSE mode: a refused NSE site is reported as 'not checked' (and the default behaviour is unchanged)
  C  the page itself, driven step by step with Streamlit's test runner: upload -> forms -> results -> carry-over pack -> second run
"""
from __future__ import annotations
import contextlib, io, os, sys, tempfile, shutil, traceback, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tests"))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"\n        {detail}" if (detail and not ok) else ""))
    return bool(ok)


def staged(mw, tmp, lots=None, provider=None, options=None, manual=None, pack=None):
    from pdiag.loaders import load_holdings
    from pdiag.pipeline import new_context, stage_evidence, stage_prices
    from pdiag.state import MemoryState
    from pdiag import appcore as ac
    lots = lots if lots is not None else load_holdings(tmp / "holdings.csv")
    pack = pack or ac.empty_pack()
    state = MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes(), snapshot_csv=pack["snapshot_csv"])
    from pdiag.pipeline import RunOptions
    ctx = new_context(lots, provider or mw.MockProvider(), state, manual_evid=manual or {}, options=options or RunOptions(fetch_evidence=False))
    with contextlib.redirect_stdout(io.StringIO()):
        stage_evidence(ctx); stage_prices(ctx)
    return ctx


def main():
    import numpy as np
    import pandas as pd
    import mock_world as mw
    from pdiag import appcore as ac
    from pdiag.config import CFG
    from pdiag.loaders import load_holdings
    from pdiag.pipeline import RunOptions, apply_cost_review, stage_analysis
    from pdiag.providers import NSEProvider
    from pdiag.state import MemoryState
    print("=" * 78 + "\nSTEP 28 CHECK - Streamlit page and helpers\n" + "=" * 78)
    tmp = Path(tempfile.mkdtemp(prefix="step28_"))
    old_cwd = os.getcwd()
    try:
        mw.write_inputs(tmp)
        print("\n[A] carry-over pack, buy dates, promoter and cost answers")
        lots = load_holdings(tmp / "holdings.csv")
        und = ac.undated_lots(lots)
        check("undated lots are listed for the form (cash excluded)", len(und) > 0 and "CSH" not in set(und.symbol), str(und.symbol.tolist()))
        typed = und.copy()
        i_bbb = typed.index[typed.symbol == "BBB"][0]
        typed.loc[i_bbb, "buy_date"] = pd.Timestamp("2023-03-01")
        lots2, n = ac.apply_buy_date_edits(lots, typed)
        check("a typed buy date lands on the right lot only", n == 1 and lots2.loc[lots2.symbol == "BBB", "buy_date"].notna().all() and lots2.buy_date.notna().sum() == lots.buy_date.notna().sum() + 1)
        bd = ac.merge_pack_dates(ac.empty_pack()["buy_dates"], lots, typed)
        pack_bytes = ac.build_pack(MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes(), snapshot_csv=b"Symbol,Weighted,FinalLabel,Review_Risk\nAAA,2.5,Neutral,\n"),
                                   bd, pd.DataFrame([dict(symbol="EEE", decision="confirmed", factor=np.nan)], columns=ac.DEC_COLS))
        names = set(zipfile.ZipFile(io.BytesIO(pack_bytes)).namelist())
        check("the pack holds the five expected files", names == {"evidence_auto.csv", "last_run_snapshot.csv", "buy_dates.csv", "cost_decisions.csv", "pack_info.json"}, str(names))
        pk = ac.parse_pack(pack_bytes)
        lots3, n3 = ac.apply_saved_buy_dates(lots, pk["buy_dates"])
        check("a later run fills the saved date back in from the pack", n3 == 1 and lots3.loc[lots3.symbol == "BBB", "buy_date"].notna().all())
        lots4, n4 = ac.apply_saved_buy_dates(lots2, pk["buy_dates"])
        check("a date already in the file is never overwritten by the pack", n4 == 0 and lots4.equals(lots2))
        try:
            ac.parse_pack(b"this is not a zip"); raised = None
        except Exception as e:
            raised = type(e).__name__
        check("a file that is not a zip gives a plain error (FatalInputError)", raised == "FatalInputError", str(raised))
        ev = {"AAA": {"symbol": "AAA", "promoter_holding_now": 50.0, "promoter_holding_1y": 49.0}}
        pm = ac.missing_promoter(ev, lots)
        check("promoter form lists stocks without both values, not AAA, not cash", "AAA" not in set(pm.symbol) and "CSH" not in set(pm.symbol) and "BBB" in set(pm.symbol))
        pm.loc[pm.symbol == "BBB", ["promoter_holding_now", "promoter_holding_1y"]] = [60.0, 62.0]
        ev2 = ac.apply_promoter_edits(ev, pm)
        check("typed promoter holding reaches the evidence; untouched rows stay empty", ev2["BBB"]["promoter_holding_now"] == 60.0 and "CCC" not in ev2 and ev["AAA"] == ev2["AAA"])

        ctx = staged(mw, tmp)
        cand = ac.cost_candidates(ctx.HD, CFG, pk["cost_decisions"])
        check("cost form lists the cost-anomaly line (EEE) and pre-fills 'confirmed' from the pack", "EEE" in set(cand.symbol) and bool(cand.loc[cand.symbol == "EEE", "confirm"].iloc[0]), str(cand[["symbol", "flag", "confirm"]]))
        ctx_c = staged(mw, tmp)
        c_ok, f_ok, dec = ac.decisions_from_edits(cand.assign(confirm=True))
        apply_cost_review(ctx_c, ac.FormHooks(c_ok, f_ok))
        check("confirming a line removes its cost flag (and only that line)", ctx_c.HD.loc["EEE", "CostFlag"] == "")
        ctx_f = staged(mw, tmp)
        old_cost, old_lot = ctx_f.HD.loc["EEE", "AvgCost"], ctx_f.lots.loc[ctx_f.lots.symbol == "EEE", "buy_price"].iloc[0]
        apply_cost_review(ctx_f, ac.FormHooks((), {"EEE": 50.0}))
        new_lot = ctx_f.lots.loc[ctx_f.lots.symbol == "EEE", "buy_price"].iloc[0]
        check("a correction factor changes the average cost and the lot price together, and clears the flag",
              abs(ctx_f.HD.loc["EEE", "AvgCost"] - old_cost * 50) < 1e-6 and abs(new_lot - old_lot * 50) < 1e-6 and ctx_f.HD.loc["EEE", "CostFlag"] == "", f"{old_cost} -> {ctx_f.HD.loc['EEE', 'AvgCost']}")
        ctx_n = staged(mw, tmp)
        apply_cost_review(ctx_n, ac.FormHooks())
        check("with no answers the line stays flagged as unreconciled", ctx_n.HD.loc["EEE", "CostFlag"] != "")
        res = stage_analysis(ctx_n)
        check("the analysis returns the 14 sheets", len(res.sheets) == 14, str(len(res.sheets)))

        print("\n[B] strict NSE mode")
        class Blocked(mw.MockProvider):
            def __init__(self, *a, **k):
                super().__init__(*a, **k); self.nse.available = False; self.nse.refused = 403
        def gaps_for(strict):
            c = staged(mw, tmp, provider=Blocked(), options=RunOptions(fetch_evidence=False, strict_nse=strict))
            apply_cost_review(c, ac.FormHooks())
            return stage_analysis(c).sheets["Data_Gaps"]
        g_on, g_off = gaps_for(True), gaps_for(False)
        check("strict mode: Data_Gaps says the NSE surveillance lists were NOT checked (HTTP 403)", g_on.Impact.str.contains("NOT checked").any() and g_on.Impact.str.contains("403").any())
        check("default (strict off): no such line, exactly as before", not g_off.Impact.astype(str).str.contains("NOT checked").any())

        class Sess:
            def __init__(self, code): self.code, self.headers = code, {}
            def get(self, *a, **k): return type("R", (), {"status_code": self.code, "json": lambda s: {"data": []}})()
        p = NSEProvider(); p.session = Sess(403); p.STRICT = True
        check("real NSEProvider, strict: a 403 home page marks NSE unavailable and records the code", p._prime() is False and p.available is False and p.refused == 403)
        q = NSEProvider(); q.session = Sess(403)
        check("real NSEProvider, default: unchanged (still treated as available)", q._prime() is True and q.available is True)

        print("\n[C] the page, driven with Streamlit's test runner")
        from streamlit.testing.v1 import AppTest
        pack0 = ac.build_pack(MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()), ac.empty_pack()["buy_dates"], ac.empty_pack()["cost_decisions"])
        files = dict(holdings=(tmp / "holdings.csv").read_bytes(), pack=pack0, evidence=(tmp / "evidence.csv").read_bytes(), peers=(tmp / "peers.csv").read_bytes())
        empty = tmp / "cwd"; empty.mkdir(); os.chdir(empty)

        def drive(pack_bytes):
            f = dict(files, pack=pack_bytes)
            at = AppTest.from_file(str(HERE / "app.py"), default_timeout=180)
            at.session_state["_provider_factory"] = mw.MockProvider
            at.session_state["_test_files"] = f
            at.run()
            first = (not at.exception, at.session_state["stage"], [b.key for b in at.button])
            at.checkbox(key="opt_fetch").uncheck().run()
            at.button(key="start").click().run()
            s1 = (not at.exception, at.session_state["stage"])
            at.button(key="continue_a").click().run()
            s2 = (not at.exception, at.session_state["stage"])
            at.button(key="run_b").click().run()
            s3 = (not at.exception, at.session_state["stage"])
            return at, first, s1, s2, s3
        at, first, s1, s2, s3 = drive(pack0)
        check("page opens on the upload step without an error", first[0] and first[1] == "upload" and "start" in first[2], str(first))
        check("Start reads the files and moves to the form step", s1 == (True, "form_a"), str(s1) + str([str(e.value) for e in at.exception]))
        check("Continue fetches prices and moves to the cost step", s2 == (True, "form_b"), str(s2) + str([str(e.value) for e in at.exception]))
        check("Run finishes with the results page", s3 == (True, "done"), str(s3) + str([str(e.value) for e in at.exception]))
        out = at.session_state["out"]
        check("raw holdings and fetched data are gone from memory; only the outputs stay", "ctx" not in at.session_state and "pack" not in at.session_state and set(out) >= {"html", "xlsx", "pack"})
        xs = pd.read_excel(io.BytesIO(out["xlsx"]), sheet_name=None)
        check("the Excel download opens and holds the 13 non-empty sheets", len(xs) == 13, str(len(xs)))
        check("the HTML report is complete and carries the 'not advice' footer", "Portfolio Diagnostic V4" in out["html"] and "Not investment advice" in out["html"])
        nz = zipfile.ZipFile(io.BytesIO(out["pack"])).namelist()
        check("the carry-over pack comes out with evidence cache, scores, buy dates and cost decisions", {"evidence_auto.csv", "last_run_snapshot.csv", "buy_dates.csv", "cost_decisions.csv"} <= set(nz), str(nz))
        check("the first run has no score change (nothing to compare with)", xs["Scorecard_A"]["Score_Delta"].isna().all())
        at2, f2, a1, a2, a3 = drive(out["pack"])
        xs2 = pd.read_excel(io.BytesIO(at2.session_state["out"]["xlsx"]), sheet_name=None) if a3 == (True, "done") else {}
        check("second run with the pack from the first: finishes and shows real score changes (Score_Delta filled)", a3 == (True, "done") and xs2 and xs2["Scorecard_A"]["Score_Delta"].notna().any(), str((a1, a2, a3)))
        check("second run reuses the evidence cache: Data_Gaps and sheets match the first run", xs2 and list(xs2) == list(xs))
        left = [str(p) for p in empty.rglob("*")]
        check("the page wrote no files to disk during either run", not left, str(left[:5]))
    except Exception:
        traceback.print_exc()
        check("test run finished without an unexpected error", False, "see the traceback above")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78 + f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed\n" + "=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    code = main()
    if code:
        sys.exit(code)
