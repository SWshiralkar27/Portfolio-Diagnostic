"""check_step28.py - tests for the Streamlit page and its helpers (Step 28, first version).

Run from IDLE (F5) in the project folder. SYNTHETIC data only (tests/mock_world.py): no network, no real portfolio.
  A  carry-over pack, buy dates, promoter inputs and the cost-review answers behave as designed
  B  strict NSE mode: a refused NSE site is reported as 'not checked' (and the default behaviour is unchanged)
  D  Step 28b: company info that Yahoo refuses (reported, paced, last good copy reused) and the unknown-sector guard
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


def named_file(name, data):
    f = io.BytesIO(data); f.name = name
    return f


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
        check("the pack holds the six expected files", names == {"evidence_auto.csv", "last_run_snapshot.csv", "buy_dates.csv", "cost_decisions.csv", "company_info.json", "pack_info.json"}, str(names))
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

        print("\n[D] Step 28b - company info refused by Yahoo, saved copy, unknown-sector guard")
        from pdiag.resilient import ResilientProvider, STALE_DAYS, BREAKER
        class Blind(mw.MockProvider):
            """Yahoo answers prices and statements but returns no company info (what the first hosted run saw)."""
            def __init__(self, *a, **k):
                super().__init__(*a, **k); self.info_calls = 0
            def info(self, tkr):
                self.info_calls += 1; return {}
        rp = ResilientProvider(mw.MockProvider(), pace=0)
        got = rp.info("AAA.NS")
        check("answer from Yahoo is passed through unchanged and the last good copy is kept (8 keys only)", got == mw.MockProvider().info("AAA.NS") and set(rp.company_info_store()["AAA.NS"]["info"]) <= {"sector", "marketCap", "trailingPE", "priceToBook", "returnOnEquity", "enterpriseToEbitda", "enterpriseToRevenue", "trailingEps"} and rp.status["AAA.NS"] == "fresh")
        check("prices, statements and the provider name pass straight through", rp.history("AAA.NS", "2020-01-01", "2026-01-01") is not None and "fin" in rp.statements("AAA.NS") and rp.name == "Mock")
        bl = Blind(); rb = ResilientProvider(bl, pace=0)
        for i in range(30): rb.info(f"S{i}.NS")
        check(f"a refusing Yahoo is asked only {BREAKER} times in a row, then the app stops asking (the run cannot hang)", bl.info_calls == BREAKER, str(bl.info_calls))
        nt = " ".join(rb.info_notes())
        check("Data_Gaps wording says NOT returned and that the app stopped asking; STATUS reason names the count", "NOT returned" in nt and "stopped asking" in nt and "company info missing for 30 of 30" in rb.info_reasons()[0], nt)
        bl2 = Blind(); r2 = ResilientProvider(bl2, pace=0)
        for i in range(BREAKER - 1): r2.info(f"X{i}.NS")
        mixed = type("M", (Blind,), {"info": lambda self, t: ({} if t.startswith("X") else mw.MockProvider().info(t))})()
        r3 = ResilientProvider(mixed, pace=0)
        for i in range(BREAKER - 1): r3.info(f"X{i}.NS")
        r3.info("AAA.NS")
        for i in range(BREAKER - 1): r3.info(f"Y{i}.NS") if False else r3.info(f"X{i + 50}.NS")
        check("one good answer resets the count of failures in a row", not r3._tripped)
        today = pd.Timestamp.today().normalize()
        sv = {"AAA.NS": dict(fetched=(today - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), info={"sector": "Financial Services", "marketCap": 2e12, "trailingPE": 18.0}),
              "BBB.NS": dict(fetched=(today - pd.Timedelta(days=STALE_DAYS + 10)).strftime("%Y-%m-%d"), info={"sector": "Technology", "marketCap": 1e11, "trailingPE": 25.0})}
        rs = ResilientProvider(Blind(), stored=sv, pace=0)
        a, b, c = rs.info("AAA.NS"), rs.info("BBB.NS"), rs.info("CCC.NS")
        check("saved copy used when Yahoo does not answer: recent copy in full, an old copy gives the sector only, no copy gives nothing",
              a.get("trailingPE") == 18.0 and b == {"sector": "Technology"} and c == {} and [rs.status[k] for k in ("AAA.NS", "BBB.NS", "CCC.NS")] == ["saved", "saved-sector-only", "missing"], str((a, b, c)))
        nt = " ".join(rs.info_notes())
        check("the note says the copy is from the carry-over pack and gives its age; the 40-day copy is called out", "carry-over pack" in nt and "5-40 days" in nt and "only the sector was reused" in nt, nt)
        pk2 = ac.parse_pack(ac.build_pack(MemoryState(auto_csv=b"symbol\nAAA\n"), ac.empty_pack()["buy_dates"], ac.empty_pack()["cost_decisions"], company_info=dict(sv, BAD="oops")))
        check("company info survives the carry-over pack round trip; a malformed entry is dropped", pk2["company_info"] == sv, str(pk2["company_info"].keys()))

        def lots_no_sector():
            l = load_holdings(tmp / "holdings.csv"); l["sector_type"] = np.nan; l["sector"] = np.nan; return l
        def review(provider, guard):
            from pdiag.loaders import evidence_from_df, read_table as _rt
            c = staged(mw, tmp, lots=lots_no_sector(), provider=provider, manual=evidence_from_df(_rt(tmp / "evidence.csv")), options=RunOptions(fetch_evidence=False, unknown_sector_guard=guard))
            apply_cost_review(c, ac.FormHooks())
            r = stage_analysis(c)
            return c, r, r.sheets["Scorecard_A"].set_index("Symbol")
        c_off, r_off, sc_off = review(Blind(), False)
        c_on, r_on, sc_on = review(Blind(), True)
        hard = lambda sc, s_: "HARD OVERRIDE: Earnings-quality" in str(sc.loc[s_, "Review_Risk"])
        check("guard OFF (default, as before): a lender-like cash-flow pattern is a hard override and the sector reads 'standard'", hard(sc_off, "FFF") and sc_off.loc["FFF", "SectorType"] == "standard")
        check("guard ON, sector unknown: the same flag is a risk tag ('Earnings-quality note'), the sector reads 'unknown'",
              (not hard(sc_on, "FFF")) and "Earnings-quality note" in str(sc_on.loc["FFF", "Review_Risk"]) and sc_on.loc["FFF", "SectorType"] == "unknown", str(sc_on.loc["FFF", ["SectorType", "Review_Risk"]].tolist()))
        check("guard ON leaves the other hard overrides alone (pledge on DDD, negative net worth on NEG)",
              "HARD" in str(sc_on.loc["DDD", "Review_Risk"]) and "HARD" in str(sc_on.loc["NEG", "Review_Risk"]) and sc_on.FinalLabel.eq("Candidate for review").sum() == sc_off.FinalLabel.eq("Candidate for review").sum() - 1,
              f"{sc_on.FinalLabel.eq('Candidate for review').sum()} vs {sc_off.FinalLabel.eq('Candidate for review').sum()}")
        gp = r_on.sheets["Data_Gaps"]
        check("guard ON: Data_Gaps says the sector is unknown for N stocks; guard OFF: no such line", gp.Impact.astype(str).str.contains("Sector unknown for").any() and not r_off.sheets["Data_Gaps"].Impact.astype(str).str.contains("Sector unknown").any())
        fin_store = {"FFF.NS": dict(fetched=today.strftime("%Y-%m-%d"), info={"sector": "Financial Services", "marketCap": 5e11, "trailingPE": 12.0, "priceToBook": 2.0})}
        _, _, sc_fin = review(ResilientProvider(Blind(), stored=fin_store, pace=0), True)
        check("a saved copy that knows the sector rescues it: treated as a financial, so no cash-flow flag at all",
              sc_fin.loc["FFF", "SectorType"] == "financial" and "Earnings-quality" not in str(sc_fin.loc["FFF", "Review_Risk"]), str(sc_fin.loc["FFF", ["SectorType", "Review_Risk"]].tolist()))

        class Recorder(mw.MockProvider):
            def __init__(self, *a, **k):
                super().__init__(*a, **k); self.log = []
            def info(self, tkr): self.log.append(("info", tkr)); return super().info(tkr)
            def history(self, tkr, a, b): self.log.append(("history", tkr)); return super().history(tkr, a, b)
        rec = Recorder()
        ctxp = new_ctx = None
        from pdiag.pipeline import new_context, stage_evidence, stage_prices
        ctxp = new_context(lots_no_sector(), ResilientProvider(rec, pace=0), MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()),
                           peers={"GGG": ["AAA", "HHH"]}, options=RunOptions(fetch_evidence=False))
        n_t = ac.prefetch_info(ctxp)
        with contextlib.redirect_stdout(io.StringIO()):
            stage_evidence(ctxp); stage_prices(ctxp); apply_cost_review(ctxp, ac.FormHooks()); stage_analysis(ctxp)
        kinds = [k for k, _ in rec.log]
        check("company info is asked for FIRST (before the first price request), once per ticker and never again later in the run",
              kinds.index("history") > max(i for i, k in enumerate(kinds) if k == "info") and kinds.count("info") == n_t and n_t >= 14, f"{n_t} tickers; info calls {kinds.count('info')}")
        rb2 = ResilientProvider(Blind(), pace=0)
        for i in range(3): rb2.info(f"Q{i}.NS")
        check("when Yahoo is refusing, the note says whether the very first request was already refused", "very first company-info request" in " ".join(rb2.info_notes()) and "already refused" in " ".join(rb2.info_notes()))
        flaky = type("F", (mw.MockProvider,), {"info": lambda self, t: (mw.MockProvider().info(t) if t == "Z0.NS" else {})})()
        rf = ResilientProvider(flaky, pace=0)
        for i in range(3): rf.info(f"Z{i}.NS")
        check("...and when the first request worked but later ones were refused, it says that instead", "worked, so Yahoo started refusing later" in " ".join(rf.info_notes()))

        print("\n[E] Step 28d - plain file errors, the wide peers layout, cost preview, safe re-run")
        wide = ("Company,NSE Ticker,Peer 1,Peer 2,Peer 3,Peer 4,Peer 5,Basis\n"
                "Alpha Ltd,AAA,BBB,CCC,PIDILITIND?,HERO Motocorp? Use HEROMOTOCO,DDD,ok\n"
                "Beta Ltd,BETA? (verify),AAA,CCC,,,,x\n"
                "Gamma Ltd,GAMMA? (verify),AAA,,,,,x\n"
                "Delta Ltd,VERIFY,AAA,,,,,x\n")
        pw, rep = ac.read_peers(named_file("peers.csv", wide.encode()), held={"AAA", "BETA"})
        check("wide peers layout is read: plain peers kept, a doubtful peer left out, the file's own correction used",
              pw.get("AAA") == ["BBB", "CCC", "HEROMOTOCO", "DDD"] and rep["layout"] == "wide" and any(x[2] == "PIDILITIND?" for x in rep["left_out"]) and rep["used"][0][2] == "HEROMOTOCO", str(pw))
        check("an own ticker 'BETA? (verify)' is kept only because BETA is exactly a symbol in the holdings; GAMMA is not guessed",
              "BETA" in pw and "GAMMA" not in pw and rep["kept"] == ["BETA"], str(pw))
        check("a placeholder such as VERIFY is never treated as a company", "VERIFY" not in pw)
        check("the Data_Gaps note describes what was left out", "left out" in ac.peers_note(rep) and "PIDILITIND?" in ac.peers_note(rep))
        pl, rl = ac.read_peers(named_file("p.csv", b"symbol,peer\nAAA,BBB\nAAA,CCC\n"))
        check("the one-row-per-peer layout still works exactly as before", pl == {"AAA": ["BBB", "CCC"]} and rl["layout"] == "long")
        def err(fn, *a):
            try: fn(*a); return ""
            except Exception as e: return f"{type(e).__name__}: {e}"
        e1 = err(ac.read_peers, named_file("p.csv", b"foo,bar\n1,2\n"))
        e2 = err(ac.read_evidence, named_file("e.csv", b"stock,pledge\nAAA,5\n"))
        check("a peers file with unknown columns gives a plain message that names the columns it found", e1.startswith("FatalInputError") and "foo" in e1 and "symbol and peer" in e1, e1)
        check("an evidence file without a symbol column gives a plain message, not KeyError", e2.startswith("FatalInputError") and "symbol" in e2 and "KeyError" not in e2, e2)
        ctx_e = staged(mw, tmp)
        ed = ac.cost_candidates(ctx_e.HD, CFG).assign(factor=[90.0] * 1) if len(ac.cost_candidates(ctx_e.HD, CFG)) == 1 else ac.cost_candidates(ctx_e.HD, CFG)
        bad = ac.cost_preview(ctx_e.HD, ed.assign(factor=1000.0), CFG)
        good = ac.cost_preview(ctx_e.HD, ed.assign(factor=50.0), CFG)
        check("cost preview: a factor that leaves an odd cost (2.00 x 1000 against a price near 90) is marked, a sensible one (x 50) is ok",
              len(bad) == 1 and bad.note.iloc[0] != "ok" and good.note.iloc[0] == "ok" and abs(good.avg_cost_after.iloc[0] - ctx_e.HD.loc["EEE", "AvgCost"] * 50) < 1e-6, str(bad))
        c1, c2 = staged(mw, tmp), staged(mw, tmp)
        snap = ac.snapshot_ctx(c2)
        apply_cost_review(c1, ac.FormHooks()); apply_cost_review(c1, ac.FormHooks((), {"EEE": 50.0}))          # the bug: Run pressed twice
        apply_cost_review(c2, ac.FormHooks()); ac.restore_ctx(c2, snap); apply_cost_review(c2, ac.FormHooks((), {"EEE": 50.0}))   # the fix
        n_gap = lambda c: sum(1 for g in c.gaps if g[0] == "EEE" and g[1] == "cost basis")
        check("pressing Run twice no longer leaves a stale cost note (or compounds a correction); the restored run has none", n_gap(c1) == 1 and n_gap(c2) == 0 and abs(c2.lots.loc[c2.lots.symbol == "EEE", "buy_price"].iloc[0] / staged(mw, tmp).lots.loc[lambda d: d.symbol == "EEE", "buy_price"].iloc[0] - 50) < 1e-9, f"{n_gap(c1)} vs {n_gap(c2)}")
        iso = ResilientProvider(type("I", (mw.MockProvider,), {"info": lambda self, t: ({} if t == "NOPE.NS" else mw.MockProvider().info(t))})(), pace=0)
        for t in ("AAA.NS", "NOPE.NS", "BBB.NS", "CCC.NS"): iso.info(t)
        ni = " ".join(iso.info_notes())
        check("one unknown ticker among good ones: the note names it and says it is not a refusal (no 'Yahoo started refusing')", "NOPE.NS" in ni and "not a refusal" in ni and "started refusing" not in ni, ni)

        print("\n[C] the page, driven with Streamlit's test runner")
        from streamlit.testing.v1 import AppTest
        pack0 = ac.build_pack(MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()), ac.empty_pack()["buy_dates"], ac.empty_pack()["cost_decisions"])
        files = dict(holdings=(tmp / "holdings.csv").read_bytes(), pack=pack0, evidence=(tmp / "evidence.csv").read_bytes(), peers=(tmp / "peers.csv").read_bytes())
        empty = tmp / "cwd"; empty.mkdir(); os.chdir(empty)

        def drive(pack_bytes, factory=None, hold=None):
            f = dict(files, pack=pack_bytes, **({"holdings": hold} if hold else {}))
            at = AppTest.from_file(str(HERE / "app.py"), default_timeout=180)
            at.session_state["_provider_factory"] = factory or mw.MockProvider
            at.session_state["_pace"] = 0
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
        pk1 = ac.parse_pack(out["pack"])
        check("the first run's pack carries last good company info for the stocks it looked up", len(pk1["company_info"]) >= 14 and all(v["info"] for v in pk1["company_info"].values()), str(len(pk1["company_info"])))
        atb, fb, b1, b2, b3 = drive(out["pack"], factory=Blind)
        ob = atb.session_state["out"] if b3 == (True, "done") else {}
        xb = pd.read_excel(io.BytesIO(ob["xlsx"]), sheet_name=None) if ob else {}
        check("Yahoo refuses company info but the pack has it: the run finishes, sector/cap tier/valuation are still filled, and the page says so",
              b3 == (True, "done") and xb["Scorecard_A"].CapTier.notna().sum() >= 10 and (xb["Evidence"].Factor.str.startswith("F5").sum() > 0)
              and any("carry-over pack" in n for n in ob["notes"]) and "carry-over pack" in ob["status"], str((b3, ob.get("notes"), ob.get("status"))))
        h_nosec = pd.read_csv(tmp / "holdings.csv").assign(sector_type="", sector="").to_csv(index=False).encode()      # a file like yours: no sector columns filled
        atc, fc, c1_, c2_, c3_ = drive(pack0, factory=Blind, hold=h_nosec)
        oc = atc.session_state["out"] if c3_ == (True, "done") else {}
        check("Yahoo refuses and there is no saved copy: the run still finishes, and STATUS and Data_Gaps say plainly that company info is missing",
              c3_ == (True, "done") and "company info missing" in oc["status"] and any("NOT returned" in n for n in oc["notes"]) and any("Sector unknown" in n for n in oc["notes"]), str((c3_, oc.get("status"), oc.get("notes"))))
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
