"""check_step13.py - tests for the financial-sector inputs (Step 13). SYNTHETIC data only; no network.
Lenders: F6 (financial health) is scored from GNPA and capital adequacy that YOU supply in evidence.csv (gnpa_pct, car_pct), with the
existing bands (unchanged): GNPA under 1.5 / 3 / 5 / 8 % scores 1 / 2 / 3 / 4, else 5; CAR under 11.5 / 13 / 15 / 18 % scores 5 / 4 / 3 / 2, else 1;
F6 is the average of the parts you give. NEW and text-only: fin_asof and fin_source appear on the F6 evidence line; f6_not_applicable = yes
marks a business that is not a lender (asset manager, insurer): F6 is then 'not applicable', not counted as missing data."""
from __future__ import annotations
import contextlib, io, sys, tempfile, shutil, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tests"))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"\n        {detail}" if (detail and not ok) else ""))


def main():
    import numpy as np
    import pandas as pd
    import mock_world as mw
    from pdiag import appcore as ac
    from pdiag.factors import score_stock
    from pdiag.loaders import load_holdings, read_table
    from pdiag.pipeline import RunOptions, new_context, stage_evidence, stage_prices, apply_cost_review, stage_analysis
    from pdiag.state import MemoryState
    print("=" * 78 + "\nSTEP 13 CHECK - financial-sector inputs\n" + "=" * 78)

    def f6(ev, st="bank"):
        S, E, *_ = score_stock(st, ev, {}, dict(maxdd_3y=np.nan), np.nan, 0, "")
        return S["F6"], [e for e in E if e[0] == "F6"]
    cases = [(dict(gnpa_pct=1.0, car_pct=20.0), 1.0), (dict(gnpa_pct=2.0, car_pct=17.0), 2.0), (dict(gnpa_pct=4.0, car_pct=14.0), 3.0),
             (dict(gnpa_pct=6.0, car_pct=12.0), 4.0), (dict(gnpa_pct=9.0, car_pct=10.0), 5.0), (dict(gnpa_pct=1.0, car_pct=10.0), 3.0)]
    check("each band scores as before, and the two parts are averaged", [f6(e)[0] for e, _ in cases] == [v for _, v in cases], str([f6(e)[0] for e, _ in cases]))
    check("only GNPA, or only CAR: that part alone; neither: unscored (never a neutral)", f6(dict(gnpa_pct=2.0))[0] == 2.0 and f6(dict(car_pct=14.0))[0] == 3.0 and f6({})[0] is None)
    check("the same bands apply to every financial type (bank, nbfc, hfc); an ordinary company ignores GNPA and CAR", f6(dict(gnpa_pct=4.0, car_pct=14.0), "nbfc")[0] == 3.0 and f6(dict(gnpa_pct=4.0, car_pct=14.0), "standard")[0] is None)
    _, ev0 = f6(dict(gnpa_pct=2.1, car_pct=16.2))
    check("without a date or source the F6 evidence line is exactly what it was before", ev0[0][3] == "GNPA 2.1%, CAR 16.2%", ev0[0][3])
    _, ev1 = f6(dict(gnpa_pct=2.1, car_pct=16.2, fin_asof="2026-09-30", fin_source="Q2 FY27 results, investor presentation"))
    check("with fin_asof and fin_source the line shows the quarter and where the figures came from", "(as of 2026-09-30)" in ev1[0][3] and "source: Q2 FY27 results" in ev1[0][3], ev1[0][3])

    dup_csv = io.BytesIO(b"symbol,gnpa_pct,car_pct,promoter_pledge_pct\nSBIN,2.0,14,\nSBIN,,,5\nICICIBANK,1.5,16,\n"); dup_csv.name = "evidence.csv"
    try: ac.read_evidence(dup_csv); err = ""
    except Exception as e: err = f"{type(e).__name__}: {e}"
    check("an evidence file with the same stock on two rows is refused with a plain message (the last row would otherwise replace the first)", err.startswith("FatalInputError") and "SBIN" in err and "one row per stock" in err, err)
    tmp = Path(tempfile.mkdtemp(prefix="step13_"))
    try:
        mw.write_inputs(tmp)
        def run(manual):
            ctx = new_context(load_holdings(tmp / "holdings.csv"), mw.MockProvider(), MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()),
                              manual_evid=manual, options=RunOptions(fetch_evidence=False))
            with contextlib.redirect_stdout(io.StringIO()):
                stage_evidence(ctx); stage_prices(ctx); apply_cost_review(ctx, ac.FormHooks()); return stage_analysis(ctx)
        base = {"CCC": {"symbol": "CCC"}}                                           # CCC is the mock's bank; no GNPA / CAR given
        r0 = run(base); sc0 = r0.sheets["Scorecard_A"].set_index("Symbol")
        check("a bank with no GNPA or CAR: F6 unscored and listed as missing data", pd.isna(sc0.loc["CCC", "F6"]) and "F6 (Financial Health)" in sc0.loc["CCC", "Review_DataGap"], str(sc0.loc["CCC", "Review_DataGap"]))
        r1 = run({"CCC": {"symbol": "CCC", "f6_not_applicable": "yes"}}); sc1 = r1.sheets["Scorecard_A"].set_index("Symbol")
        check("marked not applicable: still unscored, no longer listed as missing, and the note says why",
              pd.isna(sc1.loc["CCC", "F6"]) and "F6 (Financial Health)" not in str(sc1.loc["CCC", "Review_DataGap"]) and "not applicable to this kind of business" in sc1.loc["CCC", "ConfidenceReason"], str(sc1.loc["CCC", ["Review_DataGap", "ConfidenceReason"]].tolist()))
        check("a not-applicable mark does not change the score, coverage or label of anything", sc1.Weighted.reindex(sc0.index).fillna(-1).equals(sc0.Weighted.fillna(-1)) and sc1.Coverage.reindex(sc0.index).equals(sc0.Coverage) and sc1.FinalLabel.reindex(sc0.index).equals(sc0.FinalLabel))
        r2 = run({"CCC": {"symbol": "CCC", "gnpa_pct": 2.0, "car_pct": 17.0, "fin_asof": "2026-09-30", "fin_source": "bank results"}}); sc2 = r2.sheets["Scorecard_A"].set_index("Symbol")
        ev2 = r2.sheets["Evidence"]; l2 = ev2[(ev2.Symbol == "CCC") & ev2.Factor.str.startswith("F6")]
        check("with GNPA and CAR given the bank gets an F6 score, the Evidence sheet carries the date and source, and its weighted score moves", sc2.loc["CCC", "F6"] == 2.0 and len(l2) == 1 and "as of 2026-09-30" in l2.Evidence.iloc[0] and sc2.loc["CCC", "Weighted"] != sc0.loc["CCC", "Weighted"], str(l2.Evidence.tolist()))
        r3 = run({"AAA": {"symbol": "AAA", "f6_not_applicable": "yes"}}); sc3 = r3.sheets["Scorecard_A"].set_index("Symbol")
        check("the mark does nothing for a holding whose F6 is scored (an ordinary company)", sc3.loc["AAA", "F6"] == sc0.loc["AAA", "F6"] and "not applicable" not in sc3.loc["AAA", "ConfidenceReason"])
        tpl = HERE / "financials_template.csv"
        if tpl.exists():
            ev = ac.read_evidence(tpl)
            check("the template loads as an evidence file: 14 rows, the four non-lenders pre-marked", len(ev) == 14 and sum(1 for v in ev.values() if str(v.get("f6_not_applicable", "")).lower() == "yes") == 4, str(len(ev)))
    except Exception:
        traceback.print_exc(); check("test run finished without an unexpected error", False, "see the traceback above")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78 + f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed\n" + "=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    code = main()
    if code:
        sys.exit(code)
