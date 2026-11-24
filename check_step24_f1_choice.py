"""check_step24_f1_choice.py - chunk 1 of the F1 revenue-growth choice (engine side only; no app control yet). SYNTHETIC data only.
RunOptions.f1_revenue_leg: "off" (default: nothing changes), "nonfin" (non-financial holdings only) or "all"."""
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
    import pandas as pd
    import mock_world as mw
    from pdiag import appcore as ac
    from pdiag.loaders import load_holdings
    from pdiag.pipeline import RunOptions, new_context, stage_evidence, stage_prices, apply_cost_review, stage_analysis
    from pdiag.state import MemoryState
    print("=" * 78 + "\nF1 REVENUE CHOICE CHECK (chunk 1)\n" + "=" * 78)
    tmp = Path(tempfile.mkdtemp(prefix="f1choice_"))
    try:
        mw.write_inputs(tmp)
        def run(mode, shadow=False):
            ctx = new_context(load_holdings(tmp / "holdings.csv"), mw.MockProvider(), MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()),
                              options=RunOptions(fetch_evidence=False, f1_revenue_leg=mode, f1_revenue_shadow=shadow))
            with contextlib.redirect_stdout(io.StringIO()):
                stage_evidence(ctx); stage_prices(ctx); apply_cost_review(ctx, ac.FormHooks()); return stage_analysis(ctx)
        r_off, r_non, r_all = run("off"), run("nonfin"), run("all")
        sc = {k: r.sheets["Scorecard_A"].set_index("Symbol") for k, r in (("off", r_off), ("non", r_non), ("all", r_all))}
        check("default: the choice is off, and with it off nothing about F1 changes", RunOptions().f1_revenue_leg == "off")
        from pdiag.config import FIN_TYPES
        fin = [s for s in sc["off"].index if sc["off"].loc[s, "SectorType"] in FIN_TYPES]
        non = [s for s in sc["off"].index if s not in fin and pd.notna(sc["off"].loc[s, "F1"])]
        chg = lambda a, b, syms: [s for s in syms if abs(sc[a].loc[s, "F1"] - sc[b].loc[s, "F1"]) > 1e-9]
        check("the mock has at least one financial holding and several ordinary ones", len(fin) >= 1 and len(non) >= 5, f"{fin} / {len(non)}")
        c_non = chg("off", "non", non)
        check("non-financial only: ordinary holdings with a revenue figure get a different F1; every financial holding keeps its old F1", len(c_non) >= 3 and not chg("off", "non", fin), f"{c_non} / fin changed {chg('off', 'non', fin)}")
        check("all holdings: the financial holdings change too", len(chg("off", "all", fin)) >= 1 and set(chg("off", "all", non)) == set(c_non), str(chg("off", "all", fin)))
        ev = lambda r: r.sheets["Evidence"]
        has_leg = lambda r, s: ev(r)[(ev(r).Symbol == s) & ev(r).Factor.str.startswith("F1 Earnings") & ev(r).Evidence.str.contains("Revenue CAGR")].shape[0] == 1
        check("the Evidence line of a changed holding names the revenue leg; none appears with the choice off", all(has_leg(r_non, s) for s in c_non) and not any(has_leg(r_off, s) for s in non))
        same = lambda a, b, cols: all(sc[a][c].reindex(sc[b].index).fillna(-1).equals(sc[b][c].fillna(-1)) for c in cols)
        check("only F1 and what depends on it change: the other factor scores are identical in all three runs", same("off", "non", ["F2", "F5", "F6", "F7", "F8"]) and same("off", "all", ["F2", "F5", "F6", "F7", "F8"]))
        sm = lambda r: [str(x) for x in r.summary if str(x).startswith("F1 uses Yahoo annual revenue growth")]
        check("the summary names the choice used: non-financial only, or all holdings; nothing when off",
              len(sm(r_non)) == 1 and "non-financial holdings only" in sm(r_non)[0] and len(sm(r_all)) == 1 and "all holdings" in sm(r_all)[0] and not sm(r_off))
        s_on = run("nonfin", shadow=True); sh = s_on.sheets["Evidence"]; sh = sh[sh.Basis == "shadow"]
        check("with the shadow test also on, the shadow lines now only cover holdings the choice does not already use (the financials)", len(sh) >= 1 and set(sh.Symbol) <= set(fin), str(set(sh.Symbol)))
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
