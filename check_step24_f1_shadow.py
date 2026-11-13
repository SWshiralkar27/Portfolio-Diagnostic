"""check_step24_f1_shadow.py - tests for the F1 revenue-growth SHADOW test (9 Nov 2026). SYNTHETIC data only; no network.
Idea: F1 (earnings and growth, 25% of the weight) has no sales-growth input for any holding, because quarterly sales growth is a manual-only
field in evidence.csv. Yahoo's annual statements already hold revenue (the tool computes rev_cagr and does not use it). The shadow test adds
it as a sales-growth leg (same bands as typed sales growth: over 15% scores 1, over 5% 2, at least -2% 3, at least -10% 4, else 5)
ONLY to measure the effect. No score, label or confidence changes. Flag: RunOptions.f1_revenue_shadow (command line --f1-shadow; on in the app)."""
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
    import earnings_engine as ee
    import mock_world as mw
    from pdiag import appcore as ac
    from pdiag.loaders import load_holdings
    from pdiag.pipeline import RunOptions, new_context, stage_evidence, stage_prices, apply_cost_review, stage_analysis
    from pdiag.state import MemoryState
    print("=" * 78 + "\nF1 REVENUE SHADOW CHECK\n" + "=" * 78)
    base = dict(ni_cagr=0.10, margin_delta=0.01, roce=18.0)
    old_s, old_r = ee.score_earnings({}, dict(base, rev_cagr=0.40))
    check("original behaviour is untouched: with the leg off, revenue growth is ignored (the same score and text as before)", ee.score_earnings({}, dict(base, rev_cagr=0.40), revenue_leg=False) == (old_s, old_r) and "Revenue" not in old_r)
    cases = [(0.20, 1), (0.10, 2), (0.00, 3), (-0.05, 4), (-0.20, 5)]
    sc = [ee.score_earnings({}, dict(base, rev_cagr=g), revenue_leg=True)[0] for g, _ in cases]
    exp = [np.mean([x, 2, 2, 2]) for _, x in cases]                    # profit 10% -> 2, margin +1% -> 2, ROCE 18 -> 2
    check("with the leg on, the revenue growth band is added as one more equal part of F1", np.allclose(sc, exp), f"{sc} vs {exp}")
    check("the evidence text names the leg and its source", "Revenue CAGR 20.0% (Yahoo annual statements)" in ee.score_earnings({}, dict(base, rev_cagr=0.20), revenue_leg=True)[1])
    typed = {"sales_growth_q1": 20, "sales_growth_q2": 18}
    check("typed quarterly sales growth always wins: no double counting, and the leg adds nothing", ee.score_earnings(typed, dict(base, rev_cagr=-0.2), revenue_leg=True) == ee.score_earnings(typed, dict(base, rev_cagr=-0.2)))
    check("no revenue figure (missing or not a number): the score is unchanged", ee.score_earnings({}, dict(base), revenue_leg=True) == ee.score_earnings({}, dict(base)) and ee.score_earnings({}, dict(base, rev_cagr=float("nan")), revenue_leg=True) == ee.score_earnings({}, dict(base)))

    tmp = Path(tempfile.mkdtemp(prefix="f1shadow_"))
    try:
        mw.write_inputs(tmp)
        def run(flag):
            ctx = new_context(load_holdings(tmp / "holdings.csv"), mw.MockProvider(), MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()),
                              options=RunOptions(fetch_evidence=False, f1_revenue_shadow=flag))
            with contextlib.redirect_stdout(io.StringIO()):
                stage_evidence(ctx); stage_prices(ctx); apply_cost_review(ctx, ac.FormHooks()); return stage_analysis(ctx)
        off, on = run(False), run(True)
        s_off, s_on = off.sheets["Scorecard_A"].set_index("Symbol"), on.sheets["Scorecard_A"].set_index("Symbol")
        check("default: RunOptions.f1_revenue_shadow is False", RunOptions().f1_revenue_shadow is False)
        same = lambda c: s_off[c].reindex(s_on.index).fillna(-1).equals(s_on[c].fillna(-1))
        check("nothing real changes with the flag on: every factor score, weighted score, label and confidence is identical", all(same(c) for c in ["F1", "F2", "F5", "F6", "F7", "F8", "Weighted", "Confidence"]) and s_off.FinalLabel.reindex(s_on.index).equals(s_on.FinalLabel))
        ev = on.sheets["Evidence"]; sh = ev[ev.Basis == "shadow"]
        check("the Evidence sheet carries a clearly labelled shadow line for each holding whose F1 would change; none with the flag off", len(sh) >= 3 and sh.Factor.str.contains("NOT used in any score").all() and not (off.sheets["Evidence"].Basis == "shadow").any(), str(len(sh)))
        real = ev[ev.Factor.str.startswith("F1 Earnings") & (ev.Basis != "shadow")].set_index("Symbol").Score
        alt = sh.set_index("Symbol").Score
        check("each shadow score differs from the real F1 score of the same holding, and the real score is the one in the scorecard", all(abs(alt[s] - real[s]) > 1e-9 and abs(real[s] - s_on.loc[s, "F1"]) < 1e-9 for s in alt.index))
        line = [str(x) for x in on.summary if str(x).startswith("F1 shadow test")]
        n_chg = len(sh)
        check("the summary states how many F1 scores and labels would change and that nothing changes without approval; nothing when off",
              len(line) == 1 and f"change F1 for {n_chg} of" in line[0] and "Nothing changes unless you approve" in line[0] and not any("F1 shadow test" in str(x) for x in off.summary), str(line))
        from pdiag.factors import schemes, weighted, classify
        SC = schemes()["base"]; lab_n = 0
        for s in alt.index:
            Sx = {f: (s_on.loc[s, f] if pd.notna(s_on.loc[s, f]) else None) for f in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]}
            Sx["F1"] = alt[s]; w, c = weighted(Sx, SC)
            if s_on.loc[s, "FinalLabel"] != "Candidate for review" and classify(w, c) != s_on.loc[s, "FinalLabel"]: lab_n += 1
        check("the number of labels the summary says would change matches an independent recomputation from the sheets", f"the label of {lab_n} holdings" in line[0], f"{lab_n} vs {line[0][-140:]}")
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
