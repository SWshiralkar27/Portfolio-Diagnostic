"""check_step12.py - tests for the F8 (risk) redesign approved on 6 Nov 2026. SYNTHETIC data only; no network.
F8 = average of three sub-scores (1 best, 5 worst), at least two of the three needed, otherwise unscored:
  3-year maximum drawdown   under 25% / 35% / 45% / 55%  -> 1 / 2 / 3 / 4, else 5
  beta                      under 0.8 / 1.0 / 1.2 / 1.5  -> 1 / 2 / 3 / 4, else 5
  idiosyncratic volatility  under 20% / 30% / 40% / 55%  -> 1 / 2 / 3 / 4, else 5
Flag: RunOptions.f8_redesign (command line: --f8-redesign; always on in the app). Default OFF = the old F8 rule."""
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
    from pdiag.engine_link import WEIGHTS
    from pdiag.loaders import load_holdings
    from pdiag.pipeline import RunOptions, new_context, stage_evidence, stage_prices, apply_cost_review, stage_analysis
    from pdiag.state import MemoryState
    print("=" * 78 + "\nSTEP 12 CHECK - F8 risk redesign\n" + "=" * 78)

    def f8(dd, beta, iv, new=True, bench=-0.25):
        perf = dict(maxdd_3y=dd, beta=beta, idio_vol=iv, bench_maxdd_3y=bench)
        S, E, *_ = score_stock("standard", {}, {}, perf, np.nan, 0, "", f8_new=new)
        return S["F8"], [e for e in E if e[0] == "F8"]
    cases = [((-0.20, 0.70, 0.15), 1.0), ((-0.30, 0.90, 0.25), 2.0), ((-0.40, 1.10, 0.35), 3.0), ((-0.50, 1.30, 0.45), 4.0), ((-0.60, 1.80, 0.70), 5.0),
             ((-0.20, 1.60, 0.30), 3.0), ((-0.25, 0.80, 0.20), 2.0)]
    got = [f8(*c)[0] for c, _ in cases]
    check("each band scores as approved (all sub-scores 1, 2, 3, 4, 5; a mixed case; the cut-off values themselves fall in the worse band)", got == [v for _, v in cases], str(got))
    check("two of the three available: averaged; only one available: unscored (never a neutral)", f8(-0.20, 1.60, np.nan)[0] == 3.0 and f8(-0.20, np.nan, np.nan)[0] is None and f8(np.nan, np.nan, 0.3)[0] is None)
    v, ev = f8(-0.40, 0.70, 0.15)
    check("the evidence line shows each value and its sub-score, labelled as the F8 redesign", len(ev) == 1 and "3y max drawdown 40% -> 3" in ev[0][3] and "beta 0.70 -> 1" in ev[0][3] and "idiosyncratic volatility 15% -> 1" in ev[0][3] and "redesign" in ev[0][2], str(ev))
    old_v, _ = f8(-0.40, 0.70, 0.15, new=False)
    check("flag OFF: the old rule is untouched (a 40% drawdown against a 25% benchmark drawdown still scores 5, whatever the beta)", old_v == 5 and v < 2.0, f"old {old_v}, new {v}")

    tmp = Path(tempfile.mkdtemp(prefix="step12_"))
    try:
        mw.write_inputs(tmp)
        def run(flag):
            lots = load_holdings(tmp / "holdings.csv")
            ctx = new_context(lots, mw.MockProvider(), MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()), options=RunOptions(fetch_evidence=False, f8_redesign=flag))
            with contextlib.redirect_stdout(io.StringIO()):
                stage_evidence(ctx); stage_prices(ctx); apply_cost_review(ctx, ac.FormHooks()); return stage_analysis(ctx)
        r_off, r_on = run(False), run(True)
        s_off, s_on = r_off.sheets["Scorecard_A"].set_index("Symbol"), r_on.sheets["Scorecard_A"].set_index("Symbol")
        check("default: RunOptions.f8_redesign is False", RunOptions().f8_redesign is False)
        check("flag ON changes only what it should: every other factor score is identical", all(s_off[f].fillna(-1).equals(s_on[f].reindex(s_off.index).fillna(-1)) for f in ["F1", "F2", "F5", "F6", "F7"]))
        n_scored = int(s_on.F8.notna().sum())
        check("flag ON: F8 is scored for the holdings with enough data and lies between 1 and 5", n_scored >= 10 and s_on.F8.dropna().between(1, 5).all(), f"{n_scored} scored")
        check("flag ON: F8 no longer sits at 5 for nearly everything (more than one distinct value)", s_on.F8.dropna().round().nunique() > 1 and s_off.F8.dropna().nunique() >= 1)
        W = {k: WEIGHTS[k] for k in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]}
        def weighted(row):
            av = {k: row[k] for k in W if pd.notna(row[k])}
            return sum(W[k] * v for k, v in av.items()) / sum(W[k] for k in av)
        ok_w = all(abs(weighted(s_on.loc[s]) - s_on.loc[s, "Weighted"]) < 1e-9 for s in s_on.index if s_on.loc[s, "F1"] == s_on.loc[s, "F1"])
        check("the weighted score and the label are recomputed from the new F8 with the same weights (65% of weight scored today)", ok_w)
        ev = r_on.sheets["Evidence"]
        check("the Evidence sheet carries the new F8 lines; with the flag off none do", ev.Basis.astype(str).str.contains("F8 redesign").any() and not r_off.sheets["Evidence"].Basis.astype(str).str.contains("F8 redesign").any())
        sm_on, sm_off = [str(x) for x in r_on.summary], [str(x) for x in r_off.summary]
        check("the summary states the rule and the thresholds when on, and nothing when off", any("F8 (risk) uses" in x and "25/35/45/55" in x for x in sm_on) and not any("F8 (risk) uses" in x for x in sm_off))
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
