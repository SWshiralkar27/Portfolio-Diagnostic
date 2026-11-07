"""check_step11.py - tests for the confidence rule approved on 7 Nov 2026 (option B1). SYNTHETIC data only; no network.
  High    only if >= 90% of the FULL specification weight is scored, scorable coverage >= 85%, no proxy valuation, no hard override
  Medium  scorable coverage >= 70%
  Low     scorable coverage < 70%   (Low also turns the research signal into 'Insufficient evidence')
Flag: RunOptions.confidence_ceiling (command line: --confidence-ceiling; always on in the app). Default OFF = the old rule."""
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
    from pdiag.confidence import confidence_ceiling_rule, true_coverage
    from pdiag.engine_link import WEIGHTS, FACTORS
    from pdiag.loaders import load_holdings
    from pdiag.pipeline import RunOptions, new_context, stage_evidence, stage_prices, apply_cost_review, stage_analysis
    from pdiag.state import MemoryState
    print("=" * 78 + "\nSTEP 11 CHECK - confidence ceiling (B1)\n" + "=" * 78)
    allf = {k: 2.0 for k in FACTORS}
    six = {k: (2.0 if k in ("F1", "F2", "F5", "F6", "F7", "F8") else None) for k in FACTORS}
    lvl = lambda S, cov=1.0, proxies=(), hard=False: confidence_ceiling_rule(S, WEIGHTS, cov, list(proxies), hard, "coverage")[0]
    check("with the six scorable factors scored, true coverage is 65% of the full weight", abs(true_coverage(six, WEIGHTS) - 0.65) < 1e-9)
    check("six factors scored: never High, even at 100% scorable coverage; the note explains the ceiling", lvl(six) == "Medium" and "ceiling Medium: scored weight 65%" in confidence_ceiling_rule(six, WEIGHTS, 1.0, [], False, "coverage 100%")[1])
    check("scorable coverage 70% is Medium, 69% is Low (the specification's boundary)", lvl(six, 0.70) == "Medium" and lvl(six, 0.69) == "Low" and lvl(six, 0.46) == "Low")
    check("all ten factors scored: High returns by itself (no code change needed once more factors are built)", lvl(allf) == "High" and "ceiling" not in confidence_ceiling_rule(allf, WEIGHTS, 1.0, [], False, "coverage 100%")[1])
    check("...but not with a proxy valuation, a hard override, or scorable coverage under 85%", lvl(allf, proxies=["valuation"]) == "Medium" and lvl(allf, hard=True) == "Medium" and lvl(allf, 0.80) == "Medium")
    nine = dict(allf, F9=None)
    check("ten factors minus one low-weight factor (95% of the weight) can still be High", lvl(nine) == "High")

    tmp = Path(tempfile.mkdtemp(prefix="step11_"))
    try:
        mw.write_inputs(tmp)
        def run(flag):
            ctx = new_context(load_holdings(tmp / "holdings.csv"), mw.MockProvider(), MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()),
                              options=RunOptions(fetch_evidence=False, confidence_ceiling=flag))
            with contextlib.redirect_stdout(io.StringIO()):
                stage_evidence(ctx); stage_prices(ctx); apply_cost_review(ctx, ac.FormHooks()); return stage_analysis(ctx)
        r_off, r_on = run(False), run(True)
        s_off, s_on = r_off.sheets["Scorecard_A"].set_index("Symbol"), r_on.sheets["Scorecard_A"].set_index("Symbol")
        check("default: RunOptions.confidence_ceiling is False", RunOptions().confidence_ceiling is False)
        old_rule = [("High" if (c >= 0.85 and "proxy:" not in r) else "Medium" if c >= 0.60 else "Low") for c, r in zip(s_off.Coverage, s_off.ConfidenceReason)]
        check("flag OFF: the old rule is unchanged (High at 85% coverage without a proxy, Medium from 60%, else Low; no ceiling note)", old_rule == list(s_off.Confidence) and not s_off.ConfidenceReason.str.contains("ceiling").any(), str(s_off.Confidence.value_counts().to_dict()))
        check("flag ON: no holding is High (65% of the full weight is the most that can be scored today)", (s_on.Confidence == "High").sum() == 0 and set(s_on.Confidence) <= {"Medium", "Low"}, str(s_on.Confidence.value_counts().to_dict()))
        exp = s_on.Coverage.map(lambda c: "Medium" if c >= 0.70 else "Low")
        check("flag ON: Medium exactly when scorable coverage is at least 70%, Low below", (exp == s_on.Confidence).all(), str(pd.concat([s_on.Coverage, s_on.Confidence], axis=1)[exp != s_on.Confidence]))
        check("flag ON: every holding's note states the ceiling", s_on.ConfidenceReason.str.contains("confidence ceiling Medium").all())
        same = lambda c: s_off[c].reindex(s_on.index).fillna(-1).equals(s_on[c].fillna(-1))
        check("scores, weighted score and labels are identical with the flag on or off (only Confidence and its signal change)", all(same(c) for c in ["F1", "F2", "F5", "F6", "F7", "F8", "Weighted"]) and s_off.FinalLabel.reindex(s_on.index).equals(s_on.FinalLabel))
        low = s_on.index[s_on.Confidence == "Low"]
        check("a Low holding's research signal is 'Insufficient evidence' (or a review flag for a hard override)", len(low) >= 1 and s_on.loc[low, "ResearchSignal"].isin(["Insufficient evidence", "Review flag"]).all(), str(s_on.loc[low, "ResearchSignal"].to_dict()))
        sm_on, sm_off = [str(x) for x in r_on.summary], [str(x) for x in r_off.summary]
        check("the summary states the ceiling when on, and nothing when off", any("Confidence ceiling: Medium" in x and "% of the full specification weight is scored today" in x for x in sm_on) and not any("Confidence ceiling" in x for x in sm_off), str([x for x in sm_on if "ceiling" in x]))
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
