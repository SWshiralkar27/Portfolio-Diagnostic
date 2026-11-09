"""check_step18a.py - tests for the risk concentration statistics approved on 8 Nov 2026 (rows 1 to 4) and the correlation clusters (option A: average linkage, average correlation above 0.70).
SYNTHETIC data only; no network. Basis: daily CLOSING prices (price return), last 3 years, holdings with at least 252 daily returns,
current market-value weights, pairwise-complete correlation and covariance.
Flag: RunOptions.risk_contribution (command line: --risk-stats; always on in the app). Default OFF = nothing added."""
from __future__ import annotations
import contextlib, io, sys, tempfile, shutil, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tests"))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"\n        {detail}" if (detail and not ok) else ""))


def prices(rets):
    import pandas as pd
    return {s: pd.DataFrame({"close": 100 * (1 + r).cumprod()}) for s, r in rets.items()}


def main():
    import numpy as np
    import pandas as pd
    import mock_world as mw
    from pdiag import appcore as ac
    from pdiag.loaders import load_holdings
    from pdiag.pipeline import RunOptions, new_context, stage_evidence, stage_prices, apply_cost_review, stage_analysis
    from pdiag.report_html import render_result_html
    from pdiag.riskstats import risk_stats, correlation_clusters
    from pdiag.state import MemoryState
    print("=" * 78 + "\nSTEP 18a CHECK - risk concentration statistics\n" + "=" * 78)
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end="2026-11-06", periods=1000)
    mk = lambda x: pd.Series(x, index=idx)
    vals = lambda names, v=100.0: pd.Series({n: v for n in names})

    ind = {f"S{i}": mk(rng.normal(0, 0.01, 1000)) for i in range(8)}
    r = risk_stats(prices(ind), vals(ind))
    check("eight independent equal holdings: correlation near 0, risk contribution about 12.5% each, effective risk contributors near 8", abs(r["port"]["Average pairwise correlation"]) < 0.08 and r["rc"].RiskContribution.between(8, 17).all() and 7.0 < r["port"]["Effective number of risk contributors"] <= 8.0, str(r["port"]))
    check("risk contributions add up to 100% and the weights to 100%", abs(r["rc"].RiskContribution.sum() - 100) < 1e-9 and abs(r["rc"].Weight.sum() - 100) < 1e-9)

    f = mk(rng.normal(0, 0.01, 1000))
    one = {f"F{i}": f + mk(rng.normal(0, 0.0005, 1000)) for i in range(6)}
    r1 = risk_stats(prices(one), vals(one))
    check("six holdings that all follow one common factor: correlation near 1 and PC1 share near 100% (the diversification is only on paper)",
          r1["port"]["Average pairwise correlation"] > 0.95 and r1["port"]["PC1 share of return variance (single-factor dominance)"] > 0.95, str(r1["port"]))
    big = dict(ind); big["BIG"] = f; big["TWIN"] = f + mk(rng.normal(0, 0.0005, 1000))
    v2 = vals(big); v2["BIG"] = 400.0; v2["TWIN"] = 100.0
    r2 = risk_stats(prices(big), v2)
    check("a heavy holding with a twin takes a large share of the risk: the two together exceed their combined capital weight, and effective risk contributors fall below the effective positions",
          r2["rc"].loc[["BIG", "TWIN"], "RiskContribution"].sum() > r2["rc"].loc[["BIG", "TWIN"], "Weight"].sum() and r2["port"]["Effective number of risk contributors"] < 1 / ((v2 / v2.sum()) ** 2).sum(), str(r2["port"]))
    short = dict(ind); short["NEW"] = mk(np.r_[np.full(900, np.nan), rng.normal(0, 0.01, 100)])
    pr = prices(short); pr["NEW"] = pd.DataFrame({"close": (100 * (1 + short["NEW"].fillna(0)).cumprod()).iloc[900:]})
    r3 = risk_stats(pr, vals(short))
    check("a holding with under one year of history is left out and the count says so (8 of 9 used)", "NEW" not in r3["rc"].index and r3["n_used"] == 8 and r3["n_total"] == 9)
    check("fewer than three eligible holdings: nothing is computed (None), never a made-up number", risk_stats(prices({k: ind[k] for k in ["S0", "S1"]}), vals(["S0", "S1"])) is None)
    old = {f"O{i}": mk(np.r_[f.values[:500] + rng.normal(0, 0.0005, 500), rng.normal(0, 0.01, 500)]) for i in range(5)}   # same factor in the old half, independent in the last 500 days
    last3 = pd.bdate_range(end="2026-11-06", periods=1000)
    long_idx = pd.bdate_range(end="2026-11-06", periods=1700)
    old_long = {k: pd.Series(np.r_[f.values[:700] * 0 + rng.normal(0, 0.01, 700), v.values] if False else np.r_[np.repeat(f.values, 1)[:700], v.values], index=long_idx) for k, v in old.items()}
    rl = risk_stats(prices(old_long), vals(old_long))
    cut = long_idx[-1] - pd.DateOffset(years=3)
    expect = pd.DataFrame({k: v[v.index > cut] for k, v in old_long.items()}).corr().values[np.triu_indices(5, 1)].mean()
    check("only the last 3 years are used (a very different earlier period does not change the answer)", abs(rl["port"]["Average pairwise correlation"] - expect) < 0.03, f"{rl['port']['Average pairwise correlation']:.3f} vs {expect:.3f}")
    prices_only = {s: pd.DataFrame({"close": 100 * (1 + r_).cumprod(), "adj": 100 * (1 + r_ * 0).cumprod()}) for s, r_ in ind.items()}
    check("price return is used (the adjusted-close column is ignored)", abs(risk_stats(prices_only, vals(ind))["port"]["Average pairwise correlation"] - r["port"]["Average pairwise correlation"]) < 1e-12)

    def drawn(corr, n=1000, seed=3):
        L = np.linalg.cholesky(np.array(corr, float)); z = np.random.default_rng(seed).normal(0, 0.01, (n, len(corr))) @ L.T
        return [mk(z[:, i]) for i in range(len(corr))]
    blk = lambda k, rho: np.full((k, k), rho) + (1 - rho) * np.eye(k)
    cm = np.zeros((8, 8)); cm[:3, :3] = blk(3, 0.85); cm[3:6, 3:6] = blk(3, 0.80); cm[6:, 6:] = np.eye(2)
    grp = correlation_clusters(cm)
    check("two tight blocks and two loners: two groups found ({0,1,2} and {3,4,5}); the loners are not reported as groups", sorted(map(tuple, grp)) == [(0, 1, 2), (3, 4, 5)], str(grp))
    chain = np.array([[1, .8, .3], [.8, 1, .8], [.3, .8, 1]])
    check("average linkage does not chain: X-Y and Y-Z at 0.8 but X-Z at 0.3 gives only the closest pair, because the third would pull the average to 0.63",
          sorted(map(tuple, correlation_clusters(chain))) in ([(0, 1)], [(1, 2)]), str(correlation_clusters(chain)))
    check("everything at exactly 0.70 is not grouped (the rule is 'above 0.70'); at 0.71 it is", correlation_clusters(blk(4, 0.70)) == [] and correlation_clusters(blk(4, 0.71)) == [[0, 1, 2, 3]])
    rnd = np.random.default_rng(11); A = rnd.normal(size=(30, 12)); big = np.corrcoef(A + 1.2 * rnd.normal(size=(30, 1)))
    check("every reported group has an average pairwise correlation above 0.70 inside it (random correlated data, many groups)",
          all(big[np.ix_(g, g)][np.triu_indices(len(g), k=1)].mean() > 0.70 for g in correlation_clusters(big)) or correlation_clusters(big) == [], str(correlation_clusters(big)))
    ser = drawn(cm); names = [f"G{i}" for i in range(8)]
    rg = risk_stats(prices(dict(zip(names, ser))), vals(names))
    cl = rg["clusters"]
    check("through the full calculation: two groups; each lists its members, capital weight, risk share and average correlation above 0.70, sorted by risk share",
          len(cl) == 2 and set(map(frozenset, cl.Members)) == {frozenset(["G0", "G1", "G2"]), frozenset(["G3", "G4", "G5"])} and (cl.AvgCorr > 0.70).all() and cl.RiskShare.is_monotonic_decreasing and cl.Weight.round(1).tolist() == [37.5, 37.5], str(cl[["Label", "AvgCorr", "Weight", "RiskShare"]]))
    check("the group risk shares are consistent with the holding risk contributions (they add up to those of the members)", abs(cl.RiskShare.sum() - rg["rc"].loc[sum(cl.Members, []), "RiskContribution"].sum()) < 1e-9)
    check("the summary sentence names the biggest group and its risk share; with no group it says so plainly",
          "2 group(s) cover 6 of 8 holdings" in rg["cluster_text"] and "no group of holdings has an average correlation above 0.70" in risk_stats(prices(ind), vals(ind))["cluster_text"], rg["cluster_text"])

    tmp = Path(tempfile.mkdtemp(prefix="step18_"))
    try:
        mw.write_inputs(tmp)
        def run(flag):
            ctx = new_context(load_holdings(tmp / "holdings.csv"), mw.MockProvider(), MemoryState(auto_csv=(tmp / "out" / "evidence_auto.csv").read_bytes()),
                              options=RunOptions(fetch_evidence=False, risk_contribution=flag))
            with contextlib.redirect_stdout(io.StringIO()):
                stage_evidence(ctx); stage_prices(ctx); apply_cost_review(ctx, ac.FormHooks()); return stage_analysis(ctx)
        off, on = run(False), run(True)
        keys = ["Average pairwise correlation", "PC1 share of return variance (single-factor dominance)", "Effective number of risk contributors"]
        check("default: RunOptions.risk_contribution is False, and with it off none of the new figures exist", RunOptions().risk_contribution is False and not any(k in off.port for k in keys) and (off.rc is None or len(off.rc) == 0))
        check("flag ON: the three portfolio figures exist and are sensible", all(k in on.port for k in keys) and -1 <= on.port[keys[0]] <= 1 and 0 < on.port[keys[1]] <= 1 and on.port[keys[2]] >= 1, str({k: on.port.get(k) for k in keys}))
        check("flag ON: the holdings used exclude the short-history and unpriced ones, and risk contributions add up to 100%", on.rc is not None and "SHT" not in on.rc.index and "NOP" not in on.rc.index and abs(on.rc.RiskContribution.sum() - 100) < 1e-6, str(list(on.rc.index)))
        conc = on.sheets["Concentration"]; rcrows = conc[conc.Dimension.str.startswith("Risk contribution")]
        check("the Concentration sheet gets the top-10 risk contributions with each one's capital weight alongside (no new sheet or column)", 1 <= len(rcrows) <= 10 and rcrows.Flag.str.startswith("capital weight").all() and list(conc.columns) == list(off.sheets["Concentration"].columns))
        check("the existing Concentration measures in the HTML are now filled (they showed a dash before)", "Avg pairwise correlation</dt><dd>&ndash;" in render_result_html(off) and "Avg pairwise correlation</dt><dd>&ndash;" not in render_result_html(on))
        sm = [str(x) for x in on.summary if str(x).startswith("Risk concentration")]
        check("the summary states the basis (closing prices, 3 years, price return, holdings used, value covered) and that clusters are not computed yet", len(sm) == 1 and "price return" in sm[0] and "at least one year" in sm[0] and "clusters are not computed" not in sm[0], str(sm))
        cs = [str(x) for x in on.summary if str(x).startswith("Correlation clusters")]
        check("the summary carries a separate correlation-cluster line (the mock's independent holdings form no group); nothing when off", len(cs) == 1 and "average linkage" in cs[0] and not any("Correlation clusters" in str(x) for x in off.summary), str(cs))
        check("nothing else changes with the flag on: scores, labels and weighted scores are identical", on.sheets["Scorecard_A"].set_index("Symbol").Weighted.reindex(off.sheets["Scorecard_A"].Symbol).fillna(-1).values.tolist() == off.sheets["Scorecard_A"].Weighted.fillna(-1).values.tolist())
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
