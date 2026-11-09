"""Risk concentration statistics (Step 18, rows 1 to 4 approved on 8 Nov 2026).

Common basis: daily CLOSING prices (price return, dividends excluded: label it so), the last RISK_WINDOW_YEARS (3) years, and only
holdings with at least MIN_OBS (252, about one year) of daily returns in that window. Current market-value weights, renormalised over the
holdings used. Correlation and covariance are pairwise-complete.

  Average pairwise correlation   simple mean of the correlation of every pair of holdings used
  Risk contribution %            RC_i = w_i x (Cov w)_i / portfolio variance  (= w_i x vol_i x corr(i, portfolio) / portfolio vol); sums to 100%
  Effective number of risk contributors   1 / sum of squared RC_i  (read against the effective number of positions by capital, 1 / HHI)
  PC1 share of variance          largest eigenvalue / sum of eigenvalues of the covariance of WEIGHTED returns (w_i x r_i): how much of the
                                 weighted variance one common factor explains (single-factor dominance)
  Correlation clusters (approved 8 Nov 2026, option A): average-linkage grouping. Start with every holding alone and keep merging the two groups
                                 whose AVERAGE pairwise correlation is highest, as long as it is above CLUSTER_CORR (0.70). Every group reported has an
                                 average pairwise correlation above 0.70 inside it. Single holdings are not reported as groups."""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_OBS = 252
MIN_HOLDINGS = 3
CLUSTER_CORR = 0.70


def correlation_clusters(corr, threshold=CLUSTER_CORR):
    """Average-linkage groups of indices; corr is a square matrix (NaN counts as 0). Returns a list of index lists, each with 2 or more members."""
    C = np.where(np.isfinite(corr), corr, 0.0).astype(float)
    n = C.shape[0]
    np.fill_diagonal(C, 1.0)
    S, size, members, active = C.copy(), np.ones(n), [[i] for i in range(n)], list(range(n))
    while len(active) > 1:
        best, pair = -2.0, None
        for ii, a in enumerate(active):
            for b in active[ii + 1:]:
                avg = S[a, b] / (size[a] * size[b])
                if avg > best: best, pair = avg, (a, b)
        if best <= threshold: break
        a, b = pair
        members[a] += members[b]; size[a] += size[b]
        for c in active:
            if c not in (a, b): S[a, c] = S[c, a] = S[a, c] + S[b, c]
        active.remove(b)
    return [sorted(members[a]) for a in active if len(members[a]) > 1]


def _cluster_table(groups, C, used, w, rc):
    rows = []
    for g in groups:
        sub = C[np.ix_(g, g)]
        avg = float(np.nanmean(sub[np.triu_indices(len(g), k=1)]))
        names = sorted([used[i] for i in g], key=lambda s: -rc[used.index(s)])
        label = ", ".join(names[:8]) + (f" (+{len(names) - 8} more)" if len(names) > 8 else "")
        rows.append(dict(Members=names, N=len(g), AvgCorr=avg, Weight=float(w[g].sum() * 100), RiskShare=float(rc[g].sum() * 100), Label=label))
    t = pd.DataFrame(rows, columns=["Members", "N", "AvgCorr", "Weight", "RiskShare", "Label"])
    if len(t):
        t = t.sort_values("RiskShare", ascending=False).reset_index(drop=True)
        t["Label"] = [f"C{i + 1}: {l}" for i, l in enumerate(t.Label)]
        t["Flag"] = [f"risk share {r.RiskShare:.1f}%; average correlation {r.AvgCorr:.2f}; {r.N} holdings" for r in t.itertuples()]
    else:
        t["Flag"] = []
    return t


def _cluster_text(t, n_used):
    basis = "Correlation clusters (average linkage, average correlation above 0.70, same basis)"
    if not len(t): return f"{basis}: no group of holdings has an average correlation above 0.70."
    top = t.iloc[0]
    return (f"{basis}: {len(t)} group(s) cover {int(t.N.sum())} of {n_used} holdings, {t.Weight.sum():.1f}% of capital and {t.RiskShare.sum():.1f}% of risk. "
            f"Largest by risk: {', '.join(top.Members[:6])}{' ...' if top.N > 6 else ''} ({top.RiskShare:.1f}% of risk, {top.Weight:.1f}% of capital, average correlation {top.AvgCorr:.2f}).")


def risk_stats(H, values, years=3, min_obs=MIN_OBS):
    """H: {symbol: price DataFrame with 'close'}; values: Series of current market values by symbol. Returns None if too few holdings qualify."""
    rets = {}
    for s, h in H.items():
        c = h["close"].dropna()
        if len(c) > 1: rets[s] = c.pct_change().dropna()
    if not rets: return None
    last = max(r.index[-1] for r in rets.values())
    start = last - pd.DateOffset(years=years)
    R = pd.concat({s: r[r.index > start] for s, r in rets.items()}, axis=1)
    used = [s for s in R.columns if R[s].notna().sum() >= min_obs and s in values.index and values[s] > 0]
    if len(used) < MIN_HOLDINGS: return None
    R = R[used]
    corr = R.corr(min_periods=min_obs)
    iu = np.triu_indices(len(used), k=1)
    pairs = corr.values[iu]
    avg_corr = float(np.nanmean(pairs)) if np.isfinite(pairs).any() else float("nan")
    cov = R.cov(min_periods=min_obs).values * 252.0
    cov = np.where(np.isfinite(cov), cov, 0.0)
    v = values[used].astype(float)
    w = (v / v.sum()).values
    mc = cov @ w
    var_p = float(w @ mc)
    if not var_p > 0: return None
    rc = w * mc / var_p
    eff_rc = float(1.0 / np.sum(rc ** 2))
    C = (w[:, None] * cov) * w[None, :]
    ev = np.clip(np.linalg.eigvalsh((C + C.T) / 2.0), 0.0, None)
    pc1 = float(ev.max() / ev.sum()) if ev.sum() > 0 else float("nan")
    out = pd.DataFrame(dict(Weight=w * 100.0, Vol=np.sqrt(np.diag(cov)), RiskContribution=rc * 100.0), index=used)
    total_val = float(values[values.index.isin(H.keys())].sum())
    port = {"Average pairwise correlation": avg_corr,
            "PC1 share of return variance (single-factor dominance)": pc1,
            "Effective number of risk contributors": eff_rc,
            "Risk statistics: holdings used (of those with prices)": float(len(used)),
            "Risk statistics: share of priced equity value covered (%)": float(v.sum() / total_val * 100.0) if total_val > 0 else float("nan"),
            "Risk statistics: portfolio volatility of those holdings (annual, price return)": float(np.sqrt(var_p))}
    groups = correlation_clusters(corr.values)
    tab = _cluster_table(groups, np.where(np.isfinite(corr.values), corr.values, 0.0), used, w, rc)
    return dict(port=port, rc=out, n_used=len(used), n_total=len(H), clusters=tab, cluster_text=_cluster_text(tab, len(used)))
