"""Risk concentration statistics (Step 18, rows 1 to 4 approved on 8 Nov 2026).

Common basis: daily CLOSING prices (price return, dividends excluded: label it so), the last RISK_WINDOW_YEARS (3) years, and only
holdings with at least MIN_OBS (252, about one year) of daily returns in that window. Current market-value weights, renormalised over the
holdings used. Correlation and covariance are pairwise-complete.

  Average pairwise correlation   simple mean of the correlation of every pair of holdings used
  Risk contribution %            RC_i = w_i x (Cov w)_i / portfolio variance  (= w_i x vol_i x corr(i, portfolio) / portfolio vol); sums to 100%
  Effective number of risk contributors   1 / sum of squared RC_i  (read against the effective number of positions by capital, 1 / HHI)
  PC1 share of variance          largest eigenvalue / sum of eigenvalues of the covariance of WEIGHTED returns (w_i x r_i): how much of the
                                 weighted variance one common factor explains (single-factor dominance)
Not computed here: correlation clusters (threshold and grouping method still to be approved)."""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_OBS = 252
MIN_HOLDINGS = 3


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
    return dict(port=port, rc=out, n_used=len(used), n_total=len(H))
