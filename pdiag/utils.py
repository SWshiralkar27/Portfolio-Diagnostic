"""Returns, drawdown, XIRR, CAPM and other small helpers."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def fnum(x, d=np.nan):
    try: return float(x) if np.isfinite(float(x)) else d
    except: return d

def has(x) -> bool: return x is not None and str(x).strip().lower() not in ("", "nan", "none", "nat")

def yes(x) -> bool: return str(x).strip().lower() in ("y", "yes", "true", "1", "t")

def bucket(v, cuts, scores):
    for c, s in zip(cuts, scores):
        if v < c: return s
    return scores[-1]

def pct(x, d=1): return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"

def to_ticker(sym) -> str:
    s = str(sym).strip().upper()
    return s if (s.startswith("^") or s.endswith(".NS") or s.endswith(".BO")) else s + ".NS"

def bare(sym) -> str: return str(sym).strip().upper().replace(".NS", "").replace(".BO", "")

def hist_years(s: pd.Series) -> float: return (s.index[-1] - s.index[0]).days / 365.25 if len(s) > 1 else 0.0

def cagr(s: pd.Series, years: int) -> float:
    s = s.dropna()
    if len(s) < 2: return np.nan
    end = s.index[-1]; target = end - pd.DateOffset(years=years)
    if s.index[0] > target + pd.Timedelta(days=10): return np.nan
    p0 = s[:target].iloc[-1] if (s.index <= target).any() else s.iloc[0]
    return (s.iloc[-1] / p0) ** (1 / years) - 1 if p0 > 0 else np.nan

def total_return_since(s: pd.Series) -> float:
    s = s.dropna(); return s.iloc[-1] / s.iloc[0] - 1 if len(s) > 1 else np.nan

def approx_tr(price: pd.Series, y: float) -> pd.Series:
    t = (price.index - price.index[0]).days.values / 365.25
    return price * np.exp(y * t)

def max_drawdown(s: pd.Series) -> float:
    s = s.dropna(); return float((s / s.cummax() - 1).min()) if len(s) else np.nan

def monthly(s: pd.Series) -> pd.Series: return s.groupby(s.index.to_period("M")).last()

def xirr(flows):
    fl = [(pd.Timestamp(d), float(a)) for d, a in flows if pd.notna(d) and a and np.isfinite(a)]
    if len(fl) < 2 or not (any(a < 0 for _, a in fl) and any(a > 0 for _, a in fl)): return np.nan
    t0 = min(d for d, _ in fl); t = np.array([(d - t0).days / 365.0 for d, _ in fl]); a = np.array([x for _, x in fl])
    f = lambda r: float(np.sum(a / (1 + r) ** t))
    lo, hi = -0.95, 50.0; flo, fhi = f(lo), f(hi)
    if flo * fhi > 0: return np.nan
    mid = np.nan
    for _ in range(200):
        mid = (lo + hi) / 2; fm = f(mid)
        if abs(fm) < 1e-7: break
        if flo * fm <= 0: hi = mid
        else: lo, flo = mid, fm
    return mid

def capm(rs: pd.Series, rm: pd.Series, rf: float, min_obs=120) -> dict:
    df = pd.concat([rs, rm], axis=1, join="inner").dropna()
    if len(df) < min_obs: return {}
    rf_d = (1 + rf) ** (1 / 252) - 1; y, x = df.iloc[:, 0].values - rf_d, df.iloc[:, 1].values - rf_d
    xm, ym = x.mean(), y.mean(); sxx = ((x - xm) ** 2).sum(); beta = ((x - xm) * (y - ym)).sum() / sxx; a = ym - beta * xm
    resid = y - (a + beta * x); n = len(df); s2 = (resid ** 2).sum() / (n - 2); se_a = math.sqrt(s2 * (1 / n + xm ** 2 / sxx))
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return dict(beta=beta, alpha_ann=a * 252, alpha_t=a / se_a if se_a > 0 else np.nan, r2=r2, idio_vol=resid.std(ddof=1) * math.sqrt(252))

def capture(port_wealth: pd.Series, bench_wealth: pd.Series):
    df = pd.concat([monthly(port_wealth).pct_change(), monthly(bench_wealth).pct_change()], axis=1, join="inner").dropna()
    df.columns = ["p", "b"]; up, dn = df[df.b > 0], df[df.b < 0]
    return (up.p.mean() / up.b.mean() if len(up) >= 6 else np.nan, dn.p.mean() / dn.b.mean() if len(dn) >= 6 else np.nan)

def clean_hist(h):
    if h is None or len(h) == 0: return None
    h = h.copy(); idx = pd.to_datetime(h.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    h.index = idx.normalize(); h = h[~h.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame(index=h.index)
    out["close"] = h["Close"]; out["adj"] = h["Adj Close"] if "Adj Close" in h else h["Close"]
    out["high"] = h["High"] if "High" in h else h["Close"]; out["low"] = h["Low"] if "Low" in h else h["Close"]
    out["volume"] = h["Volume"] if "Volume" in h else 0.0
    out["div"] = h["Dividends"] if "Dividends" in h else 0.0
    out["split"] = h["Stock Splits"] if "Stock Splits" in h else 0.0
    out = out.dropna(subset=["close"])
    return out if len(out) else None
