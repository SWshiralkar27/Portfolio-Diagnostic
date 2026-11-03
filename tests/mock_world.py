"""A small SYNTHETIC world for check_step27.py: fake prices, fake Yahoo statements, fake NSE data, fake pop-ups.
No network and no real portfolio. Shapes follow the real Yahoo / dalal / NSE responses the engine reads."""
from __future__ import annotations
import json
import types
import zlib
from pathlib import Path
import numpy as np
import pandas as pd

TODAY = pd.Timestamp.today().normalize()
SCHEMA = 5

# symbol, target value (INR), buy-price factor vs last price, buy_date, sector, sector_type, asset_type, target_weight, broker xirr
HOLDINGS = [
    ("AAA", 12_000_000, 0.60, "2022-06-15", "Industrials", "standard", "equity", 9.0, 18.5),
    ("AAA", 2_000_000, 0.90, "2024-02-10", "Industrials", "standard", "equity", 9.0, 18.5),
    ("BBB", 6_000_000, 0.80, "", "Technology", "standard", "equity", None, None),
    ("CCC", 5_000_000, 0.70, "2023-01-02", "Financial Services", "bank", "equity", None, 14.0),
    ("DDD", 3_000_000, 0.85, "", "Energy", "cyclical", "equity", None, None),
    ("EEE", 4_000_000, None, "", "Industrials", "standard", "equity", None, None),     # average cost Rs 2: cost anomaly
    ("FFF", 3_000_000, 0.75, "2023-01-02", "Industrials", "standard", "equity", None, None),  # cash flow far below profit
    ("RED", 2_000_000, 0.90, "", "Healthcare", "standard", "equity", None, None),          # auditor resignation disclosure
    ("ILQ", 8_000_000, 0.80, "", "Industrials", "standard", "equity", None, None),         # illiquid, ASM
    ("SPL", 3_000_000, 0.70, "2023-01-02", "Technology", "standard", "equity", None, None),   # split in the history
    ("JMP", 3_000_000, 0.95, "2023-01-02", "Energy", "cyclical", "equity", None, None),       # unexplained one-day fall
    ("NOP", 1_000_000, 0.80, "", "Technology", "standard", "equity", None, None),          # no price returned
    ("SHT", 2_000_000, 0.90, "", "Technology", "standard", "equity", None, None),          # short price history
    ("CSH", 2_000_000, None, "", "", "", "cash", None, None),
    ("GGG", 4_000_000, 0.70, "2023-01-02", "Technology", "standard", "equity", None, None),
    ("HHH", 4_000_000, 0.70, "2023-01-02", "Technology", "standard", "equity", None, None),
    ("NEG", 2_000_000, 0.90, "", "Industrials", "standard", "equity", None, None),         # negative net worth
]
INDEX_TICKERS = {"^CRSLDX", "^NSEI", "^CNX100", "NIFTYMIDCAP150.NS", "^NSEMDCP50", "MID150BEES.NS", "NIFTYSMLCAP250.NS", "^CNXSC", "HDFCSML250.NS"}
NSE_STATE = {"ILQ": "ASM"}
PROMOTER = {"AAA": (50.4, 50.4), "BBB": (62.0, 62.5), "DDD": (55.0, 60.0), "EEE": (40.0, 41.0), "FFF": (71.0, 70.0), "RED": (45.0, 45.0),
            "ILQ": (60.0, 60.0), "SPL": (30.0, 31.0), "JMP": (52.0, 52.0), "SHT": (66.0, 66.0), "GGG": (44.0, 48.0), "HHH": (58.0, 57.5), "NEG": (51.0, 51.0)}


def _rng(key):
    return np.random.default_rng(zlib.crc32(str(key).encode()))


def _bare(t):
    return str(t).replace(".NS", "").replace(".BO", "")


def _hist(tkr):
    b = _bare(tkr)
    if b == "NOP": return None
    n = 120 if b == "SHT" else 1600
    rng = _rng(tkr)
    idx = pd.bdate_range(end=TODAY, periods=n)
    vol = 0.010 if tkr in INDEX_TICKERS else 0.018
    close = (100 + zlib.crc32(b.encode()) % 300) * np.exp(np.cumsum(rng.normal(0.0004, vol, n)))
    if b == "JMP": close[n - 300:] *= 0.6
    adj = close * np.linspace(0.85, 1.0, n)
    volume = rng.integers(300_000, 900_000, n).astype(float)
    if b == "ILQ": volume = rng.integers(2_000, 4_000, n).astype(float)
    div = np.zeros(n)
    if b in ("AAA", "GGG"): div[[n - 400, n - 150]] = 2.0
    split = np.zeros(n)
    if b == "SPL": split[n - 700] = 2.0
    return pd.DataFrame(dict(close=close, adj=adj, high=close * 1.01, low=close * 0.99, volume=volume, div=div, split=split), index=idx)


def _statements(tkr):
    b = _bare(tkr)
    cols = pd.to_datetime(["2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"])
    rev = np.array([0.7, 0.8, 0.9, 1.0]) * 5e10
    ni = rev * 0.10
    cfo = rev * 0.12
    equity = rev * 0.8
    if b == "FFF": cfo = np.array([-1e8, -2e8, -3e8, -5e8])
    if b == "NEG": equity = np.array([-1e9, -1e9, -1e9, -1.5e9])
    fin = pd.DataFrame([rev, rev * 0.2, rev * 0.15, ni, rev * 0.01], index=["Total Revenue", "EBITDA", "EBIT", "Net Income", "Interest Expense"], columns=cols)
    bs = pd.DataFrame([rev * 0.2, rev * 0.05, equity, rev * 1.2, rev * 0.3], index=["Total Debt", "Cash And Cash Equivalents", "Stockholders Equity", "Total Assets", "Current Liabilities"], columns=cols)
    cf = pd.DataFrame([cfo, cfo * 0.6], index=["Operating Cash Flow", "Free Cash Flow"], columns=cols)
    return dict(fin=fin, bs=bs, cf=cf)


class MockProvider:
    name = "Mock"

    def __init__(self, cache_dir=None, use_cache=True):
        self.nse = types.SimpleNamespace(available=True)

    def history(self, tkr, start, end):
        return _hist(tkr)

    def info(self, tkr):
        b = _bare(tkr); k = zlib.crc32(b.encode())
        return dict(trailingPE=10 + k % 40, priceToBook=1 + (k % 50) / 10, enterpriseToEbitda=6 + k % 12, enterpriseToRevenue=1 + (k % 30) / 10,
                    trailingEps=5.0, returnOnEquity=0.14, marketCap=(2e12 if b == "AAA" else 5e11 if k % 2 else 1e11),
                    sector=("Industrials" if k % 3 == 0 else "Technology"), regularMarketPrice=100.0)

    def statements(self, tkr):
        return _statements(tkr)

    def surveillance(self, symbol):
        return NSE_STATE.get(_bare(symbol), "")

    def corporate_actions(self, symbol):
        return [dict(exDate="01-Jan-2025", subject="Stock split from Rs 10 to Rs 5")] if _bare(symbol) == "SPL" else []


class FakeTicker:
    def __init__(self, sym):
        self.sym = sym

    @property
    def calendar(self):
        return {"Earnings Date": [(TODAY + pd.Timedelta(days=10)).date()]}


class FakeDalal:
    @staticmethod
    def shareholding(sym):
        now, ago = PROMOTER.get(sym, (None, None))
        if now is None: return []
        rows = [("30-SEP-2026", now), ("30-JUN-2026", now), ("31-MAR-2026", (now + ago) / 2), ("31-DEC-2025", ago), ("30-SEP-2025", ago)]
        return [{"date": d, "pr_and_prgrp": f"{v:.2f}", "public_val": f"{100 - v:.2f}"} for d, v in rows]

    @staticmethod
    def announcements(sym):
        d = (TODAY - pd.Timedelta(days=20)).strftime("%d-%b-%Y 10:00:00")
        if sym == "RED":
            return [dict(an_dt=d, desc="Cessation", attchmntText="Resignation of the statutory auditors of the Company with immediate effect", attchmntFile="https://example.test/red.pdf")]
        return [dict(an_dt=d, desc="General Updates", attchmntText="Disclosure under Regulation 30 of SEBI LODR: outcome of board meeting", attchmntFile="")]


class FakeIA:
    """Stand-in for input_assistant.py: records the calls, changes nothing except the cost review (EEE confirmed)."""
    def __init__(self):
        self.calls = []

    def check_holdings_layout(self, path):
        self.calls.append(("check_holdings_layout",))

    def apply_saved_buy_dates(self, lots, folder):
        self.calls.append(("apply_saved_buy_dates",)); return lots

    def run_prompts(self, lots, evid, peers, folder=None, mode="auto", evidence_path=None, peers_path=None):
        self.calls.append(("run_prompts", mode)); return lots, evid, peers

    def review_costs(self, HD, cfg, folder=None, mode="auto"):
        self.calls.append(("review_costs", mode)); return HD, {"EEE"}, {}


def _auto_rows():
    rows = []
    for sym in sorted({h[0] for h in HOLDINGS if h[6] == "equity"}):
        now, ago = PROMOTER.get(sym, (np.nan, np.nan))
        ann = ""; flag_log = "[]"
        if sym == "RED":
            ann = "RED_FLAG: auditor resignation (%s) [matched: 'resignation of the statutory auditors']: Resignation of the statutory auditors of the Company" % (TODAY - pd.Timedelta(days=20)).strftime("%d-%b")
            flag_log = json.dumps([dict(kind="red", label="auditor resignation", matched="resignation of the statutory auditors", date=(TODAY - pd.Timedelta(days=20)).strftime("%Y-%m-%d"), url="https://example.test/red.pdf", text="Resignation of the statutory auditors")])
        rows.append(dict(symbol=sym, promoter_holding_now=now, promoter_holding_1y=ago, promoter_holding_asof="2026-09-30" if now == now else "",
                         next_earnings=str((TODAY + pd.Timedelta(days=10)).date()), announcements=ann, ann_watch="", flag_log=flag_log,
                         fetched_at=pd.Timestamp.now().isoformat(timespec="seconds"), fetch_notes="ok", schema=SCHEMA))
    return pd.DataFrame(rows)


def write_inputs(work):
    work = Path(work); (work / "out").mkdir(parents=True, exist_ok=True)
    rows = []
    for sym, val, f, bdate, sec, stype, atype, tgt, xi in HOLDINGS:
        if atype == "cash":
            price, qty = 2_000_000.0, 1
        else:
            h = _hist(sym + ".NS")
            last = float(h.close.iloc[-1]) if h is not None else 100.0
            qty = max(1, int(round(val / last)))
            price = 2.0 if f is None else round(last * f, 2)
        rows.append(dict(symbol=sym, quantity=qty, buy_price=price, buy_date=bdate, sector=sec, sector_type=stype, asset_type=atype,
                         target_weight_pct=tgt if tgt is not None else "", xirr=xi if xi is not None else ""))
    pd.DataFrame(rows).to_csv(work / "holdings.csv", index=False)
    pd.DataFrame([dict(symbol="DDD", promoter_pledge_pct=40, pledge_rising_2q="yes"),
                  dict(symbol="CCC", gnpa_pct=2.1, car_pct=16.2),
                  dict(symbol="AAA", sales_growth_q1=18, sales_growth_q2=20, sales_growth_q3=17, sales_growth_q4=22)]).to_csv(work / "evidence.csv", index=False)
    pd.DataFrame([dict(symbol="GGG", peer="AAA"), dict(symbol="GGG", peer="HHH"), dict(symbol="HHH", peer="GGG"), dict(symbol="HHH", peer="AAA")]).to_csv(work / "peers.csv", index=False)
    _auto_rows().to_csv(work / "out" / "evidence_auto.csv", index=False)
