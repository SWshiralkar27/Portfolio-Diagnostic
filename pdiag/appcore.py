"""Helpers behind the Streamlit page. Pure Python (no Streamlit import) so check_step28.py can test them.

Carry-over pack = one .zip the user downloads at the end of a run and uploads at the next:
  evidence_auto.csv     NSE evidence already fetched (so the next run only fetches stocks older than 7 days)
  last_run_snapshot.csv this run's scores (so the next run can show Score change)
  buy_dates.csv         buy dates the user typed (symbol, quantity, buy_price, buy_date)
  cost_decisions.csv    cost lines the user confirmed or corrected (symbol, decision, factor)
  company_info.json     last good Yahoo company info per stock (used, labelled with its age, when Yahoo does not answer)
  pack_info.json        tool version and date
The pack holds private information (dates, costs, the list of stocks). It is for the user's own device only.
"""
from __future__ import annotations
import datetime as dt
import io
import json
import re
import zipfile
import numpy as np
import pandas as pd
from pdiag.config import VERSION
from pdiag.errors import FatalInputError
from pdiag.hooks import NullHooks
from pdiag.loaders import read_table, evidence_from_df, peers_from_df
from pdiag.report_excel import build_xlsx_bytes
from pdiag.report_html import render_result_html
from pdiag.utils import has

LABEL = ("Provisional portfolio diagnostic and research-support report. Not personalized investment advice. "
         "Not a recommendation to buy, sell, or hold any security.")
HOLDINGS_TEMPLATE = (
    "symbol,quantity,buy_price,buy_date,xirr,target_weight_pct,asset_type,sector,sector_type,cap_tier,benchmark,surveillance,yahoo_ticker,horizon\n"
    "EXAMPLE1,100,250.50,2022-05-10,14.2,5,equity,,,,,,,\n"
    "EXAMPLE1,40,310.00,2024-01-15,14.2,5,equity,,,,,,,\n"
    "CASH,1,50000,,,,cash,,,,,,,\n")
BUY_COLS = ["symbol", "quantity", "buy_price", "buy_date"]
DEC_COLS = ["symbol", "decision", "factor"]
COST_COLS = ["symbol", "avg_cost", "price", "pnl_pct", "flag", "confirm", "factor"]


def progress_to(bar):
    """A progress callback (stage, i, n, label) that drives a Streamlit progress bar."""
    def cb(stage, i, n, label):
        try:
            bar.progress(min(1.0, i / max(n, 1)), text=f"{stage}: {i} of {n} ({label})")
        except Exception:
            pass
    return cb


def prefetch_info(ctx):
    """Ask Yahoo for company info FIRST, before any other Yahoo call (prices, results dates). The provider keeps the answers,
    so the later stages use them without asking again. If Yahoo refuses, the very first request tells us whether it was refused
    from the start or only after this run's other requests."""
    from pdiag.utils import to_ticker
    eq = ctx.lots[~ctx.lots.asset_type.isin(["cash", "bond"])]
    tickers, seen = [], set()
    for s, yt in zip(eq.symbol, eq.yahoo_ticker):
        t = to_ticker(yt if has(yt) else s)
        if t not in seen: seen.add(t); tickers.append(t)
    for plist in ctx.peers.values():
        for p in plist:
            t = to_ticker(p)
            if t not in seen: seen.add(t); tickers.append(t)
    for i, t in enumerate(tickers, 1):
        ctx.progress("company info", i, len(tickers), t)
        ctx.provider.info(t)
    return len(tickers)


# ---------------- input files: plain errors, and the wide peers layout ----------------
_SYM = re.compile(r"^[A-Z0-9][A-Z0-9&\-]*$")
_USE = re.compile(r"\buse\s+([A-Za-z0-9&\-]+)\s*$", re.I)
TICKER_COLS = ("symbol", "nse_ticker", "ticker", "nse_symbol")
PLACEHOLDERS = {"VERIFY", "TBD", "NA", "N/A", "NONE", "NIL", "UNKNOWN"}      # words people type instead of a symbol


def _cols(df):
    return ", ".join(str(c) for c in list(df.columns)[:8])


def read_evidence(f):
    df = read_table(f)
    if "symbol" not in df.columns:
        raise FatalInputError(f"The evidence file needs a column named symbol (one row per stock). Yours has: {_cols(df)}.")
    return evidence_from_df(df)


def read_peers(f, held=None):
    """Peers as {symbol: [peer, ...]} plus a report of what was kept or left out.
    Accepts one row per peer (columns symbol, peer) or one row per company (a ticker column, then Peer 1, Peer 2 ...).
    In the wide layout a peer cell that carries a doubt ('PIDILITIND?') is left out; the file's own correction ('... Use HAVELLS') is used;
    an own ticker written 'XYZ? (verify)' is kept only if XYZ is exactly a symbol in your holdings. Nothing is guessed."""
    df = read_table(f)
    cols = list(df.columns)
    if {"symbol", "peer"} <= set(cols):
        return peers_from_df(df), dict(layout="long", pairs=int(len(df)), left_out=[], used=[], kept=[])
    tcol = next((c for c in TICKER_COLS if c in cols), None)
    pcols = [c for c in cols if re.fullmatch(r"peer_?\d+", c)]
    if tcol is None or not pcols:
        raise FatalInputError("The peers file needs either two columns named symbol and peer (one row per peer), or a ticker column "
                              f"(NSE Ticker) followed by Peer 1, Peer 2 ... columns. Yours has: {_cols(df)}.")
    held = {str(s).upper() for s in (held or [])}
    pairs, left_out, used, kept = [], [], [], []
    for _, r in df.iterrows():
        raw_t = str(r[tcol]).strip(); t = raw_t.upper()
        if t in PLACEHOLDERS or not _SYM.match(t):
            tok = re.split(r"[?\s(]", t)[0]
            if tok not in PLACEHOLDERS and _SYM.match(tok) and tok in held: kept.append(tok); t = tok
            else: left_out.append((raw_t, "own ticker", raw_t)); continue
        for pc in pcols:
            raw = r[pc]
            if raw is None or str(raw).strip() in ("", "nan"): continue
            p = str(raw).strip(); pu = p.upper()
            if pu in PLACEHOLDERS: left_out.append((t, pc, p)); continue
            if _SYM.match(pu): q = pu
            else:
                m = _USE.search(p)
                if m and _SYM.match(m.group(1).upper()): q = m.group(1).upper(); used.append((t, p, q))
                else: left_out.append((t, pc, p)); continue
            if q != t: pairs.append((t, q))
    pairs = list(dict.fromkeys(pairs))
    peers = {}
    for t, q in pairs: peers.setdefault(t, []).append(q)
    return peers, dict(layout="wide", pairs=len(pairs), left_out=left_out, used=used, kept=kept)


def peers_note(rep):
    if rep.get("layout") != "wide": return ""
    ex = "; ".join(f"{a} {b}: {c}" for a, b, c in rep["left_out"][:4])
    return (f"Peers file read in the one-row-per-company layout: {rep['pairs']} peer pairs. {len(rep['left_out'])} entries were left out because the file marks them uncertain"
            f" ({ex}{' ...' if len(rep['left_out']) > 4 else ''}); {len(rep['used'])} used the file's own correction; {len(rep['kept'])} own tickers marked '? (verify)' were matched to your holdings. "
            "Stocks with fewer than 5 peers are low-confidence peer comparisons.")


# ---------------- Step 3 safety: preview of a cost correction, and re-running the step safely ----------------
def cost_preview(HD, edits, cfg):
    """What each correction factor would do, before the analysis runs (catches a factor typed the wrong way round)."""
    rows = []
    for r in edits.itertuples():
        if pd.notna(r.factor) and float(r.factor) > 0 and r.symbol in HD.index:
            h = HD.loc[r.symbol]; new = float(h.AvgCost) * float(r.factor)
            pnl = (float(h.Value) / (float(h.Qty) * new) - 1) * 100 if new > 0 and float(h.Qty) > 0 else float("nan")
            still = new < cfg["ANOM_MIN_AVGCOST"] or pnl > cfg["ANOM_MAX_GAIN_PCT"] or pnl < cfg["ANOM_MIN_PNL_PCT"]
            rows.append(dict(symbol=r.symbol, avg_cost_now=float(h.AvgCost), avg_cost_after=new, price=float(h.CMP), pnl_pct_after=pnl,
                             note=("still looks wrong: check the factor" if still else "ok")))
    return pd.DataFrame(rows, columns=["symbol", "avg_cost_now", "avg_cost_after", "price", "pnl_pct_after", "note"])


def snapshot_ctx(ctx):
    return (ctx.lots.copy(), ctx.HD.copy(), list(ctx.gaps), list(ctx.notes_global))


def restore_ctx(ctx, snap):
    """Put the pre-Step-3 tables back, so pressing Run twice (or after an interrupted run) cannot apply a correction twice."""
    ctx.lots, ctx.HD, ctx.gaps, ctx.notes_global = snap[0].copy(), snap[1].copy(), list(snap[2]), list(snap[3])


# ---------------- carry-over pack ----------------
def empty_pack():
    return dict(auto_csv=None, snapshot_csv=None, buy_dates=pd.DataFrame(columns=BUY_COLS), cost_decisions=pd.DataFrame(columns=DEC_COLS), company_info={}, info={})


def _bytes(src):
    if isinstance(src, (bytes, bytearray)): return bytes(src)
    if hasattr(src, "seek"): src.seek(0)
    return src.read()


def _csv(zf, name, cols):
    if name not in zf.namelist(): return pd.DataFrame(columns=cols)
    try: df = pd.read_csv(io.BytesIO(zf.read(name)))
    except pd.errors.EmptyDataError: return pd.DataFrame(columns=cols)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df[cols]


def parse_pack(src):
    try: zf = zipfile.ZipFile(io.BytesIO(_bytes(src)))
    except zipfile.BadZipFile: raise FatalInputError("The carry-over pack is not a valid .zip file. Upload the .zip you downloaded at the end of your last run.")
    names = set(zf.namelist())
    bd = _csv(zf, "buy_dates.csv", BUY_COLS)
    bd["buy_date"] = pd.to_datetime(bd["buy_date"], errors="coerce")
    for c in ("quantity", "buy_price"): bd[c] = pd.to_numeric(bd[c], errors="coerce")
    bd["symbol"] = bd["symbol"].astype(str).str.strip().str.upper()
    dec = _csv(zf, "cost_decisions.csv", DEC_COLS)
    dec["symbol"] = dec["symbol"].astype(str).str.strip().str.upper()
    dec["factor"] = pd.to_numeric(dec["factor"], errors="coerce")
    try: info = json.loads(zf.read("pack_info.json")) if "pack_info.json" in names else {}
    except Exception: info = {}
    try: cinfo = json.loads(zf.read("company_info.json")) if "company_info.json" in names else {}
    except Exception: cinfo = {}
    cinfo = {k: v for k, v in cinfo.items() if isinstance(v, dict) and isinstance(v.get("info"), dict) and v.get("fetched")}
    return dict(auto_csv=zf.read("evidence_auto.csv") if "evidence_auto.csv" in names else None,
                snapshot_csv=zf.read("last_run_snapshot.csv") if "last_run_snapshot.csv" in names else None,
                buy_dates=bd.dropna(subset=["buy_date"]).reset_index(drop=True), cost_decisions=dec.reset_index(drop=True), company_info=cinfo, info=info)


def build_pack(state, buy_dates, cost_decisions, company_info=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in state.carry_over_csv().items():
            zf.writestr(name, text)
        bd = buy_dates.copy()
        if len(bd): bd["buy_date"] = pd.to_datetime(bd["buy_date"]).dt.strftime("%Y-%m-%d")
        zf.writestr("buy_dates.csv", bd[BUY_COLS].to_csv(index=False))
        zf.writestr("cost_decisions.csv", cost_decisions[DEC_COLS].to_csv(index=False))
        zf.writestr("company_info.json", json.dumps(company_info or {}))
        zf.writestr("pack_info.json", json.dumps(dict(tool_version=VERSION, created=dt.datetime.now().isoformat(timespec="seconds"),
                                                      note="Private: buy dates, cost decisions and your list of stocks. Keep on your own device.")))
    return buf.getvalue()


# ---------------- buy dates ----------------
def lot_keys(lots):
    keys = []
    for s, q, p in zip(lots.symbol, lots.quantity, lots.buy_price):
        keys.append((str(s).upper(), round(float(q), 4), round(float(p), 2)) if pd.notna(q) and pd.notna(p) else None)
    return keys


def apply_saved_buy_dates(lots, buy_dates):
    """Fill missing buy dates from the pack (matched on symbol + quantity + buy price). Returns (lots, how many filled)."""
    if buy_dates is None or buy_dates.empty: return lots, 0
    saved = {}
    for r in buy_dates.itertuples():
        if pd.notna(r.buy_date) and pd.notna(r.quantity) and pd.notna(r.buy_price):
            saved[(str(r.symbol).upper(), round(float(r.quantity), 4), round(float(r.buy_price), 2))] = pd.Timestamp(r.buy_date)
    lots, n = lots.copy(), 0
    for i, k in zip(lots.index, lot_keys(lots)):
        if k in saved and pd.isna(lots.at[i, "buy_date"]):
            lots.at[i, "buy_date"] = saved[k]; n += 1
    return lots, n


def undated_lots(lots):
    m = lots.buy_date.isna() & ~lots.asset_type.isin(["cash", "bond"])
    d = lots.loc[m, ["symbol", "quantity", "buy_price"]].copy()
    d.insert(0, "lot", d.index)
    d["buy_date"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    return d.reset_index(drop=True)


def apply_buy_date_edits(lots, edits):
    lots, n = lots.copy(), 0
    for r in edits.itertuples():
        if pd.notna(r.buy_date):
            lots.at[int(r.lot), "buy_date"] = pd.Timestamp(r.buy_date); n += 1
    return lots, n


def merge_pack_dates(old, lots, edits):
    rows = [dict(symbol=r.symbol, quantity=r.quantity, buy_price=r.buy_price, buy_date=r.buy_date) for r in old.itertuples()] if len(old) else []
    for r in edits.itertuples():
        if pd.notna(r.buy_date) and pd.notna(lots.at[int(r.lot), "buy_price"]):
            l = lots.loc[int(r.lot)]
            rows.append(dict(symbol=str(l.symbol).upper(), quantity=float(l.quantity), buy_price=round(float(l.buy_price), 2), buy_date=pd.Timestamp(r.buy_date)))
    df = pd.DataFrame(rows, columns=BUY_COLS)
    return df.drop_duplicates(subset=["symbol", "quantity", "buy_price"], keep="last").reset_index(drop=True)


# ---------------- promoter holding the NSE fetch could not supply ----------------
def missing_promoter(evid, lots):
    syms = sorted(set(lots.loc[~lots.asset_type.isin(["cash", "bond"]), "symbol"].astype(str)))
    rows = [dict(symbol=s, promoter_holding_now=np.nan, promoter_holding_1y=np.nan) for s in syms
            if not (has(evid.get(s, {}).get("promoter_holding_now")) and has(evid.get(s, {}).get("promoter_holding_1y")))]
    return pd.DataFrame(rows, columns=["symbol", "promoter_holding_now", "promoter_holding_1y"])


def apply_promoter_edits(evid, edits):
    evid = {k: dict(v) for k, v in evid.items()}
    for r in edits.itertuples():
        if pd.notna(r.promoter_holding_now) and pd.notna(r.promoter_holding_1y):
            evid.setdefault(r.symbol, {"symbol": r.symbol}).update(promoter_holding_now=float(r.promoter_holding_now), promoter_holding_1y=float(r.promoter_holding_1y))
    return evid


# ---------------- cost check ----------------
def cost_candidates(HD, cfg, saved=None):
    """Lines whose average cost looks wrong (same rules as the cost-basis check), pre-filled from the pack."""
    sv = {}
    if saved is not None and len(saved):
        for r in saved.itertuples(): sv[str(r.symbol).upper()] = (str(r.decision), r.factor)
    rows = []
    for s, r in HD.iterrows():
        if not (r.Priced and r.AssetType == "equity"): continue
        why = []
        if r.AvgCost < cfg["ANOM_MIN_AVGCOST"]: why.append(f"average cost below Rs {cfg['ANOM_MIN_AVGCOST']:.0f}")
        if r.PnLPct > cfg["ANOM_MAX_GAIN_PCT"]: why.append(f"gain above {cfg['ANOM_MAX_GAIN_PCT']:.0f}%")
        if r.PnLPct < cfg["ANOM_MIN_PNL_PCT"]: why.append(f"loss beyond {abs(cfg['ANOM_MIN_PNL_PCT']):.0f}%")
        if why:
            d, f = sv.get(str(s).upper(), ("", np.nan))
            rows.append(dict(symbol=s, avg_cost=float(r.AvgCost), price=float(r.CMP), pnl_pct=float(r.PnLPct), flag="; ".join(why),
                             confirm=(d == "confirmed"), factor=(f if d == "factor" else np.nan)))
    return pd.DataFrame(rows, columns=COST_COLS)


def decisions_from_edits(edits):
    confirmed, factors, rows = set(), {}, []
    for r in edits.itertuples():
        if bool(r.confirm):
            confirmed.add(r.symbol); rows.append(dict(symbol=r.symbol, decision="confirmed", factor=np.nan))
        if pd.notna(r.factor) and float(r.factor) > 0:
            factors[r.symbol] = float(r.factor); rows.append(dict(symbol=r.symbol, decision="factor", factor=float(r.factor)))
    return confirmed, factors, pd.DataFrame(rows, columns=DEC_COLS)


def merge_decisions(old, new, shown):
    keep = old[~old.symbol.isin(set(shown))] if len(old) else old
    return pd.concat([keep, new], ignore_index=True)[DEC_COLS]


class FormHooks(NullHooks):
    """The page's answers to the cost review: symbols whose cost the user confirmed, and correction factors."""
    def __init__(self, confirmed=(), factors=None):
        self.confirmed, self.factors = set(confirmed), dict(factors or {})

    def cost_review(self, HD, cfg):
        HD = HD.copy()
        for s, f in self.factors.items():
            if s in HD.index and f and f > 0:
                HD.loc[s, "AvgCost"] = HD.loc[s, "AvgCost"] * f
                HD.loc[s, "Invested"] = HD.loc[s, "Qty"] * HD.loc[s, "AvgCost"]
                if bool(HD.loc[s, "Priced"]):
                    HD.loc[s, "PnL"] = HD.loc[s, "Value"] - HD.loc[s, "Invested"]
                    HD.loc[s, "PnLPct"] = HD.loc[s, "PnL"] / HD.loc[s, "Invested"] * 100
        return HD, set(self.confirmed), dict(self.factors)


def package_outputs(res, state, pack, decisions, shown_cost_symbols=(), company_info=None):
    """Everything the user downloads at the end of a run."""
    dec = merge_decisions(pack["cost_decisions"], decisions, shown_cost_symbols)
    gaps = res.sheets["Data_Gaps"]
    return dict(html=render_result_html(res), xlsx=build_xlsx_bytes(res.sheets), pack=build_pack(state, pack["buy_dates"], dec, company_info),
                n_sheets=int(sum(1 for v in res.sheets.values() if len(v) > 0)),
                status=next((x for x in res.summary if str(x).startswith("STATUS")), ""),
                notes=[str(x) for x in gaps.loc[gaps.Item == "portfolio", "Impact"]] if len(gaps) else [])
