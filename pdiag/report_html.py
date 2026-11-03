"""HTML report. render_html returns text (used by the app); write_html writes the file (command line)."""
from __future__ import annotations
import datetime as dt
import html as _html
import numpy as np
import pandas as pd
from pdiag.config import CFG, VERSION
from pdiag.engine_link import FACTORS, WEIGHTS, FACTOR_NAMES


BADGE_CLS = {"Overweight": "ok", "Neutral": "neu", "Underweight": "bad", "Not classified": "warn", "Candidate for review": "alert"}

_CSS = """
:root{--bg:#f4f6fb;--card:#fff;--card2:#f8f9fd;--ink:#192037;--muted:#66708d;--line:#e3e7f1;--accent:#3556d8;--accent-soft:rgba(53,86,216,.09);--pos:#168a5c;--neg:#cf3d3d;--warn:#c98216;--s1:#1f9d6b;--s2:#78b957;--s3:#e0b52f;--s4:#e8843a;--s5:#d64545;--shadow:0 1px 2px rgba(25,32,55,.05),0 8px 24px rgba(25,32,55,.06)}
@media (prefers-color-scheme:dark){:root{--bg:#0e1220;--card:#161b2e;--card2:#1b2138;--ink:#e9ecf7;--muted:#98a1c0;--line:#272f4a;--accent:#7f99ff;--accent-soft:rgba(127,153,255,.14);--pos:#3cc48f;--neg:#ff6b6b;--warn:#e9a23b;--s1:#2fae7c;--s2:#82c761;--s3:#d9b640;--s4:#e58a42;--s5:#e05252;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.28)}}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;font-variant-numeric:tabular-nums}
.wrap{max-width:1280px;margin:0 auto;padding:0 22px 60px}
header.hero{padding:34px 0 18px}.hero h1{margin:0;font-size:30px;letter-spacing:-.02em}
.hero .sub{color:var(--muted);margin-top:6px;font-size:13.5px}
.demo{background:var(--neg);color:#fff;padding:8px 14px;border-radius:10px;font-weight:600;margin-top:14px;display:inline-block}
nav.top{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 86%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
nav.top .in{max-width:1280px;margin:0 auto;padding:10px 22px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
nav.top a{color:var(--muted);text-decoration:none;padding:5px 11px;border-radius:999px;font-size:13px}
nav.top a:hover{background:var(--accent-soft);color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:8px 0 26px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;box-shadow:var(--shadow)}
.kpi .l{font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.kpi .v{font-size:25px;font-weight:700;margin:4px 0 2px;letter-spacing:-.01em}
.kpi .s{font-size:12.5px;color:var(--muted)}
.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--muted)}
section.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px 24px;margin:0 0 22px;box-shadow:var(--shadow)}
section.card h2{margin:0 0 4px;font-size:19px;letter-spacing:-.01em}
section.card .lead{color:var(--muted);margin:0 0 16px;font-size:13.5px}
h3{font-size:14px;margin:18px 0 8px;color:var(--ink)}
ul.clean{margin:0;padding-left:18px}ul.clean li{margin:5px 0}.summary li{margin:8px 0}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13.2px}
th{background:var(--card2);color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;text-align:left;padding:10px 12px;white-space:nowrap;border-bottom:1px solid var(--line)}
td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:middle}
tr:last-child td{border-bottom:none}tbody tr:hover td{background:var(--accent-soft)}
td.r,th.r{text-align:right}td.c,th.c{text-align:center}.sym{font-weight:700}
.chip{display:inline-block;min-width:30px;text-align:center;padding:2px 7px;border-radius:8px;font-weight:700;font-size:12px;color:#fff}
.chip.s1{background:var(--s1)}.chip.s2{background:var(--s2)}.chip.s3{background:var(--s3);color:#2b2200}.chip.s4{background:var(--s4)}.chip.s5{background:var(--s5)}
.chip.na{background:transparent;color:var(--muted);border:1px dashed var(--line);font-weight:500}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap;border:1px solid transparent}
.badge.ok{background:rgba(31,157,107,.14);color:var(--pos)}.badge.neu{background:var(--accent-soft);color:var(--accent)}.badge.bad{background:rgba(214,69,69,.14);color:var(--neg)}.badge.warn{background:rgba(224,170,47,.18);color:var(--warn)}.badge.alert{background:var(--neg);color:#fff}
.bar{height:8px;background:var(--line);border-radius:6px;overflow:hidden;min-width:70px}
.bar span{display:block;height:100%;background:var(--accent);border-radius:6px}
.bar.s1 span{background:var(--s1)}.bar.s2 span{background:var(--s2)}.bar.s3 span{background:var(--s3)}.bar.s4 span{background:var(--s4)}.bar.s5 span{background:var(--s5)}.bar.alt span{background:var(--warn)}
.ws{display:flex;align-items:center;gap:8px}.ws b{min-width:34px}
.dbar{position:relative;height:8px;min-width:110px;background:var(--line);border-radius:6px}.dbar:after{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--muted);opacity:.6}
.dbar span{position:absolute;top:0;bottom:0;border-radius:6px}.dbar span.pos{background:var(--pos)}.dbar span.neg{background:var(--neg)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:24px}
.row{display:grid;grid-template-columns:120px 1fr 60px;gap:10px;align-items:center;margin:7px 0;font-size:13px}
.row .lab{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.row .num{text-align:right;color:var(--muted)}
.pair{display:grid;grid-template-columns:120px 1fr 56px;gap:10px;align-items:center;margin:5px 0;font-size:13px}.pair .bars{display:grid;gap:3px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin:2px 0 10px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-2px}
svg.chart{width:100%;height:auto;display:block}svg .grid{stroke:var(--line);stroke-width:1}svg .axis{fill:var(--muted);font-size:11px}
svg .line{fill:none;stroke-width:2.2;stroke-linejoin:round}svg .line.a{stroke:var(--accent)}svg .line.b{stroke:var(--muted);stroke-dasharray:5 4}
details.hold{border:1px solid var(--line);border-radius:14px;margin:10px 0;background:var(--card2)}
details.hold>summary{list-style:none;cursor:pointer;padding:13px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
details.hold>summary::-webkit-details-marker{display:none}
details.hold>summary:before{content:"\\25B8";color:var(--muted);transition:transform .15s}
details.hold[open]>summary:before{transform:rotate(90deg)}
details.hold .body{padding:4px 18px 18px;border-top:1px solid var(--line)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:22px}
dl.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;margin:0;font-size:13px}dl.kv dt{color:var(--muted)}dl.kv dd{margin:0;text-align:right}
.tag{display:inline-block;font-size:12px;padding:2px 9px;border-radius:8px;margin:2px 4px 2px 0}
.tag.risk{background:rgba(214,69,69,.13);color:var(--neg)}.tag.data{background:rgba(224,170,47,.17);color:var(--warn)}.tag.bull{background:rgba(31,157,107,.13);color:var(--pos)}.tag.bear{background:rgba(214,69,69,.10);color:var(--neg)}
.note{background:var(--accent-soft);border-radius:12px;padding:12px 16px;font-size:13px;margin:12px 0}
.heat td,.heat th{padding:5px 6px;text-align:center;font-size:11.5px}.heat th{white-space:nowrap}
footer{color:var(--muted);font-size:12.5px;margin-top:30px;padding-top:16px;border-top:1px solid var(--line)}
@media print{nav.top{display:none}body{background:#fff}details.hold .body{display:block}section.card{box-shadow:none;break-inside:avoid-page}}
"""

def esc(x): return "" if x is None or (isinstance(x, float) and np.isnan(x)) else _html.escape(str(x))

def inr(x):
    if x is None or pd.isna(x): return "&ndash;"
    n = int(round(float(x))); sign, s = ("-" if n < 0 else ""), str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]; parts = [head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1]; s = ",".join(parts) + "," + tail
    return f"{sign}&#8377;{s}"

def inr_short(x):
    if x is None or pd.isna(x): return "&ndash;"
    a, sign = abs(float(x)), ("-" if x < 0 else "")
    if a >= 1e7: return f"{sign}&#8377;{a / 1e7:.2f} Cr"
    if a >= 1e5: return f"{sign}&#8377;{a / 1e5:.2f} L"
    return f"{sign}&#8377;{a:,.0f}"

def p1(x, d=1, sign=False): return "&ndash;" if x is None or pd.isna(x) else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")

def num(x, d=2, suffix=""): return "&ndash;" if x is None or pd.isna(x) else f"{x:.{d}f}{suffix}"

def tone(x): return "" if x is None or pd.isna(x) or abs(x) < 1e-9 else ("pos" if x > 0 else "neg")

def chip(v):
    if v is None or pd.isna(v): return '<span class="chip na" title="not scored">&ndash;</span>'
    r = int(min(5, max(1, round(float(v))))); return f'<span class="chip s{r}" title="score {float(v):.2f}">{float(v):.1f}</span>'

def badge(t): return f'<span class="badge {BADGE_CLS.get(t, "neu")}">{esc(t)}</span>'

def td(content, cls=""): return f'<td class="{cls}">{content}</td>' if cls else f"<td>{content}</td>"

def th(text, cls=""): return f'<th class="{cls}">{text}</th>' if cls else f"<th>{text}</th>"

def table(headers, rows, extra=""): return f'<div class="scroll"><table {extra}><thead><tr>{"".join(th(h, c) for h, c in headers)}</tr></thead><tbody>{"".join("<tr>" + "".join(r) + "</tr>" for r in rows)}</tbody></table></div>'

def bar(value, vmax, cls=""): return f'<div class="bar {cls}"><span style="width:{0 if (pd.isna(value) or vmax <= 0) else min(100, max(0, value / vmax * 100)):.1f}%"></span></div>'

def divbar(v, scale=0.30):
    if v is None or pd.isna(v): return '<span class="muted">&ndash;</span>'
    w = min(50.0, abs(v) / scale * 50)
    return f'<div class="dbar"><span class="pos" style="left:50%;width:{w:.1f}%"></span></div>' if v >= 0 else f'<div class="dbar"><span class="neg" style="right:50%;width:{w:.1f}%"></span></div>'

def svg_growth(series, w=960, h=320):
    ml, mr, mt, mb = 50, 14, 12, 28
    frame = pd.concat([s.rename(i) for i, (_, s, _) in enumerate(series)], axis=1).dropna(how="all")
    if frame.empty: return ""
    d0, d1 = frame.index.min(), frame.index.max(); lo, hi = float(frame.min().min()), float(frame.max().max())
    pad = (hi - lo) * 0.06 or 1.0; lo, hi = lo - pad, hi + pad; span = max(1, (d1 - d0).days)
    X = lambda d: ml + (d - d0).days / span * (w - ml - mr); Y = lambda v: mt + (1 - (v - lo) / (hi - lo)) * (h - mt - mb)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Growth of 100">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        o.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text class="axis" x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    for yr in range(d0.year + 1, d1.year + 1):
        d = pd.Timestamp(year=yr, month=1, day=1)
        if d0 < d < d1: o.append(f'<line class="grid" x1="{X(d):.1f}" x2="{X(d):.1f}" y1="{mt}" y2="{h - mb}"/><text class="axis" x="{X(d):.1f}" y="{h - 8}" text-anchor="middle">{yr}</text>')
    for _, s, cls in series:
        s = s.dropna(); s2 = s.iloc[::max(1, len(s) // 320)]
        if s.index[-1] != s2.index[-1]: s2 = pd.concat([s2, s.iloc[[-1]]])
        o.append(f'<polyline class="line {cls}" points="' + " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in s2.items()) + '"/>')
    o.append("</svg>")
    return "".join(o)

def render_html(sheets, meta, pr, bm_r, port, port_x, summary, conc_pairs, fps, RC, liq, tax_est, stamp=None):
    """The report as HTML text. Nothing is written to disk."""
    stamp = stamp or dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    G = lambda k: sheets.get(k, pd.DataFrame())
    SCD = G("Scorecard_A"); HD = G("Holdings").set_index("Symbol") if len(G("Holdings")) else pd.DataFrame()
    perf = G("Performance_Risk").set_index("Symbol").to_dict("index") if len(G("Performance_Risk")) else {}
    TV = HD.Value.sum() if len(HD) else 0; TC = HD.loc[HD.Priced, "Invested"].sum() if len(HD) else 0; pnl = TV - TC
    n_hard = int(SCD.Review_Risk.str.contains("HARD").sum()) if len(SCD) else 0
    n_risk = int((SCD.Review_Risk != "").sum()) if len(SCD) else 0
    n_data = int((SCD.Review_DataGap != "").sum()) if len(SCD) else 0
    asof = (HD.PriceAsOf.max().date() if (len(HD) and HD.PriceAsOf.notna().any()) else "n/a")
    px = port_x or {}
    kp = [("Portfolio value", inr_short(TV), f"cost {inr_short(TC)}", ""), ("Unrealised P&amp;L", p1(pnl / TC, 1, True) if TC else "&ndash;", inr_short(pnl), tone(pnl)), ("Your XIRR", p1(px.get("XIRR")), ("money-weighted, lot level" if not pd.isna(px.get("XIRR", np.nan)) else "not calculable: no dated transactions"), tone(px.get("XIRR"))), ("XIRR vs Nifty 500", p1(px.get("Excess"), 1, True), f"index on same cash flows: {p1(px.get('Bench'))}", tone(px.get("Excess"))), ("Beta / volatility", f"{num(port.get('Beta vs Nifty 500'))} / {p1(port.get('Annualised volatility'), 0)}", "current weights, back-cast", ""), ("Max drawdown", p1(port.get("Max drawdown (window)")), f"{CFG['RISK_WINDOW_YEARS']}y window, back-cast", "neg" if not pd.isna(port.get("Max drawdown (window)", np.nan)) else ""), ("Effective positions", num(port.get("Effective number of positions (1/HHI)"), 1), f"top-5 = {num(port.get('Top-5 weight % (of risk assets)'), 0)}% of risk assets", ""), ("Review flags", f"{n_hard} hard &middot; {n_risk} risk", f"{n_data} with data gaps (separate)", "neg" if n_hard else "")]
    kpis = "".join(f'<div class="kpi"><div class="l">{l}</div><div class="v {c}">{v}</div><div class="s">{s}</div></div>' for l, v, s, c in kp)
    summ = "<ul class='clean summary'>" + "".join(f"<li>{esc(s)}</li>" for s in summary) + "</ul>"
    if conc_pairs: summ += f"<h3>Most correlated pairs</h3><p class='muted'>{esc('; '.join(conc_pairs))}</p>"
    sc_rows = []
    for i, r in enumerate(SCD.itertuples(), 1):
        sens = ' <span title="classification changes" class="muted">&#9888;</span>' if r.ClassificationSensitive else ""
        w_sc = f'<div class="ws"><b>{r.Weighted:.2f}</b>{bar((r.Weighted - 1) / 4 * 100, 100, "s" + str(int(min(5, max(1, round(r.Weighted))))))}</div>' if not pd.isna(r.Weighted) else "&ndash;"
        change_str = "&ndash;"
        if not pd.isna(r.Score_Delta):
            if r.Score_Delta > 0.05:
                change_str = f"<span class='neg'>↑ {r.Score_Delta:.2f}</span>"
            elif r.Score_Delta < -0.05:
                change_str = f"<span class='pos'>↓ {abs(r.Score_Delta):.2f}</span>"
            else:
                change_str = "<span class='muted'>0.00</span>"
                
        sc_rows.append([td(str(i), "c"), td(f'<a class="sym" href="#h-{esc(r.Symbol)}" style="color:inherit;text-decoration:none">{esc(r.Symbol)}</a>'), td(f"{r.Weight:.1f}%", "r"), td(esc(r.Sector) or "&ndash;")] + [td(chip(getattr(r, f)), "c") for f in FACTORS] + [td(w_sc), td(f"{r.Coverage * 100:.0f}%", "r"), td(badge(r.Classification) + sens), td(badge(r.FinalLabel)), td(esc(r.Confidence)), td(change_str, "c")])        

    sc_tbl = table([("#", "c"), ("Stock", ""), ("Weight", "r"), ("Sector", "")]
                   + [(f, "c") for f in FACTORS]
                   + [("Weighted", ""), ("Coverage", "r"), ("Classification", ""),
                      ("Final label", ""), ("Confidence", ""), ("Δ Score", "c")], sc_rows) \
             if sc_rows else "<p>No scorable holdings.</p>"
    fnames = " &middot; ".join(f"<b>{f}</b> {esc(FACTOR_NAMES[f])} ({int(WEIGHTS[f] * 100)}%)" for f in FACTORS)
    legend = '<div class="legend"><span><i style="background:var(--s1)"></i>1 favourable</span><span><i style="background:var(--s3)"></i>3 mixed</span><span><i style="background:var(--s5)"></i>5 unfavourable</span><span>&ndash; not scored</span></div>'
    EV, TR = G("Evidence"), G("Monitoring_Triggers"); cards = []
    for r in SCD.itertuples():
        s, p = r.Symbol, perf.get(r.Symbol, {}); risk = [x.strip() for x in str(r.Review_Risk).split(" | ") if x.strip()]; data = [x.strip() for x in str(r.Review_DataGap).split(" | ") if x.strip()]
        tags = "".join(f'<span class="tag risk">{esc(x)}</span>' for x in risk) + "".join(f'<span class="tag data">{esc(x)}</span>' for x in data)
        ev_rows = []
        if len(EV):
            for e in EV[EV.Symbol == s].itertuples(): ev_rows.append([td(esc(e.Factor)), td(chip(e.Score), "c"), td(esc(e.Basis)), td(esc(e.Evidence))])
        ev_tbl = table([("Factor", ""), ("Score", "c"), ("Basis", ""), ("Evidence", "")], ev_rows) if ev_rows else "<p class='muted'>No factor scored.</p>"
        trg = ""
        if len(TR):
            tt = TR[TR.Symbol == s]
            if len(tt): trg = "<ul class='clean'>" + "".join(f"<li>{esc(x)}</li>" for x in str(tt.iloc[0].Triggers).split(" || ")) + "</ul>"
        L = liq.get(s)
        act_text = "HOLD"
        if r.Action == "Add": act_text = f"<b class='pos'>ADD ₹{r.Amount_INR/1e5:.1f}L</b> <span class='muted'>to your target</span>"
        elif r.Action == "Trim": act_text = f"<b class='neg'>TRIM ₹{r.Amount_INR/1e5:.1f}L</b> <span class='muted'>to your target</span>"
        elif r.Action == "Review / Exit": act_text = "<b class='neg'>REVIEW</b> <span class='muted'>verify the flag before any sale</span>"
        elif r.Action == "Risk alert": act_text = f"<b class='neg'>RISK ALERT</b> <span class='muted'>illustrative: ₹{r.Amount_INR/1e5:.1f}L to reach the limit</span>"
        
        tax_line = f"{r.Tax_Note}" if r.Tax_Note else "&ndash;"
        liq_line = f"{r.Liquidity_Note}" if r.Liquidity_Note else "&ndash;"
        kv = [
            ("Horizon Profile", str(r.Horizon).title()),
            ("Weight / target", f"{r.Weight:.1f}% / {num(HD.loc[s, 'Target'], 1, '%') if (s in HD.index and not pd.isna(HD.loc[s, 'Target'])) else '&ndash;'}"),
            ("Action", act_text),
            ("Research signal", r.ResearchSignal),
            ("Action Rationale", r.Reason),
            ("Tax Note", tax_line),
            ("Liquidity Note", liq_line),
            ("Total return 1y / 3y / 5y", f"{p1(p.get('tr_1y'))} / {p1(p.get('tr_3y'))} / {p1(p.get('tr_5y'))}"),
            (f"Primary benchmark ({esc(p.get('primary_bench', ''))}) 3y", p1(p.get("bench_3y"))),
            ("Excess vs primary 3y / 5y (p.a.)", f"{p1(p.get('excess_3y'), 1, True)} / {p1(p.get('excess_5y'), 1, True)}"),
            ("Excess vs Nifty 500 3y (p.a.)", p1(p.get("excess_broad_3y"), 1, True)),
            ("Beta / alpha p.a. (t-stat)", f"{num(p.get('beta'))} / {p1(p.get('alpha_ann'), 1, True)} ({num(p.get('alpha_t'), 1)})"),
            ("Your Broker XIRR", f"<b>{num(HD.loc[s, 'UserXIRR'], 1, '%')}</b>" if 'UserXIRR' in HD.columns and s in HD.index and not pd.isna(HD.loc[s, 'UserXIRR']) else "&ndash;"),
            ("Script XIRR vs index", f"{p1(p.get('XIRR'))} vs {p1(p.get('XIRR_bench_same_flows'))}"),
            ("12-1m momentum / 6m rel. to index", f"{p1(p.get('mom_12_1'), 0, True)} / {p1(p.get('rel_6m'), 0, True)}"),
            ("vs 200-DMA / from 52w high", f"{p1(p.get('dma200_gap'), 1, True)} / {p1(p.get('dist_52w_high'), 1, True)}"),
            ("Max drawdown (3y)", p1(p.get("maxdd_3y"))),
            ("Days to exit @10% / @20% ADTV", f"{num(L['DaysToExit_10pct'], 1)} / {num(L['DaysToExit_20pct'], 1)}" if L else "&ndash;")
        ]


        kvh = "<dl class='kv'>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in kv) + "</dl>"
        sig = "".join(f'<span class="tag bull">{esc(x)}</span>' for x in str(r.Bull_Signals).split("; ") if x.strip()) + "".join(f'<span class="tag bear">{esc(x)}</span>' for x in str(r.Bear_Signals).split("; ") if x.strip())
        change_tags = ""
        if not pd.isna(r.Score_Delta):
            if r.Score_Delta > 0.05:
                change_tags += f"<span class='badge bad'>Score ↑ {r.Score_Delta:.2f}</span> "
            elif r.Score_Delta < -0.05:
                change_tags += f"<span class='badge ok'>Score ↓ {abs(r.Score_Delta):.2f}</span> "
        if r.Label_Changed:
            change_tags += f"<span class='badge warn'>Label: {esc(r.Prev_Label)} &rarr; {esc(r.FinalLabel)}</span> "
        is_open = " open" if any(x.startswith("HARD") for x in risk) else ""
        change_tags = ""
        if not pd.isna(r.Score_Delta):
            if r.Score_Delta > 0.05:
                change_tags += f"<span class='badge bad'>Score ↑ {r.Score_Delta:.2f}</span> "
            elif r.Score_Delta < -0.05:
                change_tags += f"<span class='badge ok'>Score ↓ {abs(r.Score_Delta):.2f}</span> "
        if r.Label_Changed:
            change_tags += f"<span class='badge warn'>Label: {esc(r.Prev_Label)} &rarr; {esc(r.FinalLabel)}</span> "

        is_open = " open" if any(x.startswith("HARD") for x in risk) else ""
        cards.append(
            f'<details class="hold" id="h-{esc(s)}"{is_open}><summary>'
            f'<span class="sym" style="font-size:16px">{esc(s)}</span>'
            f'{badge(r.Classification)}'
            f'{badge(r.FinalLabel) if r.FinalLabel != r.Classification else ""}'
            f'{change_tags}'
            f'<span class="muted">weight {r.Weight:.1f}% &middot; score {num(r.Weighted)} '
            f'&middot; coverage {r.Coverage * 100:.0f}% &middot; confidence {esc(r.Confidence)}</span>'
            f'</summary><div class="body">'
            f'<div style="margin-top:12px">{tags or "<span class=muted>No review reasons raised.</span>"}</div>'
            f'<div class="cols"><div><h3>Key metrics</h3>{kvh}'
            f'<h3>Computed signals</h3>{sig or "<span class=muted>none</span>"}</div>'
            f'<div><h3>Monitoring triggers</h3>{trg}</div></div>'
            f'<h3>Evidence behind each score</h3>{ev_tbl}'
            f'<p class="muted" style="margin:8px 0 0">Confidence: {esc(r.ConfidenceReason)}</p>'
            f'</div></details>')
    CO = G("Concentration")
    def weight_bars(dim, cls=""):
        d = CO[CO.Dimension == dim] if len(CO) else pd.DataFrame()
        if not len(d): return "<p class='muted'>n/a</p>"
        mx = max(d.Weight.max(), 1)
        return "".join(f'<div class="row"><div class="lab" title="{esc(b.Bucket)}">{esc(b.Bucket)}</div>{bar(b.Weight, mx, cls)}<div class="num">{b.Weight:.1f}%</div></div>' for b in d.itertuples())
    # Check concentration limits
    top1_wt = port.get('Top-1 weight % (of risk assets)', 0)
    top5_wt = port.get('Top-5 weight % (of risk assets)', 0)
    top1_flag = " <span class='badge bad'>ABOVE MODEL LIMIT</span>" if top1_wt > CFG["CONC_SINGLE_PCT"] else ""
    top5_flag = " <span class='badge bad'>ABOVE MODEL LIMIT</span>" if top5_wt > CFG["CONC_TOP5_PCT"] else ""

    conc_kv = [
        ("Top-1 weight", f"{num(top1_wt, 1)}%{top1_flag}"),
        ("Top-5 weight", f"{num(top5_wt, 1)}%{top5_flag}"),
        ("Top-10 weight", f"{num(port.get('Top-10 weight % (of risk assets)'), 1)}%"),
        ("HHI (Capital)", num(port.get("HHI (risk-asset weights)"), 3)),
        ("Effective positions (Capital)", num(port.get("Effective number of positions (1/HHI)"), 1)),
        ("Effective risk contributors", num(port.get("Effective number of risk contributors"), 1)),
        ("PC1 share of variance", p1(port.get("PC1 share of return variance (single-factor dominance)"), 0)),
        ("Avg pairwise correlation", num(port.get("Average pairwise correlation"))),
        ("Cash / bonds", f"{num(port.get('Cash/bond weight % of total'), 1)}%"),
    ]
    h_rows = []
    for r in HD.reset_index().itertuples():
        dev = "" if pd.isna(r.Target) else f'<span class="{"neg" if r.OutOfBand else "muted"}">{r.TargetDev_pp:+.1f}pp</span>'
        h_rows.append([td(f'<span class="sym">{esc(r.Symbol)}</span>'), td(esc(r.AssetType)), td(esc(r.Sector) or "&ndash;"), td(f"{r.Qty:,.0f}", "r"), td(num(r.AvgCost), "r"), td(num(r.CMP), "r"), td(inr(r.Value) if r.Priced else "<span class='muted'>no price</span>", "r"), td((f'<span class="{tone(r.PnLPct)}">{r.PnLPct:+.1f}%</span>' + (f" <span class='badge bad' title='{esc(r.CostFlag)}'>unreconciled</span>" if r.CostFlag else "")) if not pd.isna(r.PnLPct) else "<span class='muted'>n/a</span>", "r"), td(f"<b>{num(r.UserXIRR, 1, '%')}</b>" if "UserXIRR" in r and not pd.isna(r.UserXIRR) else "&ndash;", "r"), td(f'<div class="ws"><b style="min-width:42px">{r.Weight:.1f}%</b>{bar(r.Weight, HD.Weight.max())}</div>'), td(dev, "r"), td(esc(r.Surveillance) or "&ndash;")])
    h_tbl = table([("Holding", ""), ("Type", ""), ("Sector", ""), ("Qty", "r"), ("Avg cost", "r"), ("CMP", "r"), ("Value", "r"), ("P&amp;L", "r"), ("Your XIRR", "r"), ("Weight", ""), ("vs target", "r"), ("Surveillance", "")], h_rows)
    growth = ""
    if len(pr) > 252:
        pw, bw = 100 * (1 + pr).cumprod(), 100 * (1 + bm_r.fillna(0)).cumprod()
        growth = '<div class="legend"><span><i style="background:var(--accent)"></i>Portfolio at today\'s weights (back-cast)</span><span><i style="background:var(--muted)"></i>Nifty 500 total return</span></div>' + svg_growth([("Portfolio", pw, "a"), ("Nifty 500", bw, "b")])
    pf_rows = []
    for s in HD.index:
        p = perf.get(s)
        if not p: continue
        pf_rows.append([td(f'<span class="sym">{esc(s)}</span>'), td(num(p.get("hist_years"), 1), "r"), td(p1(p.get("tr_1y")), "r"), td(p1(p.get("tr_3y")), "r"), td(p1(p.get("tr_5y")), "r"), td(f'<span class="{tone(p.get("excess_3y"))}">{p1(p.get("excess_3y"), 1, True)}</span>', "r"), td(divbar(p.get("excess_3y"))), td(f'<span class="{tone(p.get("excess_5y"))}">{p1(p.get("excess_5y"), 1, True)}</span>', "r"), td(f'{p1(p.get("alpha_ann"), 1, True)} <span class="muted">(t {num(p.get("alpha_t"), 1)})</span>', "r"), td(num(p.get("beta")), "r")])
    pf_tbl = table([("Holding", ""), ("History (y)", "r"), ("TR 1y", "r"), ("TR 3y", "r"), ("TR 5y", "r"), ("Excess 3y", "r"), ("", ""), ("Excess 5y", "r"), ("Alpha p.a.", "r"), ("Beta", "r")], pf_rows)
    lq_rows = []
    LQ = G("Liquidity")
    for r in (LQ.itertuples() if len(LQ) else []):
        flag = '<span class="badge bad">&gt; %d days</span>' % CFG["EXIT_DAYS_FLAG"] if r.Exit_flag else '<span class="badge ok">ok</span>'
        lq_rows.append([td(f'<span class="sym">{esc(r.Symbol)}</span>'), td(inr_short(r.ADTV_used), "r"), td(num(r.DaysToExit_10pct, 1), "r"), td(num(r.DaysToExit_20pct, 1), "r"), td(flag), td(esc(r.Surveillance) or "&ndash;")])
    lq_tbl = table([("Holding", ""), ("Traded value/day", "r"), ("Days to exit @10%", "r"), ("@20%", "r"), ("Flag", ""), ("Surveillance", "")], lq_rows) if lq_rows else "<p class='muted'>n/a</p>"
    GP, MT = G("Data_Gaps"), G("Method")
    RG = G("Evidence_Register")
    register_html = (table([("Holding", ""), ("Flag", ""), ("Observed", ""), ("Threshold", ""), ("Source", ""), ("Date", ""), ("Status", ""), ("Effect", "")],
                           [[td(f'<span class="sym">{esc(r.Symbol)}</span>'), td(esc(r.FlagType)), td(esc(r.Observed)), td(esc(r.Threshold)),
                             td(f'<a href="{esc(r.SourceLink)}" target="_blank" rel="noopener">{esc(r.Source)}</a>' if r.SourceLink else esc(r.Source)),
                             td(esc(r.SourceDate)), td(esc(r.Status)), td(esc(r.Effect))] for r in RG.itertuples()])
                     if len(RG) else "<p>No flags raised.</p>")
    gaps_html = "<ul class='clean'>" + "".join(f"<li><b>{esc(x[0])}</b> &mdash; {esc(x[1])}: {esc(x[2])}</li>" for x in GP.itertuples(index=False)) + "</ul>" if len(GP) else "<p>None recorded.</p>"
    meth_html = table([("Parameter", ""), ("Value", "")], [[td(esc(x[0])), td(esc(x[1]))] for x in MT.itertuples(index=False)]) if len(MT) else ""
    # --- RISK HTML GENERATION ---
    risk_html = "<p class='muted'>Not enough overlapping history.</p>"
    if len(RC):
        d = RC.sort_values("RiskContribution", ascending=False).head(15)
        mx = max(d.RiskContribution.max(), d.Weight.max(), 1)
        risk_html = (
            "<p class='muted' style='margin-bottom:10px;'>This compares how much <b>capital</b> you have in a stock (blue) vs how much <b>portfolio volatility</b> it generates (orange).</p>"
            '<div class="legend"><span><i style="background:var(--accent)"></i>Capital Weight</span>'
            '<span><i style="background:var(--warn)"></i>Risk Contribution</span></div>'
            + "".join(
                f'<div class="pair"><div class="lab" title="{esc(i)}">{esc(i)}</div>'
                f'<div class="bars">{bar(x.Weight, mx)}{bar(x.RiskContribution, mx, "alt")}</div>'
                f'<div class="num muted" style="text-align:right">{x.RiskContribution:.0f}%</div></div>'
                for i, x in d.iterrows()))

    CR = G("Correlation")
    heat = ""
    if len(CR) and "Symbol" in CR:
        names = [s for s in HD.sort_values("Value", ascending=False).index if s in CR.columns][:14]
        if len(names) >= 3:
            m = CR.set_index("Symbol")
            hr = []
            for a in names:
                cells = [f"<th>{esc(a)}</th>"]
                for b in names:
                    v = m.loc[a, b]
                    hue, al = (210, abs(v) * .75) if v >= 0 else (0, abs(v) * .75)
                    cells.append(f'<td style="background:hsla({hue},75%,52%,{al:.2f})" title="{esc(a)} / {esc(b)}: {v:.2f}">{v:.2f}</td>')
                hr.append("".join(cells))
            heat = ('<h3>Correlation (daily TR, last 2y)</h3><div class="scroll">'
                    '<table class="heat"><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table></div>'
                    % ("".join(f"<th>{esc(n)}</th>" for n in names), "".join(f"<tr>{r_}</tr>" for r_ in hr)))
            
    if conc_pairs:
        heat += f"<h3>Highest Correlated Pairs</h3><p class='muted' style='font-size:13px;'>{esc('; '.join(conc_pairs))}</p>"
    demo = '<div class="demo">SYNTHETIC DEMO DATA</div>' if meta.get("demo") else ""
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in [("summary", "Summary"), ("score", "Scorecard"), ("holdings", "Holdings"), ("allocation", "Allocation"), ("perf", "Performance"), ("liq", "Liquidity"), ("gaps", "Data gaps"), ("register", "Evidence register"), ("method", "Method")])
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Portfolio V4 &middot; {stamp}</title><style>{_CSS}</style></head><body>
<nav class="top"><div class="in">{nav}</div></nav><div class="wrap"><header class="hero"><h1>Portfolio Diagnostic V4</h1><div class="sub">Prices as of {asof} &middot; generated {stamp} &middot; data: {esc(meta['source'])} &middot; benchmark: {esc(meta['bench_src'])}</div>{demo}</header>
<div class="kpis">{kpis}</div><section class="card" id="summary"><h2>Executive summary</h2><p class="lead">Research and decision support only. Scores are relative ranking tools.</p>{summ}</section>
<section class="card" id="score"><h2>Scorecard</h2><p class="lead">Worst-scoring first. {fnames}</p>{legend}{sc_tbl}<h3>Holding detail</h3>{''.join(cards)}</section>
<section class="card" id="holdings"><h2>Holdings</h2><p class="lead">Weights are current market value.</p>{h_tbl}</section>
<section class="card" id="allocation"><h2>Allocation &amp; concentration</h2><div class="grid2"><div><h3>Sector (equities)</h3>{weight_bars("Sector")}</div><div><h3>Cap tier / asset type</h3>{weight_bars("Cap tier / asset type", "alt")}</div></div><h3>Concentration measures</h3><dl class="kv" style="max-width:520px">{''.join(f'<dt>{k}</dt><dd>{v}</dd>' for k, v in conc_kv)}</dl></section>
<section class="card" id="perf"><h2>Performance</h2><p class="lead">Total return vs benchmark.</p>{growth}<h3>By holding</h3>{pf_tbl}</section>
<section class="card" id="liq"><h2>Liquidity</h2><p class="lead">Days to exit assume 10&ndash;20% of ADTV.</p>{lq_tbl}</section>
<section class="card" id="gaps"><h2>Data gaps &amp; assumptions</h2><p class="lead">Absence of data is never treated as evidence of quality.</p>{gaps_html}</section>
<section class="card" id="register"><h2>Evidence register</h2><p class="lead">Every flag the report raises, with its source and status. Automated flags are not independently verified.</p>{register_html}</section>
<section class="card" id="method"><h2>Method &amp; configuration</h2><details><summary style="cursor:pointer">Show all parameters</summary><div style="margin-top:12px">{meth_html}</div></details></section>
<footer>Not investment advice. Verify anything material against exchange filings. Generated locally by portfolio_diagnostic.py v{VERSION}.</footer></div></body></html>"""
    return doc


def write_html(out, sheets, meta, pr, bm_r, port, port_x, summary, conc_pairs, fps, RC, liq, tax_est):
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M"); path = out / f"portfolio_report_{stamp}.html"
    path.write_text(render_html(sheets, meta, pr, bm_r, port, port_x, summary, conc_pairs, fps, RC, liq, tax_est, stamp=stamp), encoding="utf-8")
    return path


def render_result_html(res, stamp=None):
    """Report text for a RunResult (what the app shows and offers as a download)."""
    return render_html(res.sheets, res.meta, res.pr, res.bm_r, res.port, res.port_x, res.summary, [], res.sheets["Scorecard_A"].head(0), pd.DataFrame(), res.liq, None, stamp=stamp)
