import numpy as np

def _safe_float(val):
    if val is None or str(val).strip() in ("", "-", "nan", "None"): 
        return None
    try: 
        return float(val)
    except: 
        return None

def score_earnings(ev, fund, revenue_leg=False):
    # revenue_leg=False is the original behaviour. True (used for the F1 shadow test) adds Yahoo's revenue growth as a sales-growth leg,
    # but only when no quarterly sales growth was typed into evidence.csv.
    scores = []
    reasons = []

    # 1. SALES GROWTH (from evidence.csv)
    sales_vals = []
    for q in ["sales_growth_q1", "sales_growth_q2", "sales_growth_q3", "sales_growth_q4"]:
        v = _safe_float(ev.get(q))
        if v is not None: sales_vals.append(v)
    
    if sales_vals:
        avg_sales = sum(sales_vals) / len(sales_vals)
        if avg_sales > 15: scores.append(1)
        elif avg_sales > 5: scores.append(2)
        elif avg_sales >= -2: scores.append(3)
        elif avg_sales >= -10: scores.append(4)
        else: scores.append(5)
        reasons.append(f"Sales Growth {avg_sales:.1f}%")
    elif revenue_leg:
        rc = _safe_float(fund.get("rev_cagr"))
        if rc is not None and rc == rc:
            g = rc * 100
            if g > 15: scores.append(1)
            elif g > 5: scores.append(2)
            elif g >= -2: scores.append(3)
            elif g >= -10: scores.append(4)
            else: scores.append(5)
            reasons.append(f"Revenue CAGR {g:.1f}% (Yahoo annual statements)")

    # 2. PROFIT GROWTH (from Yahoo data)
    ni_cagr = _safe_float(fund.get("ni_cagr"))
    if ni_cagr is not None:
        if ni_cagr > 0.15: scores.append(1)
        elif ni_cagr > 0.05: scores.append(2)
        elif ni_cagr >= 0: scores.append(3)
        elif ni_cagr >= -0.10: scores.append(4)
        else: scores.append(5)
        reasons.append(f"Profit CAGR {ni_cagr*100:.1f}%")
        peak = _safe_float(fund.get("ni_peak_ratio"))
        if peak is not None and peak >= 1.5 and scores[-1] < 3:
            scores[-1] = 3
            reasons[-1] += " (capped: latest profit " + ("follows earlier losses" if peak >= 50 else f"is {peak:.1f}x the prior average") + ", so the growth may be a cyclical peak)"

    # 3. MARGIN TREND (from Yahoo data)
    margin_delta = _safe_float(fund.get("margin_delta"))
    if margin_delta is not None:
        if margin_delta > 0.02: scores.append(1)
        elif margin_delta > 0: scores.append(2)
        elif margin_delta > -0.02: scores.append(3)
        elif margin_delta > -0.05: scores.append(4)
        else: scores.append(5)
        reasons.append(f"Margin change {margin_delta*100:.1f}%")

    # 4. RETURNS ON CAPITAL (from Yahoo data)
    roce = _safe_float(fund.get("roce"))
    if roce is not None:
        if roce > 20: scores.append(1)
        elif roce > 15: scores.append(2)
        elif roce > 10: scores.append(3)
        elif roce > 5: scores.append(4)
        else: scores.append(5)
        reasons.append(f"ROCE {roce:.1f}%")

    # Combine scores
    if not scores:
        return None, "No earnings data available (missing sales, profit, margin, and ROCE)"
    
    final_score = sum(scores) / len(scores)
    reason_str = " | ".join(reasons)
    
    return final_score, reason_str
