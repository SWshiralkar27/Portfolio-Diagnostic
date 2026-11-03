import math


def _num(ev, key):
    """float value, or None when the field is missing, blank, NaN or non-numeric."""
    val = ev.get(key)
    if val is None:
        return None
    try:
        f = float(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return None
    return None if math.isnan(f) else f


def score_ownership(ev):
    """F2 Ownership & Governance (1 = best, 5 = worst).

    Evidence: change in promoter-group holding over ~1 year (NSE shareholding pattern).
    NSE's pattern list carries no FII/DII split, so the FII legs below only apply when you
    type fii_holding_now / fii_holding_1y into evidence.csv yourself.
    A fall while the promoter still holds a majority (control intact) is softened by one notch:
    it is usually a planned sell-down (e.g. a government offer-for-sale) rather than a warning.
    Returns (None, reason) when the evidence is missing - never a neutral 3.
    """
    p_now, p_1y = _num(ev, "promoter_holding_now"), _num(ev, "promoter_holding_1y")
    if p_now is None or p_1y is None:
        return None, "Promoter holding (now / 1 year ago) not available - unscored"
    if p_now < 1.0 and p_1y < 1.0:
        return None, "No promoter group (professionally managed) - promoter trend is not informative"

    chg = p_now - p_1y
    if chg <= -3.0:
        score, why = 5, f"Promoter holding fell {abs(chg):.1f}pp in a year ({p_1y:.1f}% -> {p_now:.1f}%): stake sale or heavy dilution"
    elif chg <= -1.0:
        score, why = 4, f"Promoter holding fell {abs(chg):.1f}pp in a year ({p_1y:.1f}% -> {p_now:.1f}%)"
    elif chg < 1.0:
        score, why = 3, f"Promoter holding stable ({chg:+.1f}pp; now {p_now:.1f}%)"
    elif chg < 3.0:
        score, why = 2, f"Promoter holding rose {chg:.1f}pp in a year ({p_1y:.1f}% -> {p_now:.1f}%)"
    else:
        score, why = 1, f"Promoter holding rose {chg:.1f}pp in a year ({p_1y:.1f}% -> {p_now:.1f}%)"
    if chg <= -1.0 and p_now >= 51.0:
        score, why = score - 1, why + " - majority control intact (softened one notch)"

    f_now, f_1y = _num(ev, "fii_holding_now"), _num(ev, "fii_holding_1y")
    if f_now is not None and f_1y is not None:
        f_chg = f_now - f_1y
        if f_chg < -2.0:
            score, why = max(score, 4), why + f"; FII holding fell {abs(f_chg):.1f}pp"
        elif f_chg > 1.0 and chg >= 0:
            score, why = min(score, 2), why + f"; FII holding rose {f_chg:.1f}pp"
    return float(score), why
