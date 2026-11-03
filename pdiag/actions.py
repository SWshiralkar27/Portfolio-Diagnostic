"""Research signal / risk alert / allocation layers. Nothing here is an order."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pdiag.config import CFG


def build_actions(SCD, HD, liq, total_value, TAX=None, enforce_cap=True):
    """Three separate layers per holding. Nothing here is an order.

      1. Research signal - what the factor model says about the company. Never carries an amount.
      2. Risk alert      - hard-override flags and concentration. A concentration alert shows an ILLUSTRATIVE amount
                           (what selling down to the limit would mean, before tax and costs).
      3. Allocation      - an Add/Trim amount appears ONLY when you supplied a target weight for the holding
                           (target_weight_pct in holdings.csv) and it sits outside the rebalancing band.
                           Without a target the tool says so and shows no amount.
    The single-stock limit (CONC_SINGLE_PCT) is a model setting you can change, not a regulatory rule.
    """
    cap_pct = CFG["CONC_SINGLE_PCT"]
    cap_val = cap_pct / 100.0 * total_value
    band = CFG["REBAL_BAND_PP"]

    tax_state = {}
    if TAX is not None and len(TAX) and "Status" in TAX.columns:
        for sym, g in TAX.groupby("Symbol"):
            st = g["Status"].astype(str)
            if st.str.startswith("STCG").any():
                tax_state[sym] = "Includes STCG lots (held < 1 year) - check lot selection"
            elif st.str.startswith("unknown").any():
                tax_state[sym] = "Holding period unknown (no buy_date) - tax not estimated"
            else:
                tax_state[sym] = "All lots long-term (LTCG)"
    no_tax = "Holding period unknown (no buy_date) - tax not estimated"

    rows = []
    for _, r in SCD.iterrows():
        s = r["Symbol"]
        w = float(r["Weight"])
        val = w / 100.0 * total_value
        label, conf = r["FinalLabel"], str(r.get("Confidence", ""))
        cov = r.get("Coverage", np.nan)
        tgt = HD.at[s, "Target"] if (s in HD.index and "Target" in HD.columns) else np.nan
        tgt_val = tgt / 100.0 * total_value if pd.notna(tgt) else np.nan

        # 1. research signal
        if label == "Candidate for review":
            signal = "Review flag"
        elif label == "Not classified" or conf == "Low" or (pd.notna(cov) and cov < CFG["MIN_COVERAGE"]):
            signal = "Insufficient evidence"
        else:
            signal = {"Overweight": "Favourable", "Underweight": "Unfavourable"}.get(label, "Neutral")
        act, amt, why = "Hold", 0.0, [f"Research signal: {signal}"]
        if "cyclical peak" in str(r.get("Bear_Signals", "")):
            why.append("Caution: earnings are far above their prior average (possible cyclical peak) - treat 'Favourable' with care")

        # 2. hard-override flags
        if label == "Candidate for review":
            act = "Review / Exit"
            flags = [x.replace("HARD OVERRIDE: ", "") for x in str(r.get("Review_Risk", "")).split(" | ") if x.startswith("HARD OVERRIDE")]
            why.append("Verify before any sale: " + "; ".join(flags) if flags else "Hard override raised")
        # 3. allocation: only against a target you supplied
        elif pd.notna(tgt_val):
            gap = tgt_val - val
            if abs(gap) > band / 100.0 * total_value:
                if gap < 0:
                    act, amt = "Trim", -gap
                    why.append(f"Above your {tgt:.1f}% target by more than the {band:.0f}pp band: trim back to target")
                elif signal in ("Favourable", "Neutral"):
                    act, amt = "Add", gap
                    why.append(f"Below your {tgt:.1f}% target by more than the {band:.0f}pp band: add up to target")
                else:
                    why.append(f"Below your {tgt:.1f}% target, but the research signal is {signal.lower()}: no add")
            else:
                why.append(f"Within the {band:.0f}pp band of your {tgt:.1f}% target")
        else:
            why.append("No allocation action: no target weight supplied (target_weight_pct in holdings.csv)")

        # 4. concentration risk alert (independent of the research signal)
        if enforce_cap and val > cap_val + 1:
            excess = val - cap_val
            note = f"Above the {cap_pct:.0f}% single-stock limit ({w:.1f}%) - a model setting, not a regulatory rule"
            if pd.notna(tgt) and tgt > cap_pct:
                why.append(f"Above the {cap_pct:.0f}% model limit ({w:.1f}%), but your own {tgt:.1f}% target takes precedence")
            elif act == "Hold":
                act, amt = "Risk alert", excess
                why.append(note + f". Reaching the limit would mean selling about ₹{excess / 1e5:.1f}L (illustrative; before tax and costs)")
            else:
                if act == "Trim" and excess > amt:
                    amt = excess
                why.append(note)

        tax_note = tax_state.get(s, no_tax) if act in ("Trim", "Review / Exit", "Risk alert") else ""
        liq_note = ""
        if s in liq and liq[s]["DaysToExit_10pct"] > 2.0:
            liq_note = f"Illiquid ({liq[s]['DaysToExit_10pct']:.1f} days to exit)"
        rows.append(dict(Symbol=s, ResearchSignal=signal, Action=act, Amount_INR=amt, Reason="; ".join(why), Tax_Note=tax_note, Liquidity_Note=liq_note))
    return pd.DataFrame(rows).set_index("Symbol")
