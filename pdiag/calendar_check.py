"""Is the latest price one trading day old? (Yahoo sometimes serves yesterday's last bar.) Weekends count as non-trading days;
NSE holidays are not known here, so a note can be a false alarm after a holiday and says so."""
from __future__ import annotations
import pandas as pd

OPEN_MINUTES = 9 * 60 + 30      # 09:30 IST: by then today's bar (live or final) should exist


def expected_last_bar(now_ist):
    now_ist = pd.Timestamp(now_ist)
    d = now_ist.normalize().tz_localize(None) if now_ist.tzinfo is not None else now_ist.normalize()
    if d.weekday() < 5 and now_ist.hour * 60 + now_ist.minute >= OPEN_MINUTES:
        return d
    d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d = d - pd.Timedelta(days=1)
    return d


def stale_price_note(last_bar, now_ist):
    exp, lb = expected_last_bar(now_ist), pd.Timestamp(last_bar).normalize()
    if lb < exp:
        return (f"Prices are as of {lb:%Y-%m-%d}, but the latest NSE trading day should be {exp:%Y-%m-%d} (market holidays are not considered): "
                "Yahoo may be a day behind, so values, P&L and weights are a day old. Run again later if that matters.")
    return ""
