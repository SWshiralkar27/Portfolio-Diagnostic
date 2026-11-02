"""check_step26.py - tests the Step 26 hosting-test page (step26_spike_app.py) with Streamlit's own test runner.
Needs streamlit installed (pip install streamlit). Two scenarios:
  A  the libraries are replaced by fakes that answer in the real Yahoo / dalal / NSE shapes -> everything should read OK
  B  the fakes fail the way a blocked host does (HTTP 403, rate limit, timeouts) -> the page must not crash, must show FAIL rows
     and must say what it means
The REAL network answer can only come from running the page on the host itself (that is the point of Step 26)."""
from __future__ import annotations
import sys, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"\n        {detail}" if detail and not ok else ""))


def install_fakes(mode):
    import pandas as pd
    import requests, yfinance, dalal

    class T:
        def __init__(self, tkr):
            self.tkr = tkr
            self.calendar = {"Earnings Date": [pd.Timestamp.today().date()]}

        def _boom(self):
            raise RuntimeError("YFRateLimitError: Too Many Requests. Rate limited. Try after a while.")

        @property
        def info(self):
            if mode == "block": self._boom()
            return {"marketCap": 1e12, "trailingPE": 25.0, **{f"f{i}": i for i in range(150)}}

        def history(self, period=None, start=None, end=None, auto_adjust=False, actions=True):
            if mode == "block": self._boom()
            idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=400 if start else 22).tz_localize("Asia/Kolkata")
            return pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Adj Close": 1.5, "Volume": 1000.0, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)

        def _st(self, rows):
            if mode == "block": self._boom()
            return pd.DataFrame([[1.0, 2.0]] * len(rows), index=rows, columns=[pd.Timestamp("2026-03-31"), pd.Timestamp("2025-03-31")])

        @property
        def financials(self): return self._st(["Total Revenue", "Net Income"])
        @property
        def balance_sheet(self): return self._st(["Total Assets"])
        @property
        def cashflow(self): return self._st(["Operating Cash Flow"])

    yfinance.Ticker = T

    def sh(sym):
        if mode == "block": raise dalal.RateLimited("simulated NSE rate limit") if hasattr(dalal, "RateLimited") else RuntimeError("blocked")
        return [{"date": "30-SEP-2026", "pr_and_prgrp": "50.44"}, {"date": "30-JUN-2026", "pr_and_prgrp": "50.10"}, {"date": "31-MAR-2026", "pr_and_prgrp": "50.00"}, {"date": "31-DEC-2025", "pr_and_prgrp": "49.90"}, {"date": "30-SEP-2025", "pr_and_prgrp": "50.30"}]

    def an(sym):
        if mode == "block": raise RuntimeError("simulated NSE block")
        return [{"an_dt": "24-Oct-2026 21:48:04", "desc": "General Updates", "attchmntText": "x", "attchmntFile": ""}]
    dalal.shareholding, dalal.announcements = sh, an

    class Resp:
        def __init__(self, code, body=b'{"data": []}'):
            self.status_code, self.content = code, body

        def json(self):
            return {"data": []}

    class S:
        def __init__(self): self.headers = {}

        def get(self, url, params=None, timeout=None):
            if mode == "block": return Resp(403, b"Access Denied")
            return Resp(200)
    requests.Session = S


def run_app(mode):
    from streamlit.testing.v1 import AppTest
    install_fakes(mode)
    at = AppTest.from_file(str(HERE / "step26_spike_app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button[0].click().run()
    return at


def main():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("streamlit is not installed here (pip install streamlit); this check needs it.")
        return 1
    print("[A] fakes answer like a healthy host")
    at = run_app("ok")
    check("page runs without an exception", not at.exception, str([e.value for e in at.exception]))
    code = "\n".join(c.value for c in at.code)
    check("report is produced", "STEP 26 HOSTING TEST REPORT" in code)
    check("no FAIL row on a healthy host", "[FAIL]" not in code, "\n".join(l for l in code.splitlines() if l.startswith("[FAIL]")))
    check("all tests ran: Yahoo x3 (3 symbols), dalal x2, NSE direct, app code x5", code.count("[OK ]") == 3 * 3 + 3 * 2 + 1 + 5, str(code.count("[OK ]")))
    check("the page tells the owner what it means", any("work" in i.value or "came back" in i.value for i in at.info))
    print("[B] fakes fail like a blocked host")
    at = run_app("block")
    check("page survives a blocked host without an exception", not at.exception, str([e.value for e in at.exception]))
    code = "\n".join(c.value for c in at.code)
    check("FAIL rows show the error shape", "[FAIL]" in code and ("Rate" in code or "403" in code), code[:300])
    check("the page explains the consequence (try again / refresh evidence on the PC)", any(("failed" in i.value) or ("did not come back" in i.value) for i in at.info))
    at.text_input[0].set_value("BAD SYMBOL;").run()
    at.button[0].click().run()
    check("an invalid symbol is refused", len(at.error) == 1)
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    code = main()
    if code:
        sys.exit(code)
