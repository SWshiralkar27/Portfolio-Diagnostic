"""Settings. CFG is one shared dictionary: treat it as read-only inside the app."""
from __future__ import annotations


VERSION = "4.0"

CFG = dict(
    RISK_FREE=0.06, HISTORY_YEARS=6, RISK_WINDOW_YEARS=3, CORR_WINDOW_YEARS=2,
    PARTICIPATION=(0.10, 0.20), EXIT_DAYS_FLAG=10, REBAL_BAND_PP=2.0,
    CONC_SINGLE_PCT=10.0, CONC_SECTOR_PCT=30.0, CONC_TOP5_PCT=50.0, LTCG_SOON_DAYS=90,
    TAX_STCG=0.20, TAX_LTCG=0.125, LTCG_EXEMPTION=125000.0, CESS=0.04,
    CAP_LARGE_CR=100000.0, CAP_MID_CR=30000.0,
    ANOM_MIN_AVGCOST=5.0, ANOM_MAX_GAIN_PCT=1000.0, ANOM_MIN_PNL_PCT=-90.0,
    OVERWEIGHT_MAX=2.2, UNDERWEIGHT_MIN=3.2, MIN_COVERAGE=0.50,
    BENCH_DIV_YIELD=dict(NIFTY500=0.012, LARGE=0.013, MID=0.008, SMALL=0.008, DEFAULT=0.012),
)

BENCH_CANDIDATES = dict(
    NIFTY500=["^CRSLDX", "^NSEI"], LARGE=["^CNX100", "^NSEI"],
    MID=["NIFTYMIDCAP150.NS", "^NSEMDCP50", "MID150BEES.NS"],
    SMALL=["NIFTYSMLCAP250.NS", "^CNXSC", "HDFCSML250.NS"],
)

CAP_TO_BENCH = dict(large="LARGE", mid="MID", small="SMALL")

FIN_TYPES = {"bank", "nbfc", "hfc", "insurer", "financial"}

NO_STD_TYPES = {"reit", "holdco"}

CONF_RANK = dict(Low=0, Medium=1, High=2)

HOLD_ATTRS = ["sector", "sector_type", "cap_tier", "target_weight_pct", "asset_type", "benchmark", "surveillance", "yahoo_ticker", "horizon", "xirr"]

EVIDENCE_MAX_AGE_DAYS = 7

EVIDENCE_SCHEMA = 5          # bump when the classification rules change: older rows are then re-fetched automatically

AUTO_FIELDS = ("promoter_holding_now", "promoter_holding_1y", "promoter_holding_asof", "next_earnings",
               "ex_date", "announcements", "ann_watch", "flag_log", "fetched_at", "fetch_notes", "schema")

MANUAL_FALLBACK = ("promoter_holding_now", "promoter_holding_1y")

SCORABLE = ("F1", "F2", "F5", "F6", "F7", "F8")
