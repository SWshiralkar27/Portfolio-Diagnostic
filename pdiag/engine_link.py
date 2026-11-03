"""The single place the five factor-engine files are imported. They sit next to the entry point (scoring.py,
earnings_engine.py, ownership_engine.py, catalyst_engine.py, leadership_engine.py). Moving them into the package
later only changes this file."""
from __future__ import annotations
from pdiag.errors import FatalInputError

try:
    import scoring
    import earnings_engine
    import ownership_engine
    import catalyst_engine
    import leadership_engine
except ImportError as e:
    raise FatalInputError(f"Missing engine file: {e}. Ensure scoring.py, earnings_engine.py, etc. are in the folder.")

FACTORS = scoring.FACTORS
WEIGHTS = scoring.WEIGHTS
FACTOR_NAMES = scoring.FACTOR_NAMES
