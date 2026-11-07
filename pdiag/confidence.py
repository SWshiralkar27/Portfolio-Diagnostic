"""Confidence level for one holding, with the ceiling approved on 7 Nov 2026 (rule B1).

  High    only if at least 90% of the FULL specification weight is scored (all ten factors), scorable coverage is at least 85%,
          the valuation is not a proxy, and there is no hard override (an override cannot be High until verified)
  Medium  scorable coverage (the factors this engine can score today) of at least 70%
  Low     scorable coverage under 70%
Today only F1, F2, F5, F6, F7 and F8 can be scored (65% of the weight), so High cannot be reached; it returns on its own once more
factors are scored. A thin peer set does not lower the level; it stays visible in the note ('only n peers').
A Low level turns the holding's research signal into 'Insufficient evidence' (see actions.build_actions)."""
from __future__ import annotations

HIGH_TRUE_COVERAGE = 0.90
HIGH_SCORABLE_COVERAGE = 0.85
MEDIUM_SCORABLE_COVERAGE = 0.70


def true_coverage(S, weights):
    """Scored weight as a share of the full specification weight (all ten factors)."""
    return sum(weights[k] for k in weights if S.get(k) is not None) / sum(weights.values())


def confidence_ceiling_rule(S, weights, cov, proxies, hard, reason):
    tc = true_coverage(S, weights)
    if tc >= HIGH_TRUE_COVERAGE and cov >= HIGH_SCORABLE_COVERAGE and not proxies and not hard: level = "High"
    elif cov >= MEDIUM_SCORABLE_COVERAGE: level = "Medium"
    else: level = "Low"
    if tc < HIGH_TRUE_COVERAGE:
        reason = reason + f"; confidence ceiling Medium: scored weight {tc * 100:.0f}% of the full specification (High needs {HIGH_TRUE_COVERAGE * 100:.0f}%)"
    return level, reason
