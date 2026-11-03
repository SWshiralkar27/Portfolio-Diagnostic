# scoring_v3.py
FACTORS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
WEIGHTS = {
    "F1": 0.25, "F2": 0.15, "F3": 0.15, "F4": 0.10, "F5": 0.10,
    "F6": 0.05, "F7": 0.05, "F8": 0.05, "F9": 0.05, "F10": 0.05
}
FACTOR_NAMES = {
    "F1": "Earnings & Growth",
    "F2": "Ownership & Governance",
    "F3": "Catalysts & Events",
    "F4": "Leadership & Management",
    "F5": "Valuation",
    "F6": "Financial Health",
    "F7": "Historical Return",
    "F8": "Risk",
    "F9": "Liquidity",
    "F10": "Sector / Policy"
}
HORIZON_WEIGHTS = {
    "base": WEIGHTS,
    "long": {
        "F1": 0.35, "F2": 0.20, "F3": 0.05, "F4": 0.10, "F5": 0.05,
        "F6": 0.05, "F7": 0.05, "F8": 0.05, "F9": 0.05, "F10": 0.05
    },
    "medium": {
        "F1": 0.15, "F2": 0.10, "F3": 0.25, "F4": 0.05, "F5": 0.10,
        "F6": 0.05, "F7": 0.15, "F8": 0.05, "F9": 0.05, "F10": 0.05
    }
}
