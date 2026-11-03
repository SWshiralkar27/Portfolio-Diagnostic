"""Seams where a person is asked something between stages. NullHooks asks nothing (tests, --demo, the app's own forms
fill the data before a stage runs); TkHooks calls the Tkinter pop-ups of input_assistant.py in the original order."""
from __future__ import annotations
from pathlib import Path


class NullHooks:
    def after_load(self, lots):
        return lots

    def prompts(self, lots, evid, peers):
        return lots, evid, peers

    def cost_review(self, HD, cfg):
        """Return (HD, set of symbols whose cost the owner confirmed, {symbol: factor to apply to lot prices})."""
        return HD, set(), {}


class TkHooks(NullHooks):
    def __init__(self, ia, args, folder):
        self.ia, self.args, self.folder = ia, args, Path(folder)

    def _mode(self):
        return "off" if getattr(self.args, "no_prompt", False) else ("force" if getattr(self.args, "prompt", False) else "auto")

    def after_load(self, lots):
        self.ia.check_holdings_layout(self.args.holdings)
        return self.ia.apply_saved_buy_dates(lots, self.folder)

    def prompts(self, lots, evid, peers):
        a = self.args
        return self.ia.run_prompts(
            lots, evid, peers, folder=self.folder, mode=self._mode(),
            evidence_path=a.evidence if Path(a.evidence).exists() else self.folder / "evidence.csv",
            peers_path=a.peers if Path(a.peers).exists() else self.folder / "peers.csv")

    def cost_review(self, HD, cfg):
        return self.ia.review_costs(HD, cfg, folder=self.folder, mode=self._mode())
