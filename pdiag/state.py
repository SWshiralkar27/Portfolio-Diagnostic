"""Where a run keeps what it carries to the next run: the fetched NSE evidence and last run's scores.
DiskState = files next to the outputs (command line / IDLE). MemoryState = nothing on disk; the app gives it the files the
user uploaded and takes back carry_over_csv() as downloads."""
from __future__ import annotations
import io
from pathlib import Path
import pandas as pd
from pdiag.loaders import load_evidence, evidence_from_df, read_table

SNAP_COLS = ["Symbol", "Weighted", "FinalLabel", "Review_Risk"]


class DiskState:
    def __init__(self, out, evidence_label="evidence.csv"):
        self.out = Path(out)
        self.evidence_label = evidence_label
        self.auto_path = self.out / "evidence_auto.csv"
        self.snapshot_path = self.out / ".cache" / "last_run_snapshot.csv"

    def load_auto(self):
        return load_evidence(self.auto_path) if self.auto_path.exists() else {}

    def save_auto(self, auto_evid):
        pd.DataFrame(list(auto_evid.values())).to_csv(self.auto_path, index=False)
        return f"Fetched evidence saved to {self.auto_path} (your {self.evidence_label} is never modified)"

    def has_snapshot(self):
        return self.snapshot_path.exists()

    def load_snapshot(self):
        return pd.read_csv(self.snapshot_path)

    def save_snapshot(self, df):
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.snapshot_path, index=False)


def _bytes(x):
    if x is None: return None
    if isinstance(x, str): return x.encode("utf-8")
    if hasattr(x, "read"):
        if hasattr(x, "seek"): x.seek(0)
        return x.read()
    return bytes(x)


def _file(data, name):
    f = io.BytesIO(data); f.name = name
    return f


class MemoryState:
    def __init__(self, auto_csv=None, snapshot_csv=None):
        self._auto_in, self._snap_in = _bytes(auto_csv), _bytes(snapshot_csv)
        self._auto, self._snap_out = None, None

    def load_auto(self):
        if self._auto is None:
            self._auto = evidence_from_df(read_table(_file(self._auto_in, "evidence_auto.csv"))) if self._auto_in else {}
        return dict(self._auto)

    def save_auto(self, auto_evid):
        self._auto = dict(auto_evid)
        return "Fetched evidence kept in memory for the carry-over pack (nothing is stored on the server)"

    def has_snapshot(self):
        return self._snap_in is not None

    def load_snapshot(self):
        return pd.read_csv(_file(self._snap_in, "last_run_snapshot.csv"))

    def save_snapshot(self, df):
        self._snap_out = df.copy()

    def carry_over_csv(self):
        """{file name: text} for the next run: this run's evidence cache and scores (previous ones if this run set none)."""
        out = {}
        if self._auto is not None: out["evidence_auto.csv"] = pd.DataFrame(list(self._auto.values())).to_csv(index=False)
        elif self._auto_in: out["evidence_auto.csv"] = self._auto_in.decode("utf-8")
        if self._snap_out is not None: out["last_run_snapshot.csv"] = self._snap_out.to_csv(index=False)
        elif self._snap_in: out["last_run_snapshot.csv"] = self._snap_in.decode("utf-8")
        return out
