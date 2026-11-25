"""Mumtaz MDD/HC EEG (figshare doi 10.6084/m9.figshare.4244171): file indexing and label parsing.

Imported ONLY by src/eval/external.py (phase 8). Training code never touches Mumtaz.
File names look like 'MDD S1 EC.edf' or 'H S12 EO.edf'. Channels look like 'EEG Fp1-LE' (linked-ear
reference); non-EEG channels (e.g. 'EEG A2-A1', '23A-23R', '24A-24R') are dropped by the pipeline's pick.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PATTERN = re.compile(r"^(MDD|H)\s*S\s*(\d+)\s*(EC|EO|TASK)\.edf$", re.I)


def parse_name(name: str) -> dict | None:
    m = PATTERN.match(name.strip())
    if not m:
        return None
    grp, num, cond = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
    return {"participant_id": f"mumtaz-{grp}-S{num:02d}", "group": "MDD" if grp == "MDD" else "HEALTHY",
            "label": 1 if grp == "MDD" else 0, "cond": cond, "file": name}


def index_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(folder).rglob("*.edf")):
        r = parse_name(p.name)
        if r:
            rows.append({**r, "path": str(p)})
    return pd.DataFrame(rows, columns=["participant_id", "group", "label", "cond", "file", "path"])


def count_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    t = df.pivot_table(index="group", columns="cond", values="participant_id", aggfunc="nunique", fill_value=0)
    t["participants_any"] = df.groupby("group")["participant_id"].nunique()
    return t
