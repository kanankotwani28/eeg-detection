"""Write an aggregate-only copy of the dashboard inputs to dashboard/data/ so the dashboard runs from a fresh clone.

Nothing participant-level is included: predictions are left out, per-person QC values lose their IDs, and power
spectra are reduced to group means. Run after `make report`:  python -m src.report.bundle
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from src.utils.paths import RESULTS, ROOT

OUT = ROOT / "dashboard" / "data"
COPY = ["summary_table.parquet", "comparisons.parquet", "confusion.parquet", "calibration.parquet", "roc.parquet",
        "fl_history.parquet", "lira.parquet", "lira_roc.parquet", "external.parquet", "runs.parquet", "cohort.json",
        "summary.json", "explain_lr_features.parquet", "explain_lr_grouped.parquet", "explain_band_occlusion.parquet",
        "explain_eegnet_topomaps.png"]


def anonymise_qc(qc: dict) -> dict:
    out = {k: v for k, v in qc.items() if not isinstance(v, dict | list) or k == "psd_check"}
    out["epochs_kept_values"] = sorted(qc.get("epochs_kept_per_person", {}).values())
    out["rejection_rate_values"] = sorted(qc.get("rejection_rate_per_person", {}).values())
    return out


def group_psd(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["group", "freq"], as_index=False)
              .agg(mean=("log10_power", "mean"), sd=("log10_power", "std"), n=("participant_id", "nunique")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        if (RESULTS / f).exists():
            shutil.copy2(RESULTS / f, OUT / f)
    for name in ("qc.json", "qc_mumtaz_EC.json"):
        if (RESULTS / name).exists():
            (OUT / name).write_text(json.dumps(anonymise_qc(json.loads((RESULTS / name).read_text())), indent=1))
    p = RESULTS / "psd_tdbrain_restEC.parquet"
    if p.exists():
        group_psd(pd.read_parquet(p)).to_parquet(OUT / "psd_group_tdbrain_restEC.parquet", index=False)
    # guard: no participant identifiers may leave the machine
    for f in OUT.iterdir():
        if f.suffix == ".parquet":
            d = pd.read_parquet(f)
            assert "participant_id" not in d.columns, f
            assert not any(d[c].astype(str).str.contains(r"sub-\d").any() for c in d.columns), f
        elif f.suffix == ".json":
            assert "sub-" not in f.read_text(), f
    print("dashboard bundle written to", OUT, sorted(x.name for x in OUT.iterdir()))


if __name__ == "__main__":
    main()
