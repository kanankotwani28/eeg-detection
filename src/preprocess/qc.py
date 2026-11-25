"""QC report: epochs kept, rejection rates, exclusions, bad-channel interpolation, PSD sanity check."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.signal import welch

from src.utils.logging import get_logger
from src.utils.paths import PARTICIPANTS_FILE, RESULTS, ensure_dirs

log = get_logger(__name__)
EXCLUSION_STOP_FRACTION = 0.30


def person_psd(X: np.ndarray, rows: np.ndarray, sfreq: float):
    f, p = welch(np.asarray(X[rows]), fs=sfreq, nperseg=int(2 * sfreq), noverlap=int(sfreq), axis=-1)
    return f, p.mean(axis=0)  # (n_ch, n_freq)


def write_qc(dataset: str, cond: str, labels: dict | None = None) -> dict:
    from src.preprocess.pipeline import load_cache, participant_epochs

    ensure_dirs()
    X, idx, meta = load_cache(dataset, cond)
    pp = meta["per_person"]
    chs, sf = meta["channels"], meta["sfreq"]
    rows = participant_epochs(X, idx)
    if labels is None and PARTICIPANTS_FILE.exists() and dataset.startswith("tdbrain"):
        df = pd.read_parquet(PARTICIPANTS_FILE)
        labels = dict(zip(df["participant_id"], df["primary_indication"], strict=True))
    alpha_ok, psd_rows = [], []
    for pid, r in rows.items():
        f, p = person_psd(X, r, sf)
        band = (f >= 8) & (f <= 13)
        a = p[:, band].mean(axis=1)
        post = np.mean([a[chs.index("O1")], a[chs.index("O2")]])
        front = np.mean([a[chs.index("Fp1")], a[chs.index("Fp2")]])
        alpha_ok.append(bool(post > front))
        keep = f <= 45
        for fi, val in zip(f[keep], np.log10(p[:, keep].mean(axis=0) + 1e-12), strict=True):
            psd_rows.append({"participant_id": pid, "group": (labels or {}).get(pid, "NA"), "freq": float(fi),
                             "log10_power": float(val)})
    n = len(pp)
    excluded = sorted(k for k, v in pp.items() if v["excluded"])
    frac_alpha = float(np.mean(alpha_ok)) if alpha_ok else float("nan")
    total = sum(v["n_total"] for v in pp.values())
    rej = sum(v["n_total"] - v["n_clean"] for v in pp.values())
    qc = {
        "dataset": dataset, "cond": cond, "n_participants": n,
        "n_excluded": len(excluded), "excluded_fraction": len(excluded) / max(n, 1), "excluded": excluded,
        "errors": {k: v["error"] for k, v in pp.items() if v.get("error")},
        "epoch_rejection_rate": rej / max(total, 1),
        "epochs_kept_per_person": {k: v["n_kept"] for k, v in pp.items()},
        "rejection_rate_per_person": {k: (v["n_total"] - v["n_clean"]) / max(v["n_total"], 1) for k, v in pp.items()},
        "n_recordings_with_interpolation": int(sum(v["n_interpolated"] > 0 for v in pp.values())),
        "n_channels_interpolated_total": int(sum(v["n_interpolated"] for v in pp.values())),
        "psd_check": {"fraction_EC_alpha_O_gt_Fp": frac_alpha, "passes": bool(frac_alpha > 0.5)},
        "stop_excluded_over_30pct": len(excluded) / max(n, 1) > EXCLUSION_STOP_FRACTION,
        "shape": meta["shape"],
    }
    name = "qc.json" if (dataset, cond) == ("tdbrain", "restEC") else f"qc_{dataset}_{cond}.json"
    (RESULTS / name).write_text(json.dumps(qc, indent=1))
    pd.DataFrame(psd_rows).to_parquet(RESULTS / f"psd_{dataset}_{cond}.parquet", index=False)
    if cond == "restEC" and not qc["psd_check"]["passes"]:
        log.warning("PSD sanity check FAILED: posterior alpha > frontal alpha in only %.0f%% of participants",
                    100 * frac_alpha)
    if qc["stop_excluded_over_30pct"]:
        log.warning("STOP: preprocessing excluded %.0f%% of participants (>30%%). Confirm reject_ptp_uv and "
                    "min_epochs_per_person with your EEG expert.", 100 * qc["excluded_fraction"])
    log.info("QC %s/%s: excluded %d/%d, epoch rejection %.1f%%, alpha check %.0f%%", dataset, cond, len(excluded), n,
             100 * qc["epoch_rejection_rate"], 100 * frac_alpha)
    return qc
