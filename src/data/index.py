"""Phase 0: build the participant index (artifacts/participants.parquet) and cohort report.

Rules: first session only; REPLICATION (blinded, IDs sub-19*) and UNKNOWN are dropped from the
labelled pool but counted. UNKNOWN does NOT mean healthy.

Tasks:
  T1: MDD (1) vs HEALTHY (0)
  T2: MDD (1) vs other diagnosed patients (0), healthy excluded
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from src.data.tdbrain import (
    find_participants_file,
    find_recording,
    list_sessions,
    load_raw,
    read_participants,
    resolve_root,
    verify_local,
)
from src.utils.config import load_config
from src.utils.hashing import sha256_file
from src.utils.logging import get_logger
from src.utils.paths import PARTICIPANTS_FILE, RESULTS, ensure_dirs

log = get_logger(__name__)

MUMTAZ_19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
             "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
BLINDED = {"REPLICATION"}
NOT_A_LABEL = {"UNKNOWN", "REPLICATION", "", "NAN", "N/A", "NA"}
ALIASES = {
    "id": ["participants_ID", "participant_id", "participants_id", "TDBRAIN_ID", "ID"],
    "session": ["sessID", "session", "ses", "session_id"],
    "indication": ["indication", "Indication"],
    "diagnosis": ["diagnosis", "formal_status", "Diagnosis"],
    "age": ["age", "Age"],
    "sex": ["gender", "sex", "Gender", "Sex"],
    "bdi": ["BDI", "BDI_pre", "bdi", "BDI-II", "bdi_pre"],
}


def _col(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    for c in ALIASES[key]:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"participants file has no column for '{key}' (tried {ALIASES[key]}); columns: {list(df.columns)}")
    return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def split_labels(x) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    toks = re.split(r"[/,;+&]| and ", str(x))
    return [t.strip().upper() for t in toks if t.strip() and t.strip().upper() not in {"NAN"}]


def normalise_id(x) -> str:
    s = str(x).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def build_index(data_cfg: dict, conditions: tuple[str, ...] = ("restEC", "restEO")) -> pd.DataFrame:
    root = resolve_root(data_cfg)
    verify_local(root, "restEC")
    tsv = find_participants_file(root)
    log.info("Participants file: %s", tsv.name)
    raw = read_participants(tsv)
    cid, cind = _col(raw, "id"), _col(raw, "indication")
    cses, cdx = _col(raw, "session", False), _col(raw, "diagnosis", False)
    cage, csex, cbdi = _col(raw, "age", False), _col(raw, "sex", False), _col(raw, "bdi", False)
    df = pd.DataFrame({"participant_id": raw[cid].map(normalise_id)})
    df["session_num"] = _num(raw[cses]) if cses else 1.0
    df["indication_raw"] = raw[cind].fillna("")
    df["diagnosis_raw"] = raw[cdx].fillna("") if cdx else ""
    df["age"] = _num(raw[cage]) if cage else np.nan
    df["sex_raw"] = raw[csex] if csex else None
    df["bdi"] = _num(raw[cbdi]) if cbdi else np.nan
    n_rows = len(df)
    # first session only: lowest session number per participant (ties -> first row)
    df = df.sort_values(["participant_id", "session_num"], kind="stable").drop_duplicates("participant_id")
    inds = df["indication_raw"].map(split_labels)
    df["primary_indication"] = inds.map(lambda t: t[0] if t else "UNKNOWN")
    dx = df["diagnosis_raw"].map(split_labels)
    df["n_indications"] = [len((set(a) | set(b)) - NOT_A_LABEL) for a, b in zip(inds, dx, strict=True)]
    df["comorbid"] = df["n_indications"] > 1
    df["blinded"] = df["participant_id"].str.startswith("sub-19") | (df["primary_indication"] == "REPLICATION")
    df.loc[df["blinded"], "primary_indication"] = "REPLICATION"
    # files
    for cond in conditions:
        paths = []
        for pid, s in zip(df["participant_id"], df["session_num"], strict=True):
            sessions = list_sessions(root, pid)
            want = f"ses-{int(s)}" if pd.notna(s) else None
            ses = want if want in sessions else (sessions[0] if sessions else None)
            p = find_recording(root, pid, ses, cond) if ses else None
            paths.append(str(p) if p else None)
        df[f"path_{cond}"] = paths
        df[f"has_{cond}"] = df[f"path_{cond}"].notna()
    df["first_session"] = df["session_num"].map(lambda s: f"ses-{int(s)}" if pd.notna(s) else "ses-1")
    df.attrs["n_rows_tsv"] = n_rows
    df.attrs["participants_tsv_sha256"] = sha256_file(tsv)
    return df.reset_index(drop=True)


def assign_tasks(df: pd.DataFrame, cohort_cfg: dict) -> pd.DataFrame:
    df = df.copy()
    lab = df["primary_indication"]
    labelled = ~lab.isin(NOT_A_LABEL) & ~df["blinded"]
    age_ok = df["age"].between(cohort_cfg["age_min"], cohort_cfg["age_max"])  # missing age -> excluded
    excl_comorbid = bool(cohort_cfg.get("exclude_comorbid_mdd", False)) & df["comorbid"] & (lab == "MDD")
    base = labelled & age_ok & ~excl_comorbid & df["has_restEC"]
    df["in_age_range"] = age_ok
    df["T1_label"] = np.where(base & (lab == "MDD"), 1, np.where(base & (lab == "HEALTHY"), 0, -1))
    others = cohort_cfg.get("t2_other_groups")
    is_other = (lab != "MDD") & (lab != "HEALTHY") & labelled
    if others:
        is_other &= lab.isin([o.upper() for o in others])
    df["T2_label"] = np.where(base & (lab == "MDD"), 1, np.where(base & is_other, 0, -1))
    return df


def cohort_report(df: pd.DataFrame, root, cohort_cfg: dict) -> dict:
    def per_class(task):
        out = {}
        for y, name in ((1, "MDD"), (0, "HEALTHY" if task == "T1" else "OTHER")):
            sub = df[df[f"{task}_label"] == y]
            out[name] = {
                "n": int(len(sub)),
                "age_mean": float(sub["age"].mean()) if len(sub) else None,
                "age_sd": float(sub["age"].std()) if len(sub) > 1 else None,
                "sex_raw_counts": {str(k): int(v) for k, v in sub["sex_raw"].value_counts(dropna=False).items()},
                "comorbid": int(sub["comorbid"].sum()),
                "groups": {str(k): int(v) for k, v in sub["primary_indication"].value_counts().items()},
            }
        return out

    ch_list = None
    first = df.loc[df["has_restEC"], "path_restEC"]
    if len(first):
        ch_list = load_raw(first.iloc[0], preload=False).ch_names
    renamed = [{"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(c, c) for c in (ch_list or [])]
    missing19 = [c for c in MUMTAZ_19 if c not in renamed]
    rep = {
        "n_rows_participants_tsv": int(df.attrs.get("n_rows_tsv", len(df))),
        "n_participants_first_session": int(len(df)),
        "n_with_restEC": int(df["has_restEC"].sum()),
        "n_with_restEO": int(df.get("has_restEO", pd.Series(dtype=bool)).sum()),
        "indication_counts_first_session": {str(k): int(v) for k, v in df["primary_indication"].value_counts().items()},
        "dropped_replication_blinded": int(df["blinded"].sum()),
        "dropped_unknown": int((df["primary_indication"] == "UNKNOWN").sum()),
        "outside_age_range": int((~df["in_age_range"]).sum()),
        "cohort_rules": cohort_cfg,
        "T1": per_class("T1"),
        "T2": per_class("T2"),
        "channels_one_restEC_file": ch_list,
        "mumtaz_19_missing_in_tdbrain": missing19,
        "participants_tsv_sha256": df.attrs.get("participants_tsv_sha256"),
        "flowchart": {
            "participants": int(df.attrs.get("n_participants_total", len(df))),
            "first_session": int(len(df)),
            "with_restEC": int(df["has_restEC"].sum()),
            "labelled_not_blinded": int((~df["primary_indication"].isin(NOT_A_LABEL) & df["has_restEC"]).sum()),
            "T1_final": int((df["T1_label"] >= 0).sum()),
            "T2_final": int((df["T2_label"] >= 0).sum()),
        },
    }
    n_hc = rep["T1"]["HEALTHY"]["n"]
    rep["recommend_T2_primary"] = bool(n_hc < cohort_cfg.get("min_healthy_for_t1", 50))
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    ensure_dirs()
    data_cfg = load_config("data")
    df = build_index(data_cfg, tuple(sorted(set(data_cfg["tdbrain"].get("conditions", ["restEC"])) | {"restEC", "restEO"})))
    df = assign_tasks(df, data_cfg["cohort"])
    rep = cohort_report(df, resolve_root(data_cfg), data_cfg["cohort"])
    df.attrs = {}
    df.to_parquet(PARTICIPANTS_FILE, index=False)
    (RESULTS / "cohort.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("flowchart", "indication_counts_first_session",
                                          "mumtaz_19_missing_in_tdbrain", "recommend_T2_primary")}, indent=2))
    for t in ("T1", "T2"):
        print(t, {k: (v["n"], v["age_mean"]) for k, v in rep[t].items()})
    if rep["mumtaz_19_missing_in_tdbrain"]:
        log.warning("Mumtaz channels missing from TDBRAIN: %s", rep["mumtaz_19_missing_in_tdbrain"])
    if rep["recommend_T2_primary"]:
        log.warning("HEALTHY with restEC < %d: propose making T2 the primary task.", data_cfg["cohort"]["min_healthy_for_t1"])


if __name__ == "__main__":
    main()
