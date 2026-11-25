"""SYNTHETIC test data in the TDBRAIN (BIDS/BrainVision) and Mumtaz (EDF) layouts.

Used only for tests, CI and the smoke pipeline when the licensed data are not present.
Results computed on synthetic data are meaningless for the research question and are
labelled `SYNTHETIC` everywhere (dashboard banner, report header).

The generator plants a weak class effect (alpha/theta power shift) and occasional
high-amplitude artefacts so that rejection, QC and the leakage checks can be exercised.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TDBRAIN_EEG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
               "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
TDBRAIN_EXTRA = ["VPVA", "VNVB", "HPHL", "HNHR", "Erbs", "OrbOcc", "Mass"]
MUMTAZ_EEG = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz", "Fp2", "F4", "C4", "P4", "O2", "F8", "T4",
              "T6", "Cz", "Pz"]


def _posterior_weight(ch: str) -> float:
    """Eyes-closed alpha is strongest posteriorly."""
    if ch.startswith("Fp"):
        return 0.1
    return {"O": 1.0, "P": 0.7, "C": 0.4, "T": 0.4, "F": 0.2}.get(ch[0].upper(), 0.3)


def synth_signal(chs, sfreq, dur_s, label, rng, effect=1.0, ec=True, site_gain=1.0, artefact_rate=0.03):
    n = int(sfreq * dur_s)
    t = np.arange(n) / sfreq
    # pink-ish background noise per channel
    white = rng.standard_normal((len(chs), n))
    f = np.fft.rfftfreq(n, 1 / sfreq)
    spec = np.fft.rfft(white, axis=1) / np.maximum(f, 1.0) ** 0.8
    x = np.fft.irfft(spec, n=n, axis=1)
    x = x / x.std(axis=1, keepdims=True) * 8.0
    # realistic between-subject variability: per-person global, per-band and per-channel gains
    person_gain = np.exp(rng.normal(0, 0.3))
    band_gain = {b: np.exp(rng.normal(0, 0.35)) for b in ("alpha", "theta")}
    chan_gain = np.exp(rng.normal(0, 0.15, len(chs)))
    x = x * person_gain * chan_gain[:, None]
    alpha_f = rng.uniform(8.5, 11.5)
    for i, ch in enumerate(chs):
        w = _posterior_weight(ch) * person_gain * chan_gain[i]
        amp_a = (12.0 if ec else 4.0) * w * band_gain["alpha"] * (1.0 - 0.15 * effect * label)
        if ch in ("F3", "F7"):
            amp_a *= 1.0 + 0.25 * effect * label  # planted frontal asymmetry
        amp_t = 3.0 * person_gain * chan_gain[i] * band_gain["theta"] * (1.0 + 0.4 * effect * label)
        phase = rng.uniform(0, 2 * np.pi)
        env = 1.0 + 0.3 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6))
        x[i] += amp_a * env * np.sin(2 * np.pi * alpha_f * t + phase)
        x[i] += amp_t * np.sin(2 * np.pi * rng.uniform(5, 7) * t + rng.uniform(0, 6))
        x[i] += 2.0 * np.sin(2 * np.pi * 50.0 * t)  # mains
    # artefacts: blinks on frontal channels, big spikes
    n_art = rng.poisson(artefact_rate * dur_s)
    for _ in range(n_art):
        s = rng.integers(0, n - int(sfreq * 0.5))
        L = int(sfreq * 0.4)
        for i, ch in enumerate(chs):
            if ch.startswith("Fp") or ch.startswith("F"):
                x[i, s:s + L] += 250 * np.hanning(L)
    return (x * site_gain).astype(np.float64)  # microvolts


def _write_brainvision(path: Path, data_uv: np.ndarray, sfreq: float, ch_names: list[str]) -> None:
    from pybv import write_brainvision

    path.parent.mkdir(parents=True, exist_ok=True)
    write_brainvision(data=data_uv * 1e-6, sfreq=sfreq, ch_names=ch_names, fname_base=path.stem,
                      folder_out=str(path.parent), unit="µV", overwrite=True)


def make_tdbrain(out: Path, n_per_group: dict | None = None, dur_s: float = 120.0, seed: int = 0,
                 effect: float = 1.0, sfreq: float = 500.0) -> Path:
    rng = np.random.default_rng(seed)
    n_per_group = n_per_group or {"MDD": 40, "HEALTHY": 30, "ADHD": 12, "OCD": 6, "SMC": 6, "UNKNOWN": 4,
                                  "REPLICATION": 4}
    rows, k = [], 0
    for grp, n in n_per_group.items():
        for _ in range(n):
            k += 1
            pid = f"sub-19{k:06d}" if grp == "REPLICATION" else f"sub-8{k:07d}"
            label = 1 if grp == "MDD" else 0
            ind = "REPLICATION" if grp == "REPLICATION" else grp
            if grp == "MDD" and rng.random() < 0.15:
                ind = "MDD/ADHD"
            n_ses = 2 if rng.random() < 0.1 else 1
            for ses in range(1, n_ses + 1):
                age = rng.uniform(16, 70)
                rows.append({"participants_ID": pid, "sessID": ses, "indication": ind,
                             "age": f"{age:.2f}".replace(".", ","), "gender": int(rng.integers(0, 2)),
                             "BDI": float(rng.integers(20, 45) if label else rng.integers(0, 10))})
                for cond in ("restEC", "restEO"):
                    x = synth_signal(TDBRAIN_EEG, sfreq, dur_s, label, rng, effect, ec=(cond == "restEC"))
                    extra = rng.standard_normal((len(TDBRAIN_EXTRA), x.shape[1])) * 20
                    fname = out / pid / f"ses-{ses}" / "eeg" / f"{pid}_ses-{ses}_task-{cond}_eeg.vhdr"
                    _write_brainvision(fname, np.vstack([x, extra]), sfreq, TDBRAIN_EEG + TDBRAIN_EXTRA)
    pd.DataFrame(rows).to_csv(out / "participants.tsv", sep="\t", index=False)
    (out / "SYNTHETIC").write_text("This folder contains SYNTHETIC data generated by src/data/synthetic.py\n")
    return out


def make_mumtaz(out: Path, n_mdd: int = 12, n_hc: int = 10, dur_s: float = 120.0, seed: int = 1,
                effect: float = 1.0, sfreq: float = 256.0) -> Path:
    import mne

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    names = [f"EEG {c}-LE" for c in MUMTAZ_EEG] + ["EEG A2-A1", "EEG 23A-23R", "EEG 24A-24R"]
    for grp, n, label in (("MDD", n_mdd, 1), ("H", n_hc, 0)):
        for s in range(1, n + 1):
            for cond in ("EC", "EO"):
                if rng.random() < 0.05:
                    continue  # some files missing, as in the real set
                x = synth_signal(MUMTAZ_EEG, sfreq, dur_s, label, rng, effect, ec=(cond == "EC"), site_gain=1.3)
                extra = rng.standard_normal((3, x.shape[1])) * 5
                info = mne.create_info(names, sfreq, "eeg")
                raw = mne.io.RawArray(np.vstack([x, extra]) * 1e-6, info, verbose="ERROR")
                mne.export.export_raw(out / f"{grp} S{s} {cond}.edf", raw, fmt="edf", overwrite=True, verbose="ERROR")
    (out / "SYNTHETIC").write_text("SYNTHETIC Mumtaz-layout data\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdbrain", type=Path, default=Path("data/synthetic/tdbrain"))
    ap.add_argument("--mumtaz", type=Path, default=Path("data/synthetic/mumtaz"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dur", type=float, default=120.0)
    a = ap.parse_args(argv)
    base = {"MDD": 40, "HEALTHY": 30, "ADHD": 12, "OCD": 6, "SMC": 6, "UNKNOWN": 4, "REPLICATION": 4}
    make_tdbrain(a.tdbrain, {k: max(2, int(v * a.scale)) for k, v in base.items()}, dur_s=a.dur)
    make_mumtaz(a.mumtaz, max(4, int(12 * a.scale)), max(4, int(10 * a.scale)), dur_s=a.dur)
    print("Synthetic data written to", a.tdbrain, a.mumtaz)


if __name__ == "__main__":
    main()
