"""Group-level brain figures from the cached TDBRAIN epochs (aggregate only; no individual is shown).
Relative band-power topographies, alpha-band wPLI connectivity, LR coefficient maps, EEGNet spatial patterns."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess.pipeline import load_cache, participant_epochs  # noqa: E402
from src.utils.paths import standard_montage  # noqa: E402

FIG = ROOT / "thesis" / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 9,
                     "savefig.dpi": 220})
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
REGION = {"Fp1": "L frontal", "F7": "L frontal", "F3": "L frontal", "Fp2": "R frontal", "F8": "R frontal",
          "F4": "R frontal", "Fz": "Midline", "Cz": "Midline", "Pz": "Midline", "C3": "L central", "C4": "R central",
          "T7": "L temporal", "P7": "L temporal", "T8": "R temporal", "P8": "R temporal", "P3": "L parietal",
          "P4": "R parietal", "O1": "L occipital", "O2": "R occipital"}
ORDER = ["L frontal", "R frontal", "L central", "R central", "Midline", "L temporal", "R temporal", "L parietal",
         "R parietal", "L occipital", "R occipital"]


def info19(chs):
    import mne

    info = mne.create_info(chs, 256.0, "eeg")
    info.set_montage(standard_montage())
    return info


def per_person(X, rows, sf=256.0, max_ep=30):
    """Relative band power (n_ch, 4) and alpha wPLI (n_ch, n_ch) for one participant."""
    ep = np.asarray(X[rows[:max_ep]], dtype=np.float64)  # (E, C, T)
    F = np.fft.rfft(ep * np.hanning(ep.shape[-1]), axis=-1)
    f = np.fft.rfftfreq(ep.shape[-1], 1 / sf)
    P = (np.abs(F) ** 2).mean(axis=0)  # (C, F)
    tot = P[:, (f >= 1) & (f < 30)].sum(axis=1)
    rel = np.stack([P[:, (f >= lo) & (f < hi)].sum(axis=1) / tot for lo, hi in BANDS.values()], axis=1) * 100
    a = (f >= 8) & (f < 13)
    Fa = F[:, :, a]  # (E, C, Fa)
    S = Fa[:, :, None, :] * np.conj(Fa[:, None, :, :])  # (E, C, C, Fa)
    im = S.imag
    w = np.abs(im.mean(axis=(0, 3))) / (np.abs(im).mean(axis=(0, 3)) + 1e-12)
    return rel, w


def main():
    sp = json.loads((ROOT / "artifacts" / "splits.json").read_text())
    labels = {k: int(v) for k, v in sp["tasks"]["T1"]["labels"].items()}
    X, idx, meta = load_cache("tdbrain", "restEC")
    rows = participant_epochs(X, idx)
    chs = meta["channels"]
    rel = {0: [], 1: []}
    wpli = {0: [], 1: []}
    for p, y in labels.items():
        if p in rows:
            r, w = per_person(X, rows[p])
            rel[y].append(r)
            wpli[y].append(w)
    n = {y: len(rel[y]) for y in (0, 1)}
    R = {y: np.mean(rel[y], axis=0) for y in (0, 1)}
    W = {y: np.mean(wpli[y], axis=0) for y in (0, 1)}
    info = info19(chs)
    import mne

    # ---- topographies: rows MDD, healthy, difference; columns bands
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6))
    for j, b in enumerate(BANDS):
        vmin = min(R[0][:, j].min(), R[1][:, j].min())
        vmax = max(R[0][:, j].max(), R[1][:, j].max())
        for i, (y, lab) in enumerate(((1, f"MDD (n = {n[1]})"), (0, f"Healthy (n = {n[0]})"))):
            im, _ = mne.viz.plot_topomap(R[y][:, j], info, axes=axes[i, j], show=False, cmap="YlGnBu",
                                         vlim=(vmin, vmax), contours=4, sensors=True)
            if j == 0:
                axes[i, j].set_ylabel(lab, fontsize=8.5)
            if i == 0:
                axes[i, j].set_title(f"{b.capitalize()} ({BANDS[b][0]}–{BANDS[b][1]} Hz)", fontsize=9)
        fig.colorbar(im, ax=axes[:2, j], shrink=0.55, pad=0.02, label="relative power (%)" if j == 3 else None)
        d = R[1][:, j] - R[0][:, j]
        v = np.abs(d).max()
        im2, _ = mne.viz.plot_topomap(d, info, axes=axes[2, j], show=False, cmap="RdBu_r", vlim=(-v, v), contours=4,
                                      sensors=True)
        fig.colorbar(im2, ax=axes[2, j], shrink=0.8, pad=0.02, label="MDD − healthy (pp)" if j == 3 else None)
    axes[2, 0].set_ylabel("Difference", fontsize=8.5)
    fig.savefig(FIG / "fig_topo_bands.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # ---- connectivity: region-averaged alpha wPLI, healthy / MDD / difference
    reg = [REGION[c] for c in chs]

    def regmat(M):
        out = np.full((len(ORDER), len(ORDER)), np.nan)
        for a_, ra in enumerate(ORDER):
            for b_, rb in enumerate(ORDER):
                ia = [i for i, r in enumerate(reg) if r == ra]
                ib = [i for i, r in enumerate(reg) if r == rb]
                vals = [M[i, k] for i in ia for k in ib if i != k]
                out[a_, b_] = np.mean(vals) if vals else np.nan
        return out

    H, Mdd = regmat(W[0]), regmat(W[1])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    vmin, vmax = np.nanmin([H, Mdd]), np.nanmax([H, Mdd])
    for ax, M, t in ((axes[0], H, f"Healthy (n = {n[0]})"), (axes[1], Mdd, f"MDD (n = {n[1]})")):
        im = ax.imshow(M, cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=9)
    fig.colorbar(im, ax=axes[:2], shrink=0.8, pad=0.02, label="mean wPLI, alpha band")
    D = Mdd - H
    v = np.nanmax(np.abs(D))
    im3 = axes[2].imshow(D, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[2].set_title("MDD − healthy", fontsize=9)
    fig.colorbar(im3, ax=axes[2], shrink=0.8, pad=0.02, label="Δ wPLI")
    for ax in axes:
        ax.set_xticks(range(len(ORDER)), ORDER, rotation=60, ha="right", fontsize=6.3)
        ax.set_yticks(range(len(ORDER)), ORDER if ax is axes[0] else [], fontsize=6.3)
    fig.savefig(FIG / "fig_conn_alpha.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # ---- LR coefficient maps per band
    fe = pd.read_parquet(ROOT / "artifacts" / "results" / "explain_lr_features.parquet")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    allv = []
    maps = {}
    for b in BANDS:
        s = fe[fe.band == b].set_index("channel")
        v = np.array([s.loc[c, "coef_mean"] if c in s.index else 0.0 for c in chs])
        stable = np.array([bool(s.loc[c, "stable"]) if c in s.index else False for c in chs])
        maps[b] = (v, stable)
        allv.append(np.abs(v).max())
    vm = max(allv)
    for ax, b in zip(axes, BANDS, strict=True):
        v, stable = maps[b]
        im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vm, vm), contours=4,
                                     mask=stable, mask_params=dict(marker="o", markerfacecolor="k", markersize=3))
        ax.set_title(b.capitalize(), fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="standardised coefficient")
    fig.savefig(FIG / "fig_topo_coef.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # ---- EEGNet spatial patterns (fold-0 model) from the exported table
    pp = ROOT / "artifacts" / "results" / "explain_eegnet_patterns.parquet"
    if pp.exists():
        pat = pd.read_parquet(pp)
        filt = sorted(pat["filter"].unique())[:8]
        fig, axes = plt.subplots(2, 4, figsize=(6.4, 3.4))
        for ax, k in zip(axes.ravel(), filt, strict=False):
            s = pat[pat["filter"] == k].groupby("channel")["value"].mean()
            v = np.array([s.get(c, 0.0) for c in chs])
            mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-1, 1), contours=3)
            ax.set_title(f"filter {k + 1}", fontsize=8)
        fig.savefig(FIG / "fig_eegnet_patterns.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    out = {"n_mdd": n[1], "n_hc": n[0],
           "alpha_rel_O_mdd": float(R[1][[chs.index("O1"), chs.index("O2")], 2].mean()),
           "alpha_rel_O_hc": float(R[0][[chs.index("O1"), chs.index("O2")], 2].mean()),
           "theta_rel_Fz_mdd": float(R[1][chs.index("Fz"), 1]), "theta_rel_Fz_hc": float(R[0][chs.index("Fz"), 1]),
           "beta_rel_mean_mdd": float(R[1][:, 3].mean()), "beta_rel_mean_hc": float(R[0][:, 3].mean()),
           "wpli_mean_mdd": float(np.nanmean(Mdd)), "wpli_mean_hc": float(np.nanmean(H))}
    (ROOT / "thesis" / "brain_values.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
